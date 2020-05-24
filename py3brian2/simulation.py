from collections import OrderedDict
from pathlib import Path
import logging
import numpy as np

import mnist
from py3brian2.model_params import *

PARENT_PATH = Path(__file__).resolve().parent
WEIGHT_PATH = PARENT_PATH / 'weights'
DEFAULT_INPUT_INTENSITY = 2.  # default intensity of input images

logger = logging.getLogger(__name__)


def _normalize_weights(synapses, norm):
    weights = np.reshape(
        synapses.w, (len(synapses.source), len(synapses.target))
    )
    col_sums = weights.sum(axis=0)
    ok = col_sums > 0
    col_factors = np.ones_like(col_sums)
    col_factors[ok] = norm / col_sums[ok]
    weights *= col_factors
    synapses.w = weights.flatten()


def create_network(test_mode):
    ### Neurons
    inh = b.NeuronGroup(n_neurons, neuron_eqs_i, name='Ai', method='euler',
                        threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i)
    inh.v = v_rest_i - 40. * b.mV

    if test_mode:
        exc = b.NeuronGroup(n_neurons, neuron_eqs_e_test, name='Ae', method='euler',
                            threshold=v_thresh_e, refractory=refrac_e, reset=reset_eqs_e_test)
        exc_theta = np.load(WEIGHT_PATH / 'theta_A.npy') * b.volt
    else:
        exc = b.NeuronGroup(n_neurons, neuron_eqs_e, name='Ae', method='euler',
                            threshold=v_thresh_e, refractory=refrac_e, reset=reset_eqs_e)
        exc_theta = np.ones(n_neurons) * 20.0 * b.mV
    exc.v = v_rest_e - 40. * b.mV
    exc.theta = exc_theta

    input = b.PoissonGroup(n_input, 0 * b.Hz, name='Xe')

    ### Connections. See `Diehl&Cook_MNIST_random_conn_generator.py`.
    # Excitatory to Inhibitory, one-to-one connection matrix
    exc_inh = b.Synapses(exc, inh, 'w: 1', name='AeAi', on_pre='ge_post += w')
    exc_inh.connect(j='i')
    exc_inh.w = 10.4

    # Inhibitory to Excitatory, zero diagonal matrix
    inh_exc = b.Synapses(inh, exc, 'w: 1', name='AiAe', on_pre='gi_post += w')
    inh_exc.connect()
    zero_diag = np.ones((n_neurons, n_neurons)) - np.diag(np.ones(n_neurons))
    inh_exc.w = 17 * zero_diag.flatten()

    # Input to Excitatory, random matrix + STDP
    model = 'w : 1'
    pre = 'ge_post += w'
    post = ''
    if not test_mode:
        model += eqs_stdp_ee
        pre += '; ' + eqs_stdp_pre_ee
        post = eqs_stdp_post_ee
    input_exc = b.Synapses(input, exc, model=model, name='XeAe', on_pre=pre, on_post=post)
    input_exc.connect()
    input_exc.delay = 'rand() * 10 * ms'
    input_exc.w = .3 * (np.random.random((n_input, n_neurons)) + 0.01).flatten()

    net = b.Network()
    net.add(exc, inh, input)
    net.add(exc_inh, inh_exc, input_exc)
    net.add(b.SpikeMonitor(exc, name='monitorAe'))
    return net


@b.network_operation(dt=single_example_time, when='end')
def update_input_image(t):
    if is_resting:
        clock = t.group
        logger.debug('resting time {}'.format(clock.t))
        net['Xe'].rates = 0 * b.Hz
    else:
        net['Xe'].rates = b.Hz * images[input_img_idx % len(labels), :, :].reshape(n_input) / 8. * input_intensity
        if not test_mode:
            _normalize_weights(net['XeAe'], 78.)


@b.network_operation(dt=single_example_time, when='start')
def check_spike_count(t):
    global is_resting, input_img_idx, input_intensity, accumulated_spike_count
    clock = t.group
    if clock.t < single_example_time:
        # skip first run because there was not any input yet
        return
    if is_resting:
        # skip because we were resting the previous dt
        is_resting = False
        return
    if input_img_idx >= num_examples:
        logger.info('All images have been processed. Stopping the simulation after {}'.format(clock.t))
        net.stop()
    last_spike_count = net['monitorAe'].count - accumulated_spike_count
    if np.sum(last_spike_count) < 5:
        logger.debug('spike count < 5 {}'.format(t.group.t))
        input_intensity += 1
        is_resting = True
    else:
        input_intensity = DEFAULT_INPUT_INTENSITY
        label = labels[input_img_idx]
        labels_spike_count_map[label] += last_spike_count
        input_img_idx += 1
        is_resting = True
    accumulated_spike_count += last_spike_count


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    test_mode = False
    logger.info('preparing data')
    if test_mode:
        images, labels = mnist.test_images(), mnist.test_labels()
        num_examples = len(labels)
    else:
        images, labels = mnist.train_images(), mnist.train_labels()
        num_examples = 3 * len(labels)

    is_resting = False  # whether the system is in the resting state
    input_img_idx = 0  # index of the currently processed input image
    accumulated_spike_count = np.zeros(n_neurons)  # accumulated spikes counter, number of spikes per neuron index
    input_intensity = DEFAULT_INPUT_INTENSITY  # intensity of input images
    # 10 because we have images of 0,...,9
    labels_spike_count_map = OrderedDict({label: np.zeros(n_neurons) for label in range(10)})

    net = create_network(test_mode)
    net.add(update_input_image, check_spike_count)
    logger.info('running simulation')
    # we don't know the exact time of simulation because one image might run multiple times.
    # that is why we run the network `time_reserve` times more and just stop the network
    # from `check_spike_count` when it reached the last image.
    time_reserve = 10
    # we use 2 * single_example_time because 1 single_example_time for running and 1 single_example_time for resting
    net.run(time_reserve * num_examples * 2 * single_example_time)

    # assigning excitatory neurons to the labels they were most active to
    neuron_to_labels = {}
    for neuron in range(len(net['Ae'])):
        neuron_spike_counts = [spike_counts[neuron] for _, spike_counts in labels_spike_count_map.items()]
        # labels are implicitly indices of `neuron_spike_counts`
        labels = [label for label, spike_counts in enumerate(neuron_spike_counts)
                  if spike_counts == max(neuron_spike_counts)]
        if len(labels) == 1:
            # classify only neurons that have a single label only
            neuron_to_labels[neuron] = labels[0]
    print(neuron_to_labels)
