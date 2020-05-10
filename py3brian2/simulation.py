from pathlib import Path
import numpy as np
import brian2 as b
from py3brian2.model_params import *

PARENT_PATH = Path(__file__).resolve().parent
WEIGHT_PATH = PARENT_PATH / 'weights'


def create_network(test_mode=False):
    # Neurons
    inh = b.NeuronGroup(n_neurons, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, name='Ai')
    inh.v = v_rest_i - 40. * b.mV

    if test_mode:
        exc = b.NeuronGroup(
            n_neurons, neuron_eqs_e_test, name='Ae', threshold=v_thresh_e, refractory=refrac_e, reset=reset_eqs_e_test)
        exc_theta = np.load(WEIGHT_PATH / 'theta_A.npy') * b.volt
    else:
        exc = b.NeuronGroup(n_neurons, neuron_eqs_e, name='Ae', threshold=v_thresh_e, refractory=refrac_e,
                            reset=reset_eqs_e)
        exc_theta = np.ones(n_neurons) * 20.0 * b.mV
    exc.v = v_rest_e - 40. * b.mV
    exc.theta = exc_theta

    input = b.PoissonGroup(n_input, 0 * b.Hz, name='X')

    # Connections. See `Diehl&Cook_MNIST_random_conn_generator.py`.

    # Excitatory to Inhibitory, one-to-one connection matrix
    exc_inh = b.Synapses(exc, inh, 'w: siemens', on_pre='ge_post += w')
    exc_inh.connect(j='i')
    exc_inh.w = 10.4 * b.nS

    # Inhibitory to Excitatory, zero diagonal matrix
    inh_exc = b.Synapses(inh, exc, 'w: siemens', on_pre='gi_post += w')
    inh_exc.connect()
    zero_diag = np.ones((n_neurons, n_neurons)) - np.diag(np.ones(n_neurons))
    inh_exc.w = 17 * b.nS * zero_diag.flatten()

    # Input to Excitatory, random matrix
    model = 'w : siemens'
    pre = 'ge_post += w'
    post = ''
    if not test_mode:
        model += eqs_stdp_ee
        pre += '; ' + eqs_stdp_pre_ee
        post = eqs_stdp_post_ee
    inh_exc = b.Synapses(input, exc, model=model, on_pre=pre, on_post=post)
    inh_exc.connect()
    inh_exc.delay = 'rand() * 10 * ms'
    inh_exc.w = .3 * b.nS * (np.random.random((n_input, n_neurons)) + 0.01).flatten()

    net = b.Network()
    net.add(exc, inh, input)
    net.add(exc_inh, inh_exc)
    return net


if __name__ == '__main__':
    create_network(False)
