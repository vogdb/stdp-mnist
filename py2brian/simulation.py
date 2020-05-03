import os
import numpy as np
from tqdm import tqdm
import mnist
import matplotlib.cm as cmap
from py2brian.model_params import *

# brian preferences are not actual anymore

np.random.seed(0)
set_global_preferences(
    # The default clock to use if none is provided or defined in any enclosing scope.
    defaultclock=Clock(dt=0.5 * ms),
    useweave=True,  # Defines whether or not functions should use inlined compiled C code where defined.
    gcc_options=['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler.
    # For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on
    usecodegen=True,  # Whether or not to use experimental code generation support.
    usecodegenweave=True,  # Whether or not to use C with experimental code generation support.
    usecodegenstateupdate=True,  # Whether or not to use experimental code generation support on state updaters.
    usecodegenthreshold=False,  # Whether or not to use experimental code generation support on thresholds.
    usenewpropagate=True,  # Whether or not to use experimental new C propagation functions.
    usecstdp=True,  # Whether or not to use experimental new C STDP.
)

# some paths for pretrained weights
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(PARENT_PATH, 'weights')
RANDOM_PATH = os.path.join(PARENT_PATH, 'random')


# ion()


def get_matrix_from_file(fileName):
    offset = 4
    if fileName[-4 - offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3 - offset] == 'e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1 - offset] == 'e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]
    return value_arr


def normalize_weights(connection):
    connection = connection[:]
    temp_conn = np.copy(connection)
    colSums = np.sum(temp_conn, axis=0)
    colFactors = 78. / colSums
    for j in xrange(n_e):  #
        connection[:, j] *= colFactors[j]


def get_2d_input_exc_weights(input_exc):
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = input_exc[:]
    weight_matrix = np.copy(connMatrix)

    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
            rearranged_weights[i * n_in_sqrt: (i + 1) * n_in_sqrt, j * n_in_sqrt: (j + 1) * n_in_sqrt] = \
                weight_matrix[:, i + j * n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_exc_weights(input_exc, fig_num=0):
    weights = get_2d_input_exc_weights(input_exc)
    fig = figure(fig_num, figsize=(18, 18))
    img = imshow(weights, interpolation="nearest", vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    colorbar(img)
    title('weights of input to exc connection')
    fig.canvas.draw()
    return img, fig


def update_2d_input_exc_weights(input_exc, img, fig):
    weights = get_2d_input_exc_weights(input_exc)
    img.set_array(weights)
    fig.canvas.draw()
    return img


def create_neurons(test_mode=False):
    # neurons
    if test_mode:
        exc = NeuronGroup(n_e, neuron_eqs_e_test, threshold=v_thresh_e, refractory=refrac_e, reset=reset_eqs_e_test)
        exc_theta = np.ones(n_e) * 20.0 * mV
    else:
        exc = NeuronGroup(n_e, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=reset_eqs_e)
        exc_theta = np.load(os.path.join(WEIGHT_PATH, 'theta_A.npy')) * volt
    exc.v = v_rest_e - 40. * mV
    exc.theta = exc_theta

    inh = NeuronGroup(n_i, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, )
    inh.v = v_rest_i - 40. * mV

    input = PoissonGroup(n_input, 0)

    return exc, inh, input


def connect_neurons(exc, inh, input, test_mode=False):
    conn_structure = 'dense'
    exc_inh_weights = get_matrix_from_file(os.path.join(RANDOM_PATH, 'AeAi.npy'))
    exc_inh = Connection(exc, inh, structure=conn_structure, state='ge')
    exc_inh.connect(exc, inh, exc_inh_weights)

    inh_exc_weights = get_matrix_from_file(os.path.join(RANDOM_PATH, 'AiAe.npy'))
    inh_exc = Connection(inh, exc, structure=conn_structure, state='gi')
    inh_exc.connect(inh, exc, inh_exc_weights)
    # Should we consider here the case: ee_STDP_on + 'ee' in recurrent_conn_names?

    input_exc_weights = get_matrix_from_file(os.path.join(WEIGHT_PATH, 'XeAe.npy'))
    input_exc = Connection(input, exc, structure=conn_structure,
                           state='ge', delay=True, max_delay=10 * ms)
    input_exc.connect(input, exc, input_exc_weights)  # delay=(0*ms,10*ms)
    if not test_mode:
        input_exc_stdp = STDP(
            input_exc, eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)
    return exc_inh, inh_exc, input_exc


def monitor_neurons(exc, inh, input):
    monitor_bin = (single_example_time + resting_time) / second
    exc_rate_monitor = PopulationRateMonitor(exc, bin=monitor_bin)
    inh_rate_monitor = PopulationRateMonitor(inh, bin=monitor_bin)
    input_rate_monitor = PopulationRateMonitor(input, bin=monitor_bin)

    exc_spike_monitor = SpikeMonitor(exc)
    inh_spike_monitor = SpikeMonitor(inh)


def plot_neurons_monitors(fig_num, exc_spike_monitor, inh_spike_monitor):
    figure(fig_num)
    ion()
    subplot(211)
    raster_plot(exc_spike_monitor, refresh=1000 * ms, showlast=1000 * ms)
    subplot(212)
    raster_plot(inh_spike_monitor, refresh=1000 * ms, showlast=1000 * ms)


def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments


def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


def save_weights(exc, input_exc):
    print 'save connections'
    np.save(os.path.join(WEIGHT_PATH, 'theta_A.npy'), exc.theta)
    input_exc_weights = input_exc[:]
    connListSparse = ([(i, j, input_exc_weights[i, j])
                       for i in xrange(input_exc_weights.shape[0]) for j in xrange(input_exc_weights.shape[1])])
    np.save(os.path.join(WEIGHT_PATH, 'XeAe.npy'), connListSparse)


def run_simulation(test_mode=False):
    exc, inh, input = create_neurons(test_mode)
    exc_inh, inh_exc, input_exc = connect_neurons(exc, inh, input, test_mode)
    print 'preparing data'
    if test_mode:
        data = {'x': mnist.test_images(), 'y': mnist.test_labels()}
        num_examples = len(data['y'])
    else:
        data = {'x': mnist.train_images(), 'y': mnist.train_labels()}
        num_examples = 3 * len(data['y'])

    input_intensity = 2.
    start_input_intensity = input_intensity
    update_interval = min(10000, num_examples)
    result_monitor = np.zeros((update_interval, n_e))
    input_numbers = [0] * num_examples
    output_numbers = np.zeros((num_examples, 10))
    assignments = np.zeros(n_e)
    previous_spike_count = np.zeros(n_e)
    exc_spike_counter = SpikeCounter(exc)

    print 'start %s' % 'testing' if test_mode else 'training'
    for j in tqdm(xrange(num_examples)):
        input.rate = data['x'][j % len(data['y']), :, :].reshape(n_input) / 8. * input_intensity
        # weights of input_exc are np.allclose between runs. Why he put it here and not after creating connection?
        if not test_mode:
            normalize_weights(input_exc)
        run(single_example_time)
        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(result_monitor[:], input_numbers[j - update_interval: j])
        current_spike_count = np.asarray(exc_spike_counter.count[:]) - previous_spike_count
        previous_spike_count = np.copy(exc_spike_counter.count[:])
        if np.sum(current_spike_count) < 5:
            input_intensity += 1.
            input.rate = 0
            run(resting_time)
        else:
            result_monitor[j % update_interval, :] = current_spike_count
            input_numbers[j] = data['y'][j % len(data['y'])][0]
            output_numbers[j, :] = get_recognized_number_ranking(assignments, result_monitor[j % update_interval, :])
            if j % 1000 == 0:
                print 'runs done:', j, 'of', int(num_examples)
                if j % update_interval == 0 and j > 0:
                    update_num = j / update_interval
                    start_num = update_num * update_interval
                    end_num = (update_num + 1) * update_interval
                    difference = output_numbers[start_num:end_num, 0] - input_numbers[start_num: end_num]
                    correct = len(np.where(difference == 0)[0])
                    performance = 1.0 * correct / update_interval * 100
                    print 'Classification performance', performance
            input.rate = 0
            run(resting_time)
            input_intensity = start_input_intensity

    if not test_mode:
        save_weights(exc, input_exc)


if __name__ == '__main__':
    run_simulation()
