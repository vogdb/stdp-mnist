import brian2 as b

n_input = 784
n_neurons = 400
single_example_time = 0.35 * b.second
resting_time = 0.15 * b.second

v_rest_e = -65. * b.mV
v_rest_i = -60. * b.mV
v_reset_e = -65. * b.mV
v_reset_i = -45. * b.mV
# v_thresh_e = -52. * b.mV
v_thresh_i = -40. * b.mV
refrac_e = 5. * b.ms
refrac_i = 2. * b.ms

tc_pre_ee = 20 * b.ms
tc_post_1_ee = 20 * b.ms
tc_post_2_ee = 40 * b.ms
nu_ee_pre = 0.0001  # learning rate
nu_ee_post = 0.01  # learning rate
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

tc_theta = 1e7 * b.ms
theta_plus_e = 0.05 * b.mV
reset_eqs_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
reset_eqs_e_test = 'v = v_reset_e; timer = 0*ms'

offset = 20.0 * b.mV
v_thresh_e = '(v>(theta - offset - 52*mV)) * (timer>refrac_e)'
neuron_eqs_e_base = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *         -v                               : amp
        I_synI = gi * nS * (-100.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        dtimer/dt = 0.1                                           : second        
        '''
neuron_eqs_e = neuron_eqs_e_base + \
               '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e_test = neuron_eqs_e_base + \
                    '\n  theta      :volt'
neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0
            '''
eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'
