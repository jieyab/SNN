from brian2 import *


def fusi_landmark(PI_Neurons, Landmark):

    plastic_syn_eq_landmark = '''

                    dCa/dt = (-Ca/tau_ca) : 1 (clock-driven)                     #Calcium Potential
                    updrift = 1.0*(w_fusi_landmark>theta_w) : 1
                    downdrift = 1.0*(w_fusi_landmark<=theta_w) : 1
                    dw/dt = (alpha*updrift)-(beta*downdrift) : 1 (clock-driven) # internal weight variable

                    w_fusi_landmark:1
                    wplus: 1 (shared)
                    wminus: 1  (shared)
                    theta_upl: 1  (shared, constant)
                    theta_uph: 1  (shared, constant)
                    theta_downh: 1 (shared, constant)
                    theta_downl: 1  (shared, constant)
                    alpha: 1 /second  (shared, constant)
                    beta: 1 /second   (shared, constant)
                    tau_ca: second  (shared, constant)
                    w_min: 1  (shared, constant)
                    w_max: 1   (shared, constant)
                    theta_w: 1   (shared, constant)
                    w_ca: 1       (shared, constant)                                      # Calcium weight


    '''

    preEq_landmark = '''
            up = 1. * (Ca>theta_upl) * (Ca<theta_uph)
            down = 1. *  (Ca>theta_downl) * (Ca<theta_downh)
            w_fusi_landmark += wplus * up - wminus * down
            w_fusi_landmark = clip(w_fusi_landmark,w_min,w_max)
            '''

    postEq_landmark = '''Ca += w_ca'''

    fusi_landmark_synapse = Synapses(PI_Neurons, Landmark, plastic_syn_eq_landmark, on_pre=preEq_landmark, on_post=postEq_landmark)
    fusi_landmark_synapse.connect()
    fusi_landmark_synapse.wplus = 0.5
    fusi_landmark_synapse.wminus = 0.3
    fusi_landmark_synapse.theta_upl = 1.1
    fusi_landmark_synapse.theta_uph = 10
    fusi_landmark_synapse.theta_downh = 1
    fusi_landmark_synapse.theta_downl = 0
    fusi_landmark_synapse.alpha = 0.1 * 1 / second
    fusi_landmark_synapse.beta = 0.1 * 1 / second
    fusi_landmark_synapse.tau_ca = 4 * ms
    fusi_landmark_synapse.w_ca = 1
    fusi_landmark_synapse.w_min = 0
    fusi_landmark_synapse.w_max = 1
    fusi_landmark_synapse.theta_w = 0.5

    return fusi_landmark_synapse