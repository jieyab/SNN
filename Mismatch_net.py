from CD_net import CD_net
from brian2 import *
import numpy as np
from aux_function import gaussian_spike


def mismatch_net(Non_landmark_neuron, Estimated_landmark_neuron, Landmark):
    NonEst_landmark_poisson = PoissonGroup(1, rates=700 * Hz, name="NonEstimated_landmark_poisson")
    eqs = '''dv/dt = (-v) / (10*ms) : 1'''
    ##--------------------one non estimated landmark neurons----------------------##
    Non_estimatedlandmark_neuron = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear',
                                               name='Non_estimatedlandmark_neuron')

    ##--------------------two mismatch neurons----------------------##
    Mismatch_neuron = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear', name='Mismatch_neuron')
    Non_Mismatch_neuron = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear', name='Non_Mismatch_neuron')

    ########## -------- synapses -----------######
    # synapse btw nonestimated landmark poisson and non estimate landmark neuron
    Nonestimatedlandmark_poi_synapse = Synapses(NonEst_landmark_poisson, Non_estimatedlandmark_neuron, 'w_poi_nonest:1',
                                                on_pre='v_post+=w_poi_nonest', name='Nonestlandmark_poi_synapse')
    Nonestimatedlandmark_poi_synapse.connect()
    Nonestimatedlandmark_poi_synapse.w_poi_nonest = 3

    # synapses from non estimated landmark neuron to mismatch neurons (inhibitory)
    NonEst_mistmatch_inh_synapse = Synapses(Non_estimatedlandmark_neuron, Mismatch_neuron, 'w_nonest_mis_inh:1',
                                            on_pre='v_post+=w_nonest_mis_inh', name='nonest_mismatch_inh_synapses')
    NonEst_mistmatch_inh_synapse.connect()
    NonEst_mistmatch_inh_synapse.w_nonest_mis_inh = -1

    # synapse  btw estimated landmark neurons and no estomated landmark neurons
    Est_nonest_inh_synapse = Synapses(Estimated_landmark_neuron, Non_estimatedlandmark_neuron, 'w_est_nonest_inh:1',
                                      on_pre='v_post+=w_est_nonest_inh', name='est_nonest_inh_synapse')
    Est_nonest_inh_synapse.connect()
    Est_nonest_inh_synapse.w_est_nonest_inh = -3

    # synapses between non collision neurons and 2 mismatch neurons (excitory)
    NonCol_mismatch_synapse = Synapses(Non_landmark_neuron, Mismatch_neuron, 'w_noncol_mis:1',
                                       on_pre='v_post+=w_noncol_mis', name='NonCol_mis_synap')
    NonCol_mismatch_synapse.connect()
    NonCol_mismatch_synapse.w_noncol_mis = 0.5

    NonCol_nonmismatch_synapse = Synapses(Non_landmark_neuron, Non_Mismatch_neuron, 'w_noncol_nonmis:1',
                                          on_pre='v_post+=w_noncol_nonmis', name='NonCol_nonmis_synap')
    NonCol_nonmismatch_synapse.connect()
    NonCol_nonmismatch_synapse.w_noncol_nonmis = 1

    # synapses from  estimated landmark neuron to mismatch (ex) and non mismatch(inh)
    Est_mismatch_ex_synapse = Synapses(Estimated_landmark_neuron, Mismatch_neuron, 'w_est_mis_ex:1',
                                       on_pre="v_post+=w_est_mis_ex", name='estimatedlandmark_mismatch_ex_synapse')
    Est_mismatch_ex_synapse.connect()
    Est_mismatch_ex_synapse.w_est_mis_ex = 0.8

    Est_nonmismatch_inh_synapses = Synapses(Estimated_landmark_neuron, Non_Mismatch_neuron, 'w_est_nonmis_inh:1',
                                            on_pre='v_post+=w_est_nonmis_inh',
                                            name='estimatedlandmark_nonmismatch_inh_synapse')
    Est_nonmismatch_inh_synapses.connect()
    Est_nonmismatch_inh_synapses.w_est_nonmis_inh = -2

    ## synapses from landmark to mismatch neuron
    landmark_mismatch_inh_synapse = Synapses(Landmark, Mismatch_neuron, 'w_landmark_mis_inh:1',
                                             on_pre="v_post+=w_landmark_mis_inh", name='landmark_mismatch_inh_synapse')
    landmark_mismatch_inh_synapse.connect()
    landmark_mismatch_inh_synapse.w_landmark_mis_inh = -10

    # inh synapse from mismatch to estimated landmark neurons
    Mismatch_est_inh_synpase = Synapses(Mismatch_neuron, Estimated_landmark_neuron, 'w_mis_est_inh:1',
                                        on_pre='v_post+=w_mis_est_inh', name="mismatch_est_inh_synapse")
    Mismatch_est_inh_synpase.connect()
    Mismatch_est_inh_synpase.w_mis_est_inh = -1.5

    Mismatch_landmark_inh_synpase = Synapses(Mismatch_neuron, Landmark, 'w_mis_land_inh:1',
                                             on_pre='v_post+=w_mis_land_inh', name="mismatch_land_synapse")
    Mismatch_landmark_inh_synpase.connect()
    Mismatch_landmark_inh_synpase.w_mis_land_inh = -1.5

    spikemon_Mismatch = SpikeMonitor(Mismatch_neuron, variables='v', name='spikemon_mismatch')
    # statemon_Mismatch = StateMonitor(Mismatch_neuron, 'v', record=0)
    # statemon_Mismatch = spikemon_Mismatch.values('v')

    # statemon_nonest = StateMonitor()
    spikemon_nonestimatedlandmark = SpikeMonitor(Non_estimatedlandmark_neuron, name='spikemon_nonestimated_landmark')
    return Mismatch_landmark_inh_synpase, landmark_mismatch_inh_synapse, spikemon_nonestimatedlandmark, NonEst_landmark_poisson, Non_estimatedlandmark_neuron, Mismatch_neuron, Non_Mismatch_neuron, Nonestimatedlandmark_poi_synapse, Est_nonest_inh_synapse, NonCol_mismatch_synapse, NonCol_nonmismatch_synapse, NonEst_mistmatch_inh_synapse, Est_mismatch_ex_synapse, Est_nonmismatch_inh_synapses, Mismatch_est_inh_synpase, spikemon_Mismatch
    # return landmark_mismatch_inh_synapse,NonEst_landmark_poisson, Non_estimatedlandmark_neuron, Mismatch_neuron, Non_Mismatch_neuron, Nonestimatedlandmark_poi_synapse, Est_nonest_inh_synapse, NonCol_mismatch_synapse, NonCol_nonmismatch_synapse, NonEst_mistmatch_inh_synapse, Est_mismatch_ex_synapse, Est_nonmismatch_inh_synapses, Mismatch_est_inh_synpase, spikemon_Mismatch
