from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from aux_function import gaussian_spike


def HD_error_correcting(N_HD,HD_Neuron,N_speed):
    N_HD_error_neurons = N_HD * N_HD
    HD_index_list = np.arange(N_HD_error_neurons)
    #poisson group for IMU
    Poisson_IMU =  PoissonGroup(N_HD, rates=700 * Hz,name='poisson_imu')
    #equation for error network
    eqs_HD_error = '''dv/dt = (-v) / (10*ms) : 1'''

    #neuron for IMU
    IMU_Neuron = NeuronGroup(N_HD, eqs_HD_error, threshold='v>1', reset='v=0', method='linear', name='IMU_Neuron')

    ##------ defining a group of error matrix neurons-------##
    HD_errormatrix_neurons = NeuronGroup(N_HD_error_neurons, eqs_HD_error, threshold='v>1', reset='v=0', method='linear',name='error_matrix_neurons')
    # defining the subgroud of error matrix neurons that connected to IMU output
    IMU_error_connecting = [HD_errormatrix_neurons[(hde * N_HD):((hde + 1) * N_HD)] for hde in range(N_HD)]

    ##----- defining positive/negative errors neurons-------##
    HD_positive_error = NeuronGroup(N_HD, eqs_HD_error, threshold='v>1', reset='v=0', method='linear',
                                    name='HD_positive_error')
    HD_negative_error = NeuronGroup(N_HD, eqs_HD_error, threshold='v>1', reset='v=0', method='linear',
                                    name='HD_negative_error')

    #synapse between poisson IMU and IMU input neuron
    IMU_poi_synapses = Synapses(Poisson_IMU,IMU_Neuron,'w_poi_imu:1',on_pre='v_post+=w_poi_imu',name='IMU_poison_synpases')
    IMU_poi_synapses.connect(j="i")

    # connecting IMU output with error neurons
    IMU_errors_connecting_synapses = [Synapses(IMU_Neuron[hde:(hde + 1)], IMU_error_connecting[hde], 'w_imu_error:1', on_pre='v_post+=w_imu_error',
                                               name="IMU_error_connect_synapses_" + str(hde)) for hde in range(N_HD)]

    for group in range(N_HD):
        IMU_errors_connecting_synapses[group].connect()
        IMU_errors_connecting_synapses[group].w_imu_error = 2

    # connecting HD
    HD_error_connect_synapses = Synapses(HD_Neuron, HD_errormatrix_neurons, 'w_hd_error:1',
                                          on_pre='v_post+=w_hd_error',
                                          name="HD_error_connect_synapses")

    for group in range(N_HD):
        #connecting each column of neuron matrix N_HD to the corresponding HD direction(the ycol)
        for index in HD_index_list[group::N_HD]:  # here list a index of neurons which is each ycol in the 72*72 matrix e.g.0,72,144....
            # print("index",index,"connect",connect)
            HD_error_connect_synapses.connect(i=group, j=index)  # each column is connected to one IMU output value
            HD_error_connect_synapses.w_hd_error = 2

    print("don0e first loop")

    Error_negative_synapses = Synapses(HD_errormatrix_neurons, HD_negative_error, 'w_hd_neg:1',on_pre='v_post+=w_hd_neg', name='error_negative_synapses')
    Error_positive_synapses = Synapses(HD_errormatrix_neurons, HD_positive_error, 'w_hd_pos:1',on_pre='v_post+=w_hd_pos',name='error_positive_synapses')
    for ycol in range(N_HD - 1):
        xcol = 0  # the number of xcol in the error matrix
        neg_error_value_index = N_HD - 1 - ycol  # define the negative error value(according to ycol index), assuming index 0 neuron in negative error neuron group standing for -72
        if xcol <= ycol:
            error_neuron_index = ycol + xcol * (N_HD - 1)  # define the index of neuron from error matrix that will connect to negative error neuron
            Error_negative_synapses.connect(i=error_neuron_index, j=neg_error_value_index)
            Error_negative_synapses.w_hd_neg = 2
            xcol += 1

    for ycol in range(1, N_HD - 1):
        xcol = 1
        pos_error_value_index = ycol  # the positive error neuron's value is equal to the ycol index
        if xcol <= (N_HD - ycol):
            error_neuron_index = (xcol + 1) * N_HD - ycol  # the neuron index of neuron from error matrix that will connect to positive error value neuron
            Error_positive_synapses.connect(i=error_neuron_index, j=pos_error_value_index)
            Error_positive_synapses.w_hd_pos = 2
            xcol += 1


    ######-------------------excitory/inhibitory speed correcting pool-----------###
    Ex_speed_pool_Neurons = NeuronGroup(N_HD/2, eqs_HD_error, threshold='v>1', reset='v=0', method='linear', name="excitory_speed__neuron_pool")
    Inh_speed_pool_Neurons = NeuronGroup(N_HD/2, eqs_HD_error, threshold='v>1', reset='v=0', method='linear', name='inhibitory_speed_neuron_pool')


    ######-------------------from neg/pos errors to excitory/inhibitory speed correcting synapses-----------###
   # Positive_ex_pool_synapses = [Synapses(HD_positive_error[pe:(pe+1)], Ex_speed_pool_Neurons[0:(pe+1)], 'w_posi_ex:1', on_pre='v_post+=w_posi_ex', name="Positive_excitory_correct_synapses_" + str(pe)) for pe in range(int(N_HD/2))]
   #Negative_inh_pool_synapses = [Synapses(HD_negative_error[ne:(ne+1)], Inh_speed_pool_Neurons[0:(ne+1)], 'w_neg_inh:1', on_pre='v_post+=w_neg_inh', name="Negative_inhibitory_correct_synapses_" + str(ne)) for ne in range(int(N_HD/2))]
    Positive_ex_pool_synapses = [Synapses(HD_positive_error[(pe):(pe+1)], Ex_speed_pool_Neurons[0:(int(N_HD/2)-pe)], 'w_posi_ex:1', on_pre='v_post+=w_posi_ex', name="Positive_excitory_correct_synapses_" + str(pe)) for pe in range(int(N_HD/2))]
    Negative_inh_pool_synapses = [Synapses(HD_negative_error[(ne):(ne+1)], Inh_speed_pool_Neurons[0:(int(N_HD/2)-ne)], 'w_neg_inh:1',on_pre='v_post+=w_neg_inh',name="Negative_inhibitory_correct_synapses_" + str(ne)) for ne in range(int(N_HD/2))]
    for group in range(int(N_HD/2)):
        Positive_ex_pool_synapses[group].connect()
        Positive_ex_pool_synapses[group].w_posi_ex = 1
        Negative_inh_pool_synapses[group].connect()
        Negative_inh_pool_synapses[group].w_neg_inh = 1

    #connecting the head error pool to Clockwise and countetclockwise neurons
    CW_neuron = NeuronGroup(N_speed, eqs_HD_error, threshold='v>1', reset='v=0', method='exact', name='CW_Neuron')
    CCW_neuron = NeuronGroup(N_speed, eqs_HD_error, threshold='v>1', reset='v=0', method='exact', name='CCW_Neuron')
    pospool_cw_synapses = Synapses(Ex_speed_pool_Neurons,CW_neuron,on_pre='v_post+=0.4',name='pospool_cw_synapses')
    negpool_ccw_synapses = Synapses(Inh_speed_pool_Neurons,CCW_neuron,on_pre='v_post+=0.4',name='negpool_ccw_synapses')
    pospool_cw_synapses.connect()
    negpool_ccw_synapses.connect()

    #IMU_poi_synapses.connect(j="i")


    statemon_positiveHD_error = StateMonitor(HD_positive_error,'v',record=[0,35])
    statemon_negativeHD_error = StateMonitor(HD_negative_error,'v',record=[0,35])
    spikemon_positiveHD_error = SpikeMonitor(HD_positive_error,name='posHDerrorspike')
    spikemon_negativeHD_error = SpikeMonitor(HD_negative_error,name='negHDerrorspike')
    spiketrain_pos = spikemon_positiveHD_error.spike_trains()

    return pospool_cw_synapses,negpool_ccw_synapses,CW_neuron,CCW_neuron,Poisson_IMU,IMU_Neuron,HD_errormatrix_neurons,HD_positive_error,HD_negative_error,IMU_poi_synapses,IMU_errors_connecting_synapses,HD_error_connect_synapses,Error_negative_synapses,Error_positive_synapses, Ex_speed_pool_Neurons,Inh_speed_pool_Neurons,Positive_ex_pool_synapses,Negative_inh_pool_synapses,statemon_positiveHD_error,statemon_negativeHD_error,spikemon_positiveHD_error,spikemon_negativeHD_error,spiketrain_pos






