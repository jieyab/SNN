from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from aux_function import gaussian_spike

def position_comparing(N_x_axis, N_y_axis,xcor_Neuron,ycor_Neuron):
    N_xcor_position_comparing = N_x_axis * N_x_axis
    xcor_index_list = arange(N_x_axis)

    # xcorrdinates poisson group for IMU
    Poisson_IMU_xcor = PoissonGroup(N_x_axis, rates=700 * Hz, name='poisson_imu_xcor')

    # equation for error network
    eqs = '''dv/dt = (-v) / (10*ms) : 1'''

    # xcor neurons for IMU
    xcor_IMU_Neuron = NeuronGroup(N_x_axis, eqs, threshold='v>1', reset='v=0', method='linear', name='IMU_xcor_Neuron')

    ##------ defining a group of xcor comparing matrix neurons-------##
    xcor_comparmatrix_neurons = NeuronGroup(N_xcor_position_comparing, eqs, threshold='v>1', reset='v=0',
                                         method='linear', name='xcor_matrix_neurons')
    # defining the subgroud of matrix neurons that connected to xcor IMU output
    xcor_IMU_error_comparing = [xcor_comparmatrix_neurons[(hde * N_x_axis):((hde + 1) * N_x_axis)] for hde in range(N_x_axis)]

    ##----- defining positive/negative errors neurons-------##

    xcor_positive_error = NeuronGroup(N_x_axis, eqs, threshold='v>1', reset='v=0', method='linear',
                                    name='xcor_positive_error')
    xcor_negative_error = NeuronGroup(N_x_axis, eqs, threshold='v>1', reset='v=0', method='linear',
                                    name='xcor_negative_error')

    # synapse between poisson IMU and IMU input neuron
    xcorIMU_poi_synapses = Synapses(Poisson_IMU_xcor, xcor_IMU_Neuron, 'w_poi_xcorimu:1', on_pre='v_post+=w_poi_xcorimu',
                                name='xcorIMU_poison_synpases')
    xcorIMU_poi_synapses.connect(j="i")

    # connecting IMU output with error neurons
    xcorIMU_errors_comparing_synapses = [
        Synapses(xcor_comparmatrix_neurons[hde:(hde + 1)], xcor_comparmatrix_neurons[hde], 'w_xcorimu_error:1', on_pre='v_post+=w_xcorimu_error',
                 name="xcorIMU_error_comparing_synapses_" + str(hde)) for hde in range(N_x_axis)]

    for group in range(N_x_axis):
        xcorIMU_errors_comparing_synapses[group].connect()
        xcorIMU_errors_comparing_synapses[group].w_xcorimu_error = 2

    # connecting HD
    xcor_error_connect_synapses = Synapses(xcor_Neuron, xcor_comparmatrix_neurons, 'w_xcor_error:1',
                                         on_pre='v_post+=w_xcor_error',
                                         name="xcor_error_connect_synapses")

    for group in range(N_x_axis):
        # connecting neuron matrix with a interval of N_HD to the HD direction(the ycol)
        for index in xcor_index_list[group::N_x_axis]:  # here list a index of neurons which is each ycol in the 72*72 matrix e.g.0,72,144....
            # print("index",index,"connect",connect)
            xcor_error_connect_synapses.connect(i=group, j=index)  # each column is connected to one IMU output value
            xcor_error_connect_synapses.w_xcor_error = 2

    print("don0e  loop")

    xcorError_negative_synapses = Synapses(xcor_comparmatrix_neurons, xcor_negative_error, 'w_xcor_neg:1',
                                       on_pre='v_post+=w_xcor_neg', name='xcor_error_negative_synapses')
    xcorError_positive_synapses = Synapses(xcor_comparmatrix_neurons, xcor_positive_error, 'w_xcor_pos:1',
                                       on_pre='v_post+=w_xcor_pos', name='xcor_error_positive_synapses')
    for ycol in range(N_x_axis - 1):
        xcol = 0  # the number of xcol in the error matrix
        neg_error_value_index = N_x_axis - 1 - ycol  # define the negative error value(according to ycol index), assuming index 0 neuron in negative error neuron group standing for -72
        if xcol <= ycol:
            error_neuron_index = ycol + xcol * (N_x_axis - 1)  # define the index of neuron from error matrix that will connect to negative error neuron
            xcorError_negative_synapses.connect(i=error_neuron_index, j=neg_error_value_index)
            xcorError_negative_synapses.w_hd_neg = 2
            xcol += 1

    for ycol in range(1, N_x_axis - 1):
        xcol = 1
        pos_error_value_index = ycol  # the positive error neuron's value is equal to the ycol index
        if xcol <= (N_x_axis - ycol):
            error_neuron_index = (xcol + 1) * N_x_axis - ycol  # the neuron index of neuron from error matrix that will connect to positive error value neuron
            xcorError_positive_synapses.connect(i=error_neuron_index, j=pos_error_value_index)
            xcorError_positive_synapses.w_hd_pos = 2
            xcol += 1


    ##### ------- y cor--------#####
    N_ycor_position_comparing = N_y_axis * N_y_axis
    ycor_index_list = arange(N_y_axis)

    # ycorrdinates poisson group for IMU
    Poisson_IMU_ycor = PoissonGroup(N_x_axis, rates=700 * Hz, name='poisson_imu_ycor')

    # equation for error network
    eqs = '''dv/dt = (-v) / (10*ms) : 1'''

    # ycor neurons for IMU
    ycor_IMU_Neuron = NeuronGroup(N_x_axis, eqs, threshold='v>1', reset='v=0', method='linear', name='IMU_ycor_Neuron')

    ##------ defining a group of ycor comparing matrix neurons-------##
    ycor_comparmatrix_neurons = NeuronGroup(N_ycor_position_comparing, eqs, threshold='v>1', reset='v=0',
                                         method='linear', name='ycor_matrix_neurons')
    # defining the subgroud of matrix neurons that connected to ycor IMU output
    ycor_IMU_error_comparing = [ycor_comparmatrix_neurons[(hde * N_x_axis):((hde + 1) * N_x_axis)] for hde in range(N_x_axis)]

    ##----- defining positive/negative errors neurons-------##

    ycor_positive_error = NeuronGroup(N_x_axis, eqs, threshold='v>1', reset='v=0', method='linear',
                                    name='ycor_positive_error')
    ycor_negative_error = NeuronGroup(N_x_axis, eqs, threshold='v>1', reset='v=0', method='linear',
                                    name='ycor_negative_error')

    # synapse between poisson IMU and IMU input neuron
    ycorIMU_poi_synapses = Synapses(Poisson_IMU, ycor_IMU_Neuron, 'w_poi_ycorimu:1', on_pre='v_post+=w_poi_ycorimu',
                                name='ycorIMU_poison_synpases')
    ycorIMU_poi_synapses.connect(j="i")

    # connecting IMU output with error neurons
    ycorIMU_errors_comparing_synapses = [
        Synapses(ycor_comparmatrix_neurons[hde:(hde + 1)], ycor_comparmatrix_neurons[hde], 'w_ycorimu_error:1', on_pre='v_post+=w_ycorimu_error',
                 name="ycorIMU_error_comparing_synapses_" + str(hde)) for hde in range(N_x_axis)]

    for group in range(N_x_axis):
        ycorIMU_errors_comparing_synapses[group].connect()
        ycorIMU_errors_comparing_synapses[group].w_ycorimu_error = 2

    # connecting HD
    ycor_error_connect_synapses = Synapses(ycor_Neuron, ycor_comparmatrix_neurons, 'w_ycor_error:1',
                                         on_pre='v_post+=w_ycor_error',
                                         name="ycor_error_connect_synapses")

    for group in range(N_x_axis):
        # connecting neuron matrix with a interval of N_HD to the HD direction(the ycol)
        for index in ycor_index_list[group::N_x_axis]:  # here list a index of neurons which is each ycol in the 72*72 matrix e.g.0,72,144....
            # print("index",index,"connect",connect)
            ycor_error_connect_synapses.connect(i=group, j=index)  # each column is connected to one IMU output value
            ycor_error_connect_synapses.w_ycor_error = 2

    print("don0e  loop")

    ycorError_negative_synapses = Synapses(ycor_comparmatrix_neurons, ycor_negative_error, 'w_ycor_neg:1',
                                       on_pre='v_post+=w_ycor_neg', name='ycor_error_negative_synapses')
    ycorError_positive_synapses = Synapses(ycor_comparmatrix_neurons, ycor_positive_error, 'w_ycor_pos:1',
                                       on_pre='v_post+=w_ycor_pos', name='ycor_error_positive_synapses')
    for ycol in range(N_x_axis - 1):
        xcol = 0  # the number of xcol in the error matrix
        neg_error_value_index = N_x_axis - 1 - ycol  # define the negative error value(according to ycol index), assuming index 0 neuron in negative error neuron group standing for -72
        if xcol <= ycol:
            error_neuron_index = ycol + xcol * (N_x_axis - 1)  # define the index of neuron from error matrix that will connect to negative error neuron
            ycorError_negative_synapses.connect(i=error_neuron_index, j=neg_error_value_index)
            ycorError_negative_synapses.w_hd_neg = 2
            xcol += 1

    for ycol in range(1, N_x_axis - 1):
        xcol = 1
        pos_error_value_index = ycol  # the positive error neuron's value is equal to the ycol index
        if xcol <= (N_x_axis - ycol):
            error_neuron_index = (xcol + 1) * N_x_axis - ycol  # the neuron index of neuron from error matrix that will connect to positive error value neuron
            ycorError_positive_synapses.connect(i=error_neuron_index, j=pos_error_value_index)
            ycorError_positive_synapses.w_hd_pos = 2
            xcol += 1


