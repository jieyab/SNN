import pylab
from brian2 import *
import numpy as np
import vrep
import time
import threading
import matplotlib.pyplot as plt
from aux_function import *
from CD_net import CD_net
from HD_PN_net_withHC import *
from robot_controller import *
from fusi_synapse_landmark import *
from fusi_synapse_wall import *
from Mismatch_net import *
from HD_correct import *
from speed_pool_synapses import *


def run_extendedmodel_training(sim_number, sim_time):
    # defaultclock.dt = 50*ms
    prefs.codegen.target = "numpy"
    start_scope()

    # number of sensor in robot
    N_sensor = 8
    N_vision = 2

    # robot radius
    r = 0.0586

    # CD neurons number
    N_CD = 36
    # vision detection_neurons number
    N_VD = 10

    # HD neuron number
    N_HD = 72
    # coordinate neuron
    N_x_axis, N_y_axis = 32, 32
    N_PI = N_x_axis * N_y_axis

    # Simulation time
    # Speed cell number
    N_speed = 6

    m = 5  # 60
    iter = 0
    # width and length of the arena x=width/ y=lenght
    # 1 square in vrep = 0.5*0.5
    x_scale = 2
    y_scale = 2
    # distance unit per neuron
    N = 200 / (N_x_axis * N_y_axis)

    ##--------------------------------------------------------------------------------------------------------------------##
    ##--------------------Collision detection Neural Architecture--------------------------------------------##
    ##--------------------------------------------------------------------------------------------------------------------##
    Notred_red_inh, spikemon_red, spikemon_notred, spikemon_estimatedlandmark, spikemon_nonlandmark, wall_landmark_synapse, landmark_wall_synapse, Poisson_group, Poisson_vision, Collision_neuron, Collision_or_not, Color, Wall, Landmark, Poisson_synapse, PoissonVision_synapse, Self_ex_synapse, Self_ex_synapse_color, Self_in_synapse, Self_in_synapse_color, Collide_or_not_synapse, Red_landmark_ex, Red_wall_inh, Notred_landmark_inh, spikemon_CD, spikemon_collision, spikemon_landmark, spikemon_poisson, spikemon_wall, Poisson_non_collision, Non_collision_neuron, Poisson_non_synapse, CON_noncollision_synapse, Non_collision_color_synapses, spikemon_noncollison, Estimated_landmark_neuron, Non_landmark_neuron, Poisson_nonlandmark_synapse, Landmark_nonlandmark_synapse, CON_landmark_ex, CON_wall_ex, Notred_wall_ex, Red, Notred, color_notred_synapses, color_red_synapses = CD_net(
        N_CD, N_vision, N_VD)
    # Poisson_group, Poisson_vision, Collision_neuron, Collision_or_not, Color, Wall, Landmark, Poisson_synapse, PoissonVision_synapse, Self_ex_synapse,  Self_in_synapse, Collide_or_not_synapse, Red_landmark_ex, Red_wall_inh, Notred_landmark_inh, spikemon_CD, spikemon_collision, spikemon_landmark, spikemon_poisson, spikemon_wall, Poisson_non_collision, Non_collision_neuron, Poisson_non_synapse, CON_noncollision_synapse, Non_collision_color_synapses, spikemon_noncollison, Estimated_landmark_neuron, Landmark_estLanmark_synapse, statemon_estimatedlandmark, statemon_color, Non_landmark_neuron, Poisson_nonlandmark_synapse, Landmark_nonlandmark_synapse, CON_landmark_ex, CON_wall_ex,Notred_wall_ex,Red,Notred,color_notred_synapses,color_red_synapses= CD_net(N_CD,N_vision,N_VD)

    # Left shift speed neuron / right shift speed neuton already here

    # gaussian input for 8 collison sensors
    stimuli = np.array([gaussian_spike(N_CD, j * N_CD / N_sensor, 10, 0.2) for j in range(N_sensor)])

    # print("init stimuli", stimuli)
    # gaussion distribution for 2 vision input(wall or landmark)
    stimuli_vision = np.array([gaussian_spike(N_VD, j * N_VD / N_vision, 10, 0.2) for j in range(N_vision)])
    # print("stimu_vision gaussian",stimuli_vision)
    ##--------------------------------------------------------------------------------------------------------------------##
    ##--------------------Head Direction and position Neural Architecture----------------------------------------------------------##
    ##--------------------------------------------------------------------------------------------------------------------##
    Speed_Neuron_right, Speed_Neuron_left, IHD_in_synapse, str_inh_synapse, Poisson_straight, Go_straight_neuron, Go_straight, HD_str_synapse, str_IHD_synapse, sti_PI, sti, Poisson_Left, Poisson_Right, Poisson_Compass, Left_drive, Right_drive, HD_Neuron, IHD_Neuron, Left_shift_neuron, Left_shift_speed_neuron, Right_shift_neuron, Right_shift_speed_neuron, HD_Left_synapse, Left_IHD_synapse, HD_Right_synapse, Left_drive_synapse, Right_drive_synapse, Right_IHD_synapse, HD_IHD_synapse, Reset_synapse, HD_ex_synapse, HD_in_synapse, spikemon_HD, spikemon_IHD, Poisson_PI, PI_Neurons, IPI_Neurons, Directional_Neurons, HD_directional_synapse, directional_PI_synapse, PI_shifting_neurons, North_shifting_neurons, South_shifting_neurons, East_shifting_neurons, West_shifting_neurons, PI_ex_synapse, PI_in_synapse, PI_N_synapse, PI_S_synapse, PI_E_synapse, PI_W_synapse, IPI_N_synapse, IPI_S_synapse, IPI_E_synapse, IPI_W_synapse, IPI_PI_synapse, PI_Reset_synapse, spikemon_PI, spikemon_IPI, NE_shifting_neurons, SE_shifting_neurons, WS_shifting_neurons, WN_shifting_neurons, PI_NE_synapse, PI_SE_synapse, PI_WS_synapse, PI_WN_synapse, IPI_NE_synapse, IPI_SE_synapse, IPI_WS_synapse, IPI_WN_synapse, Left_inh_synapse, Right_IHD_synapse, IPI_in_synapse, IPI_stay_synapse, Stay_stay_layer, Stay_layer, Stay, PI_stay_synapse = HD_PI_integrated_net(
        N_HD, N_speed, N_x_axis, N_y_axis, sim_time)

    ##-----------------head direction correcting matrix network-------------#
    pospool_cw_synapses, negpool_ccw_synapses, CW_neuron, CCW_neuron, Poisson_IMU, IMU_Neuron, HD_errormatrix_neurons, HD_positive_error, HD_negative_error, IMU_poi_synapses, IMU_errors_connecting_synapses, HD_error_connect_synapses, Error_negative_synapses, Error_positive_synapses, Ex_speed_pool_Neurons, Inh_speed_pool_Neurons, Positive_ex_pool_synapses, Negative_inh_pool_synapses, statemon_positiveHD_error, statemon_negativeHD_error, spikemon_positiveHD_error, spikemon_negativeHD_error, spiketrain_pos = HD_error_correcting(
        N_HD, HD_Neuron, N_speed)

    ##--------------------------------------------------------------------------------------------------------------------##
    ##--------------------Fusi Synapse between Position - CD----------------------------------------------------------##
    ##--------------------------------------------------------------------------------------------------------------------##
    '''measure collision weight bu fusi synapse'''
    '''
    CD_PI_plastic = fusi(PI_Neurons,Collision_or_not)
    w_plas_shape = np.shape(CD_PI_plastic.w_fusi)[0]
    w_plastic = CD_PI_plastic.w_fusi
    '''
    print("defining synapses")
    ##--------------------------------------------------------------------------------------------------------------------##
    ##--------------------Fusi Synapse between Position - Landmark Neuron----------------------------------------------------------##
    ##--------------------------------------------------------------------------------------------------------------------##

    '''measure collision weight by fusi synapse landmark'''
    landmark_PI_plastic = fusi_landmark(PI_Neurons, Landmark)
    w_landmark_plas_shape = np.shape(landmark_PI_plastic.w_fusi_landmark)[0]
    w_plastic_landmark = landmark_PI_plastic.w_fusi_landmark
    all_fusi_weights_landmark = landmark_PI_plastic.w_fusi_landmark

    ##--------------------------------------------------------------------------------------------------------------------##
    ##--------------------Fusi Synapse between Position - Wall Neuron----------------------------------------------------------##
    ##--------------------------------------------------------------------------------------------------------------------##
    '''measure collision weight by fusi synapse landmark'''
    wall_PI_plastic = fusi_wall(PI_Neurons, Wall)
    w_plas_shape_wall = np.shape(wall_PI_plastic.w_fusi_wall)[0]
    w_plastic_wall = wall_PI_plastic.w_fusi_wall
    all_fusi_weights_wall = wall_PI_plastic.w_fusi_wall

    ##---------STDP between clockwise/counterclockwise neurons and speed right/left neurons------
    w_plas_cw = stdp_errorCW(Speed_Neuron_right, CW_neuron)
    w_plas_cw = stdp_errorCCW(Speed_Neuron_left, CCW_neuron)

    ##--------------------------------------------------------------------------------------------------------------------##
    ##-------------------------------------------Mismatch Network--------------------------------------------------------------##
    ##--------------------------------------------------------------------------------------------------------------------##
    preeq = '''
            v_post += 1*(w_piest >= 0.5)
            '''
    PI_Neurons_est_synapse = Synapses(PI_Neurons, Estimated_landmark_neuron, 'w_piest : 1', on_pre=preeq)
    PI_Neurons_est_synapse.connect()

    Mismatch_landmark_inh_synpase, landmark_mismatch_inh_synapse, spikemon_nonestimatedlandmark, NonEst_landmark_poisson, Non_estimatedlandmark_neuron, Mismatch_neuron, Non_Mismatch_neuron, Nonestimatedlandmark_poi_synapse, Est_nonest_inh_synapse, NonCol_mismatch_synapse, NonCol_nonmismatch_synapse, NonEst_mistmatch_inh_synapse, Est_mismatch_ex_synapse, Est_nonmismatch_inh_synapses, Mismatch_est_inh_synpase, spikemon_Mismatch = mismatch_net(
        Non_landmark_neuron, Estimated_landmark_neuron, Landmark)

    ##--------------------------------------------------------------------------------------------------------------------##
    ##----------------------------------------------Network--------------------------------------------------------------##
    ##--------------------------------------------------------------------------------------------------------------------##
    print("adding network")
    # Network
    PINet = Network()
    # add collision network
    PINet.add(Notred_red_inh, spikemon_red, spikemon_notred, spikemon_estimatedlandmark, spikemon_nonlandmark,
              wall_landmark_synapse, landmark_wall_synapse, Poisson_group, Poisson_vision, Collision_neuron,
              Collision_or_not, Color, Wall, Landmark, Poisson_synapse, PoissonVision_synapse, Self_ex_synapse,
              Self_ex_synapse_color, Self_in_synapse, Self_in_synapse_color, Collide_or_not_synapse, Red_landmark_ex,
              Red_wall_inh, Notred_landmark_inh, spikemon_CD, spikemon_collision, spikemon_landmark, spikemon_poisson,
              spikemon_wall, Poisson_non_collision, Non_collision_neuron, Poisson_non_synapse, CON_noncollision_synapse,
              Non_collision_color_synapses, spikemon_noncollison, Estimated_landmark_neuron, Non_landmark_neuron,
              Poisson_nonlandmark_synapse, Landmark_nonlandmark_synapse, CON_landmark_ex, CON_wall_ex, Notred_wall_ex,
              Red, Notred, color_notred_synapses, color_red_synapses)
    # PINet.add(Poisson_group,Poisson_vision,Collision_neuron,Collision_or_not,Color,Wall,Landmark,Poisson_synapse,PoissonVision_synapse,Self_ex_synapse,Self_in_synapse,Collide_or_not_synapse,Red_landmark_ex,Red_wall_inh,Notred_landmark_inh,spikemon_CD,spikemon_collision,spikemon_landmark,spikemon_poisson,spikemon_wall,Poisson_non_collision,Non_collision_neuron,Poisson_non_synapse,CON_noncollision_synapse,Non_collision_color_synapses,spikemon_noncollison,Estimated_landmark_neuron,Landmark_estLanmark_synapse,statemon_estimatedlandmark,statemon_color, Non_landmark_neuron,Poisson_nonlandmark_synapse,Landmark_nonlandmark_synapse,CON_landmark_ex,CON_wall_ex,Notred_wall_ex,Red,Notred,color_notred_synapses,color_red_synapses)

    # add position network
    PINet.add(Speed_Neuron_right, Speed_Neuron_left, IHD_in_synapse, str_inh_synapse, Poisson_straight,
              Go_straight_neuron, Go_straight, HD_str_synapse, str_IHD_synapse, Poisson_Left, Poisson_Right,
              Poisson_Compass, Left_drive, Right_drive, HD_Neuron, IHD_Neuron, Left_shift_neuron,
              Left_shift_speed_neuron, Right_shift_neuron, Right_shift_speed_neuron, HD_Left_synapse, Left_IHD_synapse,
              HD_Right_synapse, Left_drive_synapse, Right_drive_synapse, Right_IHD_synapse, HD_IHD_synapse,
              Reset_synapse, HD_ex_synapse, HD_in_synapse, spikemon_HD, spikemon_IHD, Poisson_PI, PI_Neurons,
              IPI_Neurons, Directional_Neurons, HD_directional_synapse, directional_PI_synapse, PI_shifting_neurons,
              North_shifting_neurons, South_shifting_neurons, East_shifting_neurons, West_shifting_neurons,
              PI_ex_synapse, PI_in_synapse, PI_N_synapse, PI_S_synapse, PI_E_synapse, PI_W_synapse, IPI_N_synapse,
              IPI_S_synapse, IPI_E_synapse, IPI_W_synapse, IPI_PI_synapse, PI_Reset_synapse, spikemon_PI, spikemon_IPI,
              NE_shifting_neurons, SE_shifting_neurons, WS_shifting_neurons, WN_shifting_neurons, PI_NE_synapse,
              PI_SE_synapse, PI_WS_synapse, PI_WN_synapse, IPI_NE_synapse, IPI_SE_synapse, IPI_WS_synapse,
              IPI_WN_synapse, Left_inh_synapse, Right_IHD_synapse, IPI_in_synapse, IPI_stay_synapse, Stay_stay_layer,
              Stay_layer, Stay, PI_stay_synapse)

    # PINet.add(CD_PI_plastic)
    PINet.add(pospool_cw_synapses, negpool_ccw_synapses, CW_neuron, CCW_neuron, Poisson_IMU, IMU_Neuron,
              HD_errormatrix_neurons, HD_positive_error, HD_negative_error, IMU_poi_synapses,
              IMU_errors_connecting_synapses, HD_error_connect_synapses, Error_negative_synapses,
              Error_positive_synapses, Ex_speed_pool_Neurons, Inh_speed_pool_Neurons, Positive_ex_pool_synapses,
              Negative_inh_pool_synapses, statemon_positiveHD_error, statemon_negativeHD_error,
              spikemon_positiveHD_error, spikemon_negativeHD_error, spiketrain_pos)

    # PINet.add(Poisson_IMU,IMU_Neuron,HD_errormatrix_neurons,HD_positive_error,HD_negative_error,IMU_poi_synapses,IMU_errors_connecting_synapses,HD_error_connect_synapses,Error_negative_synapses,Error_positive_synapses,Ex_speed_pool_Neurons,Inh_speed_pool_Neurons,Positive_ex_pool_synapses,Negative_inh_pool_synapses,statemon_positiveHD_error,statemon_negativeHD_error,spikemon_positiveHD_error,spikemon_negativeHD_error,spiketrain_pos)
    # add fusi plastic synapses for landmark,wall,estLandmark neurons with PI neurons
    PINet.add(landmark_PI_plastic)
    PINet.add(wall_PI_plastic)

    PINet.add(PI_Neurons_est_synapse, Mismatch_landmark_inh_synpase, landmark_mismatch_inh_synapse,
              spikemon_nonestimatedlandmark, NonEst_landmark_poisson, Non_estimatedlandmark_neuron, Mismatch_neuron,
              Non_Mismatch_neuron, Nonestimatedlandmark_poi_synapse, Est_nonest_inh_synapse, NonCol_mismatch_synapse,
              NonCol_nonmismatch_synapse, NonEst_mistmatch_inh_synapse, Est_mismatch_ex_synapse,
              Est_nonmismatch_inh_synapses, Mismatch_est_inh_synpase, spikemon_Mismatch)

    ##-----------------------------------------------------------------------------------------##
    ##---------------------------------Robot controller----------------------------------------##
    ##-----------------------------------------------------------------------------------------##
    # Connect to the server
    print('finished adding network')

    vrep.simxFinish(-1)  # just in case, close all opened connections
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID != -1:
        print("Connected to remote API server")
    else:
        print("Not connected to remote API server")
        sys.exit("Could not connect")
    vrep.simxSynchronous(clientID, 1)

    ##----------------------------------Controller initialized------------------------------------------##
    # set motor
    err_code, l_motor_handle = vrep.simxGetObjectHandle(clientID, "KJunior_motorLeft", vrep.simx_opmode_blocking)
    err_code, r_motor_handle = vrep.simxGetObjectHandle(clientID, "KJunior_motorRight", vrep.simx_opmode_blocking)

    # Compass output=orientation
    # define robot
    err_code, robot = vrep.simxGetObjectHandle(clientID, 'KJunior', vrep.simx_opmode_blocking)

    # define Angles
    err_code, Angles = vrep.simxGetObjectOrientation(clientID, robot, -1, vrep.simx_opmode_streaming)

    # define object position
    err_code, Position = vrep.simxGetObjectPosition(clientID, robot, -1, vrep.simx_opmode_streaming)

    # get sensor
    sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8 = getSensor(clientID)
    # read point
    detectedPoint1, detectedPoint2, detectedPoint3, detectedPoint4, detectedPoint5, detectedPoint6, detectedPoint7, detectedPoint8 = getDetectedpoint(
        sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8, clientID)
    # Distance from sensor to obstacle
    sensor_val1, sensor_val2, sensor_val3, sensor_val4, sensor_val5, sensor_val6, sensor_val7, sensor_val8 = getSensorDistance(
        detectedPoint1, detectedPoint2, detectedPoint3, detectedPoint4, detectedPoint5, detectedPoint6, detectedPoint7,
        detectedPoint8)
    # get vision sensor
    err_code, camera = vrep.simxGetObjectHandle(clientID, "Vision_sensor", vrep.simx_opmode_blocking)
    err_code, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 1, vrep.simx_opmode_streaming)

    ##---------------------------------Sensor initial condition (for CD)------------------------------------##
    # sensor value of every sensor for every neuron
    # Use inv_filter to filter out noise
    # sensor_val = np.array([np.repeat(inv_filter(sensor_val1),N_CD), np.repeat(inv_filter(sensor_val2),N_CD),np.repeat(inv_filter(sensor_val3),N_CD),np.repeat(inv_filter(sensor_val7),N_CD),np.repeat(inv_filter(sensor_val8),N_CD)])
    sensor_val = np.array([np.repeat(inv_filter(sensor_val1), N_CD), np.repeat(inv_filter(sensor_val2), N_CD),
                           np.repeat(inv_filter(sensor_val3), N_CD), np.repeat(inv_filter(sensor_val4), N_CD),
                           np.repeat(inv_filter(sensor_val5), N_CD), np.repeat(inv_filter(sensor_val6), N_CD),
                           np.repeat(inv_filter(sensor_val7), N_CD), np.repeat(inv_filter(sensor_val8), N_CD)])

    # print('initi sensor_val shape is',sensor_val.shape)
    sensor_val_vision = np.zeros([N_vision, N_VD])
    # print('init sensor vision val is', sensor_val_vision.shape)
    # sum of each sensor value * its gaussian distribution --> sum to find all activity for each neurons --> WTA
    All_stimuli = np.sum(sensor_val * stimuli, axis=0)
    # print('initi all stimuli shape is', All_stimuli.shape)
    All_stimuli_vision = np.sum(sensor_val_vision * stimuli_vision, axis=0)

    # find the winner
    winner = WTA(All_stimuli)
    # print("initi winner vector is" , winner)
    for w in winner:
        Poisson_synapse.w_poi[w] = All_stimuli[w]
    winner_vision = WTA(All_stimuli_vision)
    # print("initi winner vector is", winner_vision)
    for w in winner_vision:
        PoissonVision_synapse.w_vision_poi[w] = All_stimuli_vision[w]

    ##------------------------------------------------------------------------------------------------##

    # initial speed of wheel
    l_steer = 0.24
    r_steer = 0.24
    # record the initial time
    t_int = time.time()
    # record compass angle
    compass_angle = np.array([])
    # record x and y axis at each time
    x_axis = np.array([])
    y_axis = np.array([])
    # path that is passing
    r = np.array([])
    # all_time
    all_time = np.array([])
    # collision index that the robot collided
    collision_index = np.array([])
    collision_index_during_run = []

    # record weight parameter
    # record every m sec
    weight_record_time = 1
    # weight_during_run = np.zeros([np.shape(w_plastic)[0],2*int(sim_time/m)])

    weight_during_run_landmark = np.zeros([np.shape(w_plastic_landmark)[0], 3 * int(sim_time / 5)])
    weight_during_run_wall = np.zeros([np.shape(w_plastic_wall)[0], 3 * int(sim_time / 5)])
    m_ind = 0

    time_step_list = np.array([])

    # start simulation
    while (time.time() - t_int) < sim_time:

        t1 = time.time() - t_int
        # record sensor at each time step
        sensor_val1, sensor_val2, sensor_val3, sensor_val4, sensor_val5, sensor_val6, sensor_val7, sensor_val8 = getSensorDistance(
            detectedPoint1, detectedPoint2, detectedPoint3, detectedPoint4, detectedPoint5, detectedPoint6,
            detectedPoint7, detectedPoint8)
        # print("sensor values",sensor_val1,sensor_val2,sensor_val3,sensor_val4,sensor_val5,sensor_val6,sensor_val7,sensor_val8)
        all_sensor = np.array([sensor_val1, sensor_val2, sensor_val3, sensor_val4, sensor_val5])
        # print("all sensor is ", all_sensor)
        all_sensor[all_sensor < 4.1e-20] = np.infty
        # print("all snesor", all_sensor)
        activated_sensor = np.argmin(all_sensor)
        # print("activated sensor index is",activated_sensor)

        # obtain vision sensor values
        pixelimage = set(image)

        if all_sensor[activated_sensor] < 0.2:
            if activated_sensor == 3:
                r_steer, l_steer, zeta = TurnRight(r_steer, l_steer, delta_t)
            elif activated_sensor == 4:
                r_steer, l_steer, zeta = TurnRight(r_steer, l_steer, delta_t)
            # elif activated_sensor == 5:
            #   r_steer, l_steer, zeta = TurnLeft(r_steer, l_steer, delta_t)
            elif activated_sensor == 0:
                r_steer, l_steer, zeta = TurnLeft(r_steer, l_steer, delta_t)
            elif activated_sensor == 1:
                r_steer, l_steer, zeta = TurnLeft(r_steer, l_steer, delta_t)
            elif activated_sensor == 2:
                r_steer, l_steer, zeta = TurnLeft(r_steer, l_steer, delta_t)
            else:
                l_steer = 0.24
                r_steer = 0.24
                zeta = 0
        else:
            l_steer = 0.24
            r_steer = 0.24
            zeta = 0

        ####-------------------- Record weight------------------------####

        '''
        if t1 <= weight_record_time and weight_record_time < t1+5:
            print('recording weights')
            #weight_during_run[:,m_ind] = w_plastic
            weight_during_run_landmark[:, m_ind] = w_plastic_landmark
            weight_during_run_wall[:, m_ind] = w_plastic_wall
            m_ind += 1
            weight_record_time += m
            collision_index_during_run.append(collision_index)
        '''

        weight_during_run_landmark[:, iter] = w_plastic_landmark
        weight_during_run_wall[:, iter] = w_plastic_wall
        collision_index_during_run.append(collision_index)

        ####-------------------- Start recording spike (CD) ------------------------####
        # sensor_val = np.array([np.repeat(inv_filter(sensor_val1),N_CD), np.repeat(inv_filter(sensor_val2),N_CD),np.repeat(inv_filter(sensor_val3),N_CD),np.repeat(inv_filter(sensor_val7),N_CD),np.repeat(inv_filter(sensor_val8),N_CD)])
        sensor_val = np.array([np.repeat(inv_filter(sensor_val1), N_CD), np.repeat(inv_filter(sensor_val2), N_CD),
                               np.repeat(inv_filter(sensor_val3), N_CD), np.repeat(inv_filter(sensor_val4), N_CD),
                               np.repeat(inv_filter(sensor_val5), N_CD), np.repeat(inv_filter(sensor_val6), N_CD),
                               np.repeat(inv_filter(sensor_val7), N_CD), np.repeat(inv_filter(sensor_val8), N_CD)])

        # print("sensor_val",sensor_val)
        sensor_val_vision = generate_vision_val(pixelimage, N_VD)
        # print("sensor vision",sensor_val_vision)

        # reset weight to 0 again
        Poisson_synapse.w_poi = 0
        PoissonVision_synapse.w_vision_poi = 0

        err_code = vrep.simxSetJointTargetVelocity(clientID, l_motor_handle, l_steer, vrep.simx_opmode_streaming)
        err_code = vrep.simxSetJointTargetVelocity(clientID, r_motor_handle, r_steer, vrep.simx_opmode_streaming)

        # print("current sensor_val is ", sensor_val)
        # print("current sensor vision val",sensor_val_vision)
        # print("currentpixel",pixelimage)

        # All stimuli and WTA
        All_stimuli = (np.sum(sensor_val * stimuli, axis=0))

        # print("current all stimuli is", All_stimuli)
        winner = WTA(All_stimuli)
        # print('all stimuli ', All_stimuli)
        for w in winner:
            Poisson_synapse.w_poi[w] = All_stimuli[w]
        # print("possion synapse for collision neurons", [i for i in Poisson_synapse.w_poi if i > 0])
        # print("pixel image",set(image))
        # print("current winner is",winner)

        All_stimuli_vision = np.sum(sensor_val_vision * stimuli_vision, axis=0)
        # print("current all stimuli vision",All_stimuli_vision)
        winner_vision = WTA(All_stimuli_vision)
        # print("current winner vision index is",winner_vision)
        for w in winner_vision:
            PoissonVision_synapse.w_vision_poi[w] = All_stimuli_vision[w] / 10
        # print("vision poisson synpase weight index", PoissonVision_synapse.w_vision_poi)

        # --------mismatching detecting---------##
        PI_Neurons_est_synapse.w_piest = landmark_PI_plastic.w_fusi_landmark

        ####-------------------- End recording spike ----------------------------####

        ####-------------------- Start recording Head direction ------------------------####

        # Choose neuron that is the nearest to the turning speed/direction
        # if turn left
        if r_steer == 0:
            neuron_index = nearest_neuron_speed(zeta, N_speed)
            for syn in range(N_speed):
                Left_drive_synapse[syn].w_left_drive = 0
                Right_drive_synapse[syn].w_right_drive = 0
            Left_drive_synapse[neuron_index].w_left_drive = 5
            Right_drive_synapse[neuron_index].w_right_drive = 0
            Go_straight.w_str = -3
            directional_PI_synapse.w_dir_PI = 0
            Stay_stay_layer.w_stay_stay = 2
        # if turn right
        elif l_steer == 0:
            neuron_index = nearest_neuron_speed(zeta, N_speed)
            for syn in range(N_speed):
                Left_drive_synapse[syn].w_left_drive = 0
                Right_drive_synapse[syn].w_right_drive = 0
            Left_drive_synapse[neuron_index].w_left_drive = 0
            Right_drive_synapse[neuron_index].w_right_drive = 5
            Go_straight.w_str = -3
            directional_PI_synapse.w_dir_PI = 0  # if turn position PI stay
            Stay_stay_layer.w_stay_stay = 2
        # if go straight
        else:
            for syn in range(N_speed):
                Left_drive_synapse[syn].w_left_drive = 0
                Right_drive_synapse[syn].w_right_drive = 0
            Go_straight.w_str = 10
            directional_PI_synapse.w_dir_PI = 4  # if move position PI run
            Stay_stay_layer.w_stay_stay = -4

        # no PI update if turning --> l/r =0
        ##obtain current estimated head direction

        ##-----------------reset IHD (Compass) -----------------##

        # Get heading direction
        err_code, Angles = vrep.simxGetObjectOrientation(clientID, robot, -1, vrep.simx_opmode_streaming)

        heading_dir = getHeadingdirection(Angles)
        # print("IMU head direction is", heading_dir, "angle from IMU is", Angles)
        compass_angle = np.append(compass_angle, heading_dir)

        # recalibrate head direction to nearest neuron
        recal = nearest_neuron_head(heading_dir, N_HD)
        # print("recallllllll is",recal)
        #    IMU_poi_synapses.w_poi_imu = np.array(gaussian_spike(N_HD,recal,30,0.03))

        ## connecting to head direction error network

        # set reset by compass weight upon angle atm
        Reset_synapse.w_reset = np.array(gaussian_spike(N_HD, recal, 30, 0.03))
        # print("reset synapse",Reset_synapse.w_reset)

        ##-----------------head direction error correction -----------------##
        IMU_poi_synapses.w_poi_imu = np.array(gaussian_spike(N_HD, recal, 30, 0.03))

        ##-----------------mismatch detection -----------------##

        ##----------------- Read position -----------------##
        err_code, Position = vrep.simxGetObjectPosition(clientID, robot, -1, vrep.simx_opmode_streaming)
        # print("position value is",Position)
        x_pos = Position[0]
        y_pos = Position[1]

        # recalibrate head direction to nearest neuron
        x_axis = np.append(x_axis, x_pos)
        y_axis = np.append(y_axis, y_pos)

        # recalibrate
        recal_x_axis = nearest_neuron_x_axis(x_pos, N_x_axis, x_scale)
        recal_y_axis = nearest_neuron_y_axis(y_pos, N_y_axis, y_scale)
        recal_index = N_x_axis * recal_y_axis + recal_x_axis
        # print("recali position",recal_x_axis,recal_y_axis )

        r = np.append(r, recal_index)  # is an array keeping all index that neuron fire during the run
        # set reset weight

        ### comment this line to diable using IMU
        PI_Reset_synapse.w_poi_PI = np.array(gaussian_spike(N_PI, recal_index, 20, 0.01))

        ##----------------- Index when collision -----------------##
        if l_steer == 0 or r_steer == 0:
            collision_index = np.append(collision_index, recal_index)

        ##----------------- Collect time-----------------##
        all_time = np.append(all_time, time.time() - t_int)

        # run
        PINet.run(15 * ms)

        print("current plas landmark synapese index", np.where(w_plastic_landmark >= 0.5))
        # print("current landmark", [i for i in w_plastic_landmark if i > 0.3])
        # print("current est landmark value", [i for i in w_plastic_estlandmark if i >= 0.5])
        # print('wall synpases values',[i for i in w_plastic_wall if i > 0])
        print('wall synpases values index bigger', np.where(w_plastic_wall >= 0.5))
        print('count mismatch',len(spikemon_Mismatch))
        print('count landmark', len(spikemon_landmark))
        # print('wall synpases values index bigger 0.7', np.where(w_plastic_wall > 0.7))

        # print("fusi values ",[i for i in estLandmark_PI_plastic.p if i > 0])
        # print("fusi valuessss ",[i for i in estLandmark_PI_plastic.w_fusi_estimatedlandmark if i > 0])

        # print("wall spike", list(spikemon_wall.t),"landmark spikes",list(spikemon_landmark.t))
        # print("last mismatch voltage spike time is ", list(spikemon_Mismatch.t))
        #   print("HD pos error spike ", spikemon_positiveHD_error.i[:])
        #  print('HD neg error spike',spikemon_negativeHD_error.i[:])
        #   print("HD pos spike train", spikemon_positiveHD_error.num_spikes)

        # print(defaultclock.t)
        print('####-------------------- Read sensor (new round) ----------------------------####')

        # slow down the controller for more stability
        # time.sleep(4.2)
        # Start new measurement in the next time step
        detectedPoint1, detectedPoint2, detectedPoint3, detectedPoint4, detectedPoint5, detectedPoint6, detectedPoint7, detectedPoint8 = getDetectedpoint(
            sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8, clientID)
        err_code, resolution, image = vrep.simxGetVisionSensorImage(clientID, camera, 1, vrep.simx_opmode_streaming)
        # err_code, resolution, image2 = vrep.simxGetVisionSensorImage(clientID, camera2, 1, vrep.simx_opmode_streaming)

        # print when near obstacle
        # if collision(sensor_val1, sensor_val2, sensor_val3, sensor_val4, sensor_val5, sensor_val6, sensor_val7,
        #           sensor_val8):
        # print("near collision")

        # record time step and final time
        t2 = time.time() - t_int
        delta_t = t2 - t1
        final_time = time.time()
        time_step_list = np.append(time_step_list, delta_t)
        print("delta-t", delta_t)
        print("final time t2", int(t2))
        iter += 1

    ##---------------end of simulation ---------------##

    print("Done")

    ''' For plotting results, run plotter.py'''

    # uncomment to save

    pathway = "/Users/jieyab/SNN/for_plotting/"
    ##

    exp_number = str(sim_number)

    np.save(pathway + "r" + exp_number, r)
    # np.save(pathway+"spike_time.csv", spike_time)
    np.save(pathway + "spikemon_CD_i" + exp_number, spikemon_CD.i)
    np.save(pathway + "spikemon_CD_t" + exp_number, spikemon_CD.t / ms)
    np.save(pathway + "spikemon_HD_i" + exp_number, spikemon_HD.i)
    np.save(pathway + "spikemon_HD_t" + exp_number, spikemon_HD.t / ms)
    np.save(pathway + "spikemon_PI_i" + exp_number, spikemon_PI.i)
    np.save(pathway + "spikemon_PI_t" + exp_number, spikemon_PI.t / ms)

    np.save(pathway + "spikemon_noncollision_i" + exp_number, spikemon_noncollison.i)
    np.save(pathway + "spikemon_noncollision_t" + exp_number, spikemon_noncollison.t / ms)

    np.save(pathway + "spikemon_wall_i" + exp_number, spikemon_wall.i)
    np.save(pathway + "spikemon_wall_t" + exp_number, spikemon_wall.t / ms)
    np.save(pathway + "spikemon_nonlandmark_i" + exp_number, spikemon_nonlandmark.i)
    np.save(pathway + "spikemon_nonlandmark_t" + exp_number, spikemon_nonlandmark.t / ms)
    np.save(pathway + "spikemon_landmark_i" + exp_number, spikemon_landmark.i)
    np.save(pathway + "spikemon_landmark_t" + exp_number, spikemon_landmark.t / ms)
    np.save(pathway + "weight_during_run_landmark" + exp_number, weight_during_run_landmark)
    np.save(pathway + "weight_during_run_wall" + exp_number, weight_during_run_wall)
    np.save(pathway + "weight_landmark" + exp_number, w_plastic_landmark)
    np.save(pathway + "weight__wall" + exp_number, w_plastic_wall)
    np.save(pathway + "spikemon_mismatch_t" + exp_number, spikemon_Mismatch.t / ms)
    np.save(pathway + "spikemon_mismatch_v" + exp_number, spikemon_Mismatch.v)
    np.save(pathway + "collision_index" + exp_number, collision_index)
    np.save(pathway + "collision_index_during_run" + exp_number, collision_index_during_run)
    np.save(pathway + "compass_angle" + exp_number, compass_angle)
    np.save(pathway + "all_time" + exp_number, all_time)
    np.save(pathway + "spikemon_estimatedlandmark_i" + exp_number, spikemon_estimatedlandmark.i)
    np.save(pathway + "spikemon_estimatedlandmark_t" + exp_number, spikemon_estimatedlandmark.t / ms)
    np.save(pathway + "spikemon_nonestimatedlandmark_i" + exp_number, spikemon_nonestimatedlandmark.i)
    np.save(pathway + "spikemon_nonestimatedlandmark_t" + exp_number, spikemon_nonestimatedlandmark.t / ms)
    np.save(pathway + "spikemon_red_i" + exp_number, spikemon_red.i)
    np.save(pathway + "spikemon_red_t" + exp_number, spikemon_red.t / ms)
    np.save(pathway + "spikemon_notred_i" + exp_number, spikemon_notred.i)
    np.save(pathway + "spikemon_notred_t" + exp_number, spikemon_notred.t / ms)

    np.save(pathway + "spikemon_positiveHD_error_t" + exp_number, spikemon_positiveHD_error.t / ms)
    np.save(pathway + "spikemon_positiveHD_error_i" + exp_number, spikemon_positiveHD_error.i)
    np.save(pathway + "spikemon_negativeHD_error_t" + exp_number, spikemon_negativeHD_error.t / ms)
    np.save(pathway + "spikemon_negativeHD_error_i" + exp_number, spikemon_negativeHD_error.i)
    np.save(pathway + "step_time" + exp_number, time_step_list)
    # np.save(pathway+"weight_matrix.npy",weight_matrix)

    # np.save(pathway+"weight_matrix.npy",weight_matrix)

    return clientID
