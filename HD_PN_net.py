
from brian2 import *
import numpy as np
from aux_function import gaussian_spike
  

    
    
def HD_PI_integrated_net(N_HD, N_speed, N_x_axis, N_y_axis, sim_time):
    
    
    ##------------------------------------ Poisson group for all -------------------------------------##
     
    #Reset Neuron --> need to define
    reset_freq = [3000,0,0,0]  #f=np.tile(reset_freq, int(sim_time*4))
    #time of stimulus 40ms of reseting orientation
    sti = TimedArray(np.tile(reset_freq, int(sim_time))*Hz, dt=20*ms)
    
    #Headding direction
    Poisson_Compass = PoissonGroup(N_HD, rates='sti(t)',name='poisson_compass')   #Poisson Stimuli group for the first
    
    #left drive poisson group 
    Poisson_Left = PoissonGroup(N_speed, rates=3000*Hz,name='poisson_drive_left')   #Poisson Stimuli group for the first
    #change with speed
    Left_drive = [ Poisson_Left[(inp):(inp + 1)] for inp in range(N_speed)]
  

    Poisson_Right = PoissonGroup(N_speed, rates=3000*Hz,name='poisson_drive_right')   #Poisson Stimuli group for the first
    Right_drive = [ Poisson_Right[(inp):(inp + 1)] for inp in range(N_speed)]
   
    Poisson_straight= PoissonGroup(1, rates=3000*Hz,name='poisson_straight')   #Poisson Stimuli group for the first
    
     
        
    ##------------------------------------ Head direction -------------------------------------##
    
    ##--------------------Neurons-----------------------##
    
    #Collision detection Neurons and its equation
    eqs_HD='''dv/dt = (-v) / (10*ms) : 1'''    
    #Head direction neuron
    HD_Neuron = NeuronGroup(N_HD, eqs_HD,threshold='v>1',reset='v=0',method='linear')
    #IHD neuron
    IHD_Neuron = NeuronGroup(N_HD, eqs_HD,threshold='v>1',reset='v=0',method='linear')
    #adding speed neurons
    Speed_Neuron_left = NeuronGroup(N_speed, eqs_HD,threshold='v>1',reset='v=0',method='linear')
    Speed_Neuron_right= NeuronGroup(N_speed, eqs_HD,threshold='v>1',reset='v=0',method='linear')

    #Layers
    #Left shift (N_speed) neurons
    Left_shift_neuron = NeuronGroup(N_HD * N_speed,eqs_HD,threshold='v>1',reset='v=0',method='linear')
    Left_shift_speed_neuron = [ Left_shift_neuron[(inp * N_HD):((inp + 1) * N_HD)] for inp in range(N_speed)]
    
    #Right shift (N_speed) neurons
    Right_shift_neuron = NeuronGroup(N_HD * N_speed,eqs_HD,threshold='v>1',reset='v=0',method='linear')
    Right_shift_speed_neuron = [ Right_shift_neuron[(inp * N_HD):((inp + 1) * N_HD)] for inp in range(N_speed)]
        
    #go straight
    Go_straight_neuron = NeuronGroup(N_HD ,eqs_HD,threshold='v>1',reset='v=0',method='linear',name='go_Straight_nueron')


    ##--------------------Synapse-----------------------##          
                    
    #speed neuron = xx angle -->60 is the largest angle that robot turn
    angle_per_SpeedNeuron = int(70/N_speed)
    #HD neuron = xx degree per neuron --> 360 is all HD
    angle_per_HeadNeuron = int(360/N_HD)                
    
    #minimum neuron change scale -this is the change in neuron index of heading direction due to speed neuron numbers
    change_scale =  int(angle_per_SpeedNeuron/angle_per_HeadNeuron)
    if change_scale ==0:
        change_scale =1
    
        
    #go straight
    Go_straight = Synapses(Poisson_straight, Go_straight_neuron,'w_str:1', on_pre='v_post+=w_str',name='Go_straight')
    Go_straight.connect()
    Go_straight.w_str = 2
    
    HD_str_synapse = Synapses(HD_Neuron, Go_straight_neuron,'w_HD_str:1', on_pre='v_post+=w_HD_str',name='HD_straight')
    HD_str_synapse.connect('i!=j')
    HD_str_synapse.w_HD_str = -10
    
    str_IHD_synapse = Synapses(Go_straight_neuron,IHD_Neuron,'w_str_IHD:1', on_pre='v_post+=w_str_IHD',name='straight_IHD')
    str_IHD_synapse.connect(j ='i')
    str_IHD_synapse.w_str_IHD = 2    
    
    str_inh_synapse = Synapses(Go_straight_neuron,Go_straight_neuron,'w_str_inh:1', on_pre='v_post+=w_str_inh',name='straight_inh')
    str_inh_synapse.connect('i != j')
    str_inh_synapse.w_str_inh = -10 
    
    
    #Left drive poission
    Left_drive_synapse = [Synapses(Left_drive[speed],Left_shift_speed_neuron[speed],'w_left_drive:1', on_pre='v_post+=w_left_drive') for speed in range(N_speed) ]
    for group in range(N_speed):
        Left_drive_synapse[group].connect()
        Left_drive_synapse[group].w_left_drive = 0
        
    #Right drive poission
    Right_drive_synapse = [Synapses(Right_drive[speed],Right_shift_speed_neuron[speed],'w_right_drive:1', on_pre='v_post+=w_right_drive') for speed in range(N_speed) ]
    for group in range(N_speed):
        Right_drive_synapse[group].connect()
        Right_drive_synapse[group].w_right_drive = 0

    # Left synapse
    
    #HD - Left shift
    HD_Left_synapse = [ Synapses(HD_Neuron, Left_shift_speed_neuron[speed],'w_left:1',on_pre='v_post+=w_left', name = 'HD_Left_sypanse_'+str(speed)) for speed in range(N_speed)]
    #Left shift - IHD
    Left_IHD_synapse =  [ Synapses(Left_shift_speed_neuron[speed],IHD_Neuron,'w_leftshift:1',on_pre='v_post+=w_leftshift', name = 'Left_IHD_synapse_'+str(speed)) for speed in range(N_speed)]
    #Left - Left inhibit
    Left_inh_synapse = [Synapses(Left_shift_speed_neuron[speed], Left_shift_speed_neuron[speed],'w_left_inh:1', on_pre='v_post+=w_left_inh',name = 'Left_inh_synapse_'+str(speed)) for speed in range(N_speed)]
   
    for group in range(N_speed):
        #HD - Left shift
        HD_Left_synapse[group].connect(condition='abs(i-j)>=1')
        HD_Left_synapse[group].w_left = -10
        #Left shift - IHD
        Left_inh_synapse[group].connect(condition='abs(i-j)>=1')
        Left_inh_synapse[group].w_left_inh=-5
        Left_IHD_synapse[group].connect(j='i-change_scale*(group+1)', skip_if_invalid=True)
        for g in range(change_scale*(group+1)):
            Left_IHD_synapse[group].connect(i=g,j=N_HD-(1+group)*change_scale+g)
        Left_IHD_synapse[group].w_leftshift = 4
         
    # Right synapse
    
    #HD - Right shift
    HD_Right_synapse = [ Synapses(HD_Neuron, Right_shift_speed_neuron[speed],'w_right:1',on_pre='v_post+=w_right',name = 'HD_Right_sypanse_'+str(speed)) for speed in range(N_speed)]
    #Right shift - IHD
    Right_IHD_synapse =  [ Synapses(Right_shift_speed_neuron[speed],IHD_Neuron,'w_rightshift:1',on_pre='v_post+=w_rightshift',name = 'Right_IHD_synapse_'+str(speed)) for speed in range(N_speed)]
    #Right - Right inhibit
    Right_inh_synapse = [Synapses(Right_shift_speed_neuron[speed], Right_shift_speed_neuron[speed],'w_right_inh:1', on_pre='v_post+=w_right_inh',name = 'Right_inh_synapse'+str(speed)) for speed in range(N_speed)]


        
    for group in range(N_speed):
        #HD - Right shift
        HD_Right_synapse[group].connect(condition='abs(i-j)>=1')
        HD_Right_synapse[group].w_right=-10
        #Right shift - IHD
        Right_inh_synapse[group].connect(condition='abs(i-j)>=1')
        Right_inh_synapse[group].w_right_inh=-5
        Right_IHD_synapse[group].connect(j='i+change_scale*(group+1)', skip_if_invalid=True)
        for g in range(change_scale*(group+1)):
            Right_IHD_synapse[group].connect(i=N_HD-(1+group)*change_scale+g,j=g)
        Right_IHD_synapse[group].w_rightshift = 4

    ##HD - IHD
    
    # IHD - HD synapse
    HD_IHD_synapse=Synapses(IHD_Neuron,HD_Neuron,'w_trans:1', on_pre='v_post+=w_trans',name='IHD_HD')
    HD_IHD_synapse.connect(j='i')
    HD_IHD_synapse.w_trans=17
    
    #Reset synapase when error occur // every delta_t
    Reset_synapse = Synapses(Poisson_Compass, IHD_Neuron, 'w_reset:1',on_pre='v_post+=w_reset',name='Reset_HD')
    Reset_synapse.connect(j='i')
    Reset_synapse.w_reset=np.array(gaussian_spike(N_HD,0,10,0.03))  #initial condition  #WTA
    
    #Self excitatory synapse
    HD_ex_synapse=Synapses(HD_Neuron, HD_Neuron,'w_HD_self:1', on_pre='v_post+=w_HD_self',name='HD_ex_self')
    HD_ex_synapse.connect(j='i')
    HD_ex_synapse.w_HD_self=10
    
    #self_inhibitory￼

    HD_in_synapse=Synapses(HD_Neuron, HD_Neuron,'w_HD_nself:1', on_pre='v_post+=w_HD_nself',name='HD_in_self')
    HD_in_synapse.connect(condition='i!=j')
    HD_in_synapse.w_HD_nself=-15
    
    
    IHD_in_synapse=Synapses(IHD_Neuron, IHD_Neuron,'w_IHD_nself:1', on_pre='v_post+=w_IHD_nself',name='HD_in_nself')
    IHD_in_synapse.connect(condition='i!=j')
    IHD_in_synapse.w_IHD_nself=-10
    
     ##-------------------------------------------------------------------------------------------------##
     ##------------------------------------ Position path integration -------------------------------------##
      ##-------------------------------------------------------------------------------------------------##
      
      
         #All neuron in PI
    N_PI = N_x_axis * N_y_axis
    
    #Possible direction --> Generalized?
    direction = 8
     
        #Reset Neuron --> need to define
    reset_freq1 = [2000, 0, 0,0, 0, 0,0, 0,0] 
    #f=np.tile(reset_freq, int(sim_time*4))
    #time of stimulus 40ms of reseting orientation
    #########################################################################################################################
    sti_PI = TimedArray(np.tile(reset_freq1, int(sim_time))*Hz, dt=20*ms)
   
    Poisson_PI = PoissonGroup(N_PI, rates='sti_PI(t)',name='poisson_PI')  
    Stay = PoissonGroup(1, rates=700*Hz,name='stay')  

     
    ### Receive input to choose from N,S,E,W ###
    
    #HD-direction synapse - excite
    
    #Neuron in each direction
    North_neuron = list(range(int(157.5/angle_per_HeadNeuron)+1,int(202.5/angle_per_HeadNeuron)+1))
    East_neuron = list(range(int(67.5/angle_per_HeadNeuron)+1,int(112.5/angle_per_HeadNeuron)+1))
    West_neuron = list(range(int(247.5/angle_per_HeadNeuron)+1,int(292.5/angle_per_HeadNeuron)+1))
    South_neuron = list(range(int(0/angle_per_HeadNeuron),int(22.5/angle_per_HeadNeuron)+1))+list(range(int(337.5/angle_per_HeadNeuron)+1,int(360/angle_per_HeadNeuron)))
    NE_neuron = list(range(int(112.5/angle_per_HeadNeuron)+1,int(157.5/angle_per_HeadNeuron)+1))
    WN_neuron = list(range(int(202.5/angle_per_HeadNeuron)+1,int(247.5/angle_per_HeadNeuron)+1))
    SE_neuron = list(range(int(22.5/angle_per_HeadNeuron)+1,int(67.5/angle_per_HeadNeuron)+1))
    WS_neuron = list(range(int(292.5/angle_per_HeadNeuron)+1,int(337.5/angle_per_HeadNeuron)+1))



    All_direction = [North_neuron,South_neuron,East_neuron,West_neuron,NE_neuron,SE_neuron,WS_neuron,WN_neuron]
    
    
       
                            ##--------------------Neurons-----------------------##
     
    #Neuron equation
    eqs_PI='''dv/dt = (-v) / (10*ms) : 1'''    

    #coordinate neuron
    PI_Neurons = NeuronGroup(N_PI, eqs_PI,threshold='v>1',reset='v=0',method='linear',name='PI_Neurons')
    
    #intergated coordinate neuron
    IPI_Neurons = NeuronGroup(N_PI, eqs_PI,threshold='v>1',reset='v=0',method='linear',name='IPI_Neurons') 
    
    #directional neuron
    Directional_Neurons = NeuronGroup(direction, eqs_PI,threshold='v>1',reset='v=0',method='linear',name='directional_neurons')
    North = Directional_Neurons[0:1]
    South = Directional_Neurons[1:2]
    East = Directional_Neurons[2:3]
    West = Directional_Neurons[3:4]
    North_East = Directional_Neurons[4:5]
    North_West = Directional_Neurons[5:6]
    South_East = Directional_Neurons[6:7]
    South_West = Directional_Neurons[7:8]



    #### Shifting layer  ####
    
    PI_shifting_neurons = NeuronGroup((direction+1)*N_PI, eqs_PI,threshold='v>1',reset='v=0',method='linear',name='PI_shifiting_neurons')
    
    #North shift
    North_shifting_neurons = PI_shifting_neurons[0:N_PI]
   

    #South shift
    South_shifting_neurons = PI_shifting_neurons[N_PI:2*N_PI]


    #East shift
    East_shifting_neurons = PI_shifting_neurons[2*N_PI:3*N_PI]


    #West shift
    West_shifting_neurons = PI_shifting_neurons[3*N_PI:4*N_PI]
      
    #NorthEast
    NE_shifting_neurons = PI_shifting_neurons[4*N_PI:5*N_PI]

    #SouthEast
    SE_shifting_neurons = PI_shifting_neurons[5*N_PI:6*N_PI]

    #WestSouth
    WS_shifting_neurons = PI_shifting_neurons[6*N_PI:7*N_PI]

    #WestNorth
    WN_shifting_neurons = PI_shifting_neurons[7*N_PI:8*N_PI]

    #Stay
    Stay_layer = PI_shifting_neurons[8*N_PI:9*N_PI]

                            ##--------------------Synapse-----------------------##   
                                         #### --------------------- ####
                                        #### HD - Directional ####
                                        #### --------------------- ####
                                        
    HD_directional_synapse = Synapses( HD_Neuron,Directional_Neurons,'w_dir_HD:1',on_pre='v_post+=w_dir_HD',name='HD_directional_synapse')
    for d in range(direction):
        HD_directional_synapse.connect(i=All_direction[d], j=d)
    HD_directional_synapse.w_dir_HD = 4
    
    
                                       
                                        
                                        
                                        
                                        
                                        #### --------------------- ####
                                    #### Directional - PI_shifting all ####
                                        #### --------------------- ####
                                        

    directional_PI_synapse = Synapses( Directional_Neurons,PI_shifting_neurons,'w_dir_PI:1',on_pre='v_post+=w_dir_PI',name='directional_PI_synapse')
    for d in range(direction):
        for ind in range(d*N_PI,(d+1)*N_PI):
            directional_PI_synapse.connect(i=d, j=ind)
    directional_PI_synapse.w_dir_PI = 4
    
    
                                        #### --------------------- ####
                                        #### WTA coordinate neuron ####
                                        #### --------------------- ####
    

    PI_ex_synapse = Synapses(PI_Neurons, PI_Neurons,'w_PI_self:1',on_pre='v_post+=w_PI_self', name = 'PI_ex_synapse')
    PI_ex_synapse.connect(j='i')
    PI_ex_synapse.w_PI_self = 1.5
    
    PI_in_synapse = Synapses(PI_Neurons, PI_Neurons,'w_PI_nself:1',on_pre='v_post+=w_PI_nself', name = 'PI_in_synapse')
    PI_in_synapse.connect(condition='i!=j')
    PI_in_synapse.w_PI_nself = -10
    
    #IPI with IPI
    IPI_in_synapse = Synapses(IPI_Neurons, IPI_Neurons,'w_IPI_nself:1',on_pre='v_post+=w_IPI_nself', name = 'IPI_in_synapse')
    IPI_in_synapse.connect(condition='i!=j')
    IPI_in_synapse.w_IPI_nself = -10
    
    
     
                                        #### --------------------- #### 
                                        #### PI - shifting layers ####
                                        #### --------------------- ####

    PI_shift = -6
    
     #### PI - North #### -- inhibit synapse
    
    #x_axis
    PI_N_synapse = Synapses(PI_Neurons, North_shifting_neurons,'w_PI_N:1',on_pre='v_post+=w_PI_N', name = 'PI_N_synapse')
    PI_N_synapse.connect(condition='abs(i-j)>=1')
    PI_N_synapse.w_PI_N = PI_shift
    

    #### PI - South #### -- inhibit synapse
    
    PI_S_synapse = Synapses(PI_Neurons, South_shifting_neurons,'w_PI_S:1',on_pre='v_post+=w_PI_S', name = 'PI_S_synapse')
    PI_S_synapse.connect(condition='abs(i-j)>=1')
    PI_S_synapse.w_PI_S = PI_shift
    
        
    #### PI - East#### -- inhibit synapse
        
    PI_E_synapse = Synapses(PI_Neurons, East_shifting_neurons,'w_PI_E:1',on_pre='v_post+=w_PI_E', name = 'PI_E_synapse')
    PI_E_synapse.connect(condition='abs(i-j)>=1')
    PI_E_synapse.w_PI_E = PI_shift
    
    #### PI - West #### -- inhibit synapse
        
    PI_W_synapse = Synapses(PI_Neurons, West_shifting_neurons,'w_PI_W:1',on_pre='v_post+=w_PI_W', name = 'PI_W_synapse')
    PI_W_synapse.connect(condition='abs(i-j)>=1')
    PI_W_synapse.w_PI_W = PI_shift
    
    #### PI - NorthEast #### -- inhibit synapse
        
    PI_NE_synapse = Synapses(PI_Neurons, NE_shifting_neurons,'w_PI_NE:1',on_pre='v_post+=w_PI_NE', name = 'PI_NE_synapse')
    PI_NE_synapse.connect(condition='abs(i-j)>=1')
    PI_NE_synapse.w_PI_NE = PI_shift
    
    #### PI - SouthEast #### -- inhibit synapse
        
    PI_SE_synapse = Synapses(PI_Neurons, SE_shifting_neurons,'w_PI_SE:1',on_pre='v_post+=w_PI_SE', name = 'PI_SE_synapse')
    PI_SE_synapse.connect(condition='abs(i-j)>=1')
    PI_SE_synapse.w_PI_SE = PI_shift
    
    #### PI - WestSouth #### -- inhibit synapse
        
    PI_WS_synapse = Synapses(PI_Neurons, WS_shifting_neurons,'w_PI_WS:1',on_pre='v_post+=w_PI_WS', name = 'PI_WS_synapse')
    PI_WS_synapse.connect(condition='abs(i-j)>=1')
    PI_WS_synapse.w_PI_WS = PI_shift
    
    #### PI - WestNorth #### -- inhibit synapse
        
    PI_WN_synapse = Synapses(PI_Neurons, WN_shifting_neurons,'w_PI_WN:1',on_pre='v_post+=w_PI_WN', name = 'PI_WN_synapse')
    PI_WN_synapse.connect(condition='abs(i-j)>=1')
    PI_WN_synapse.w_PI_WN = PI_shift
    
    #### PI - Stay #### -- inhibit synapse
        
    PI_stay_synapse = Synapses(PI_Neurons, Stay_layer,'w_PI_stay:1',on_pre='v_post+=w_PI_stay', name = 'PI_stay_synapse')
    PI_stay_synapse.connect(j='i')
    PI_stay_synapse.w_PI_stay = 2


    Stay_stay_layer = Synapses(Stay, Stay_layer,'w_stay_stay:1',on_pre='v_post+=w_stay_stay', name = 'stay_stay_synapse')
    Stay_stay_layer.connect()
    Stay_stay_layer.w_stay_stay = 0
  
                                     #### --------------------- ####
                                     #### shifting layers - IPI ####
                                     #### --------------------- ####
                                     
    shift_IPI = 8
    

    
     #### North - IPI #### -- exhibit synapse up shift
    IPI_N_synapse = Synapses(North_shifting_neurons,IPI_Neurons, 'w_IPI_N:1',on_pre='v_post+=w_IPI_N', name = 'IPI_N_synapse')
    IPI_N_synapse.connect(j='i+N_x_axis',skip_if_invalid = True)
    IPI_N_synapse.w_IPI_N = shift_IPI
     
    #### South - IPI #### -- inhibit synapse

    IPI_S_synapse = Synapses(South_shifting_neurons,IPI_Neurons, 'w_IPI_S:1',on_pre='v_post+=w_IPI_S', name = 'IPI_S_synapse')
    IPI_S_synapse.connect(j='i-N_x_axis',skip_if_invalid = True)
    IPI_S_synapse.w_IPI_S = shift_IPI    
    

     #### East - IPI #### -- exhibit synapse up shift +x
   
    IPI_E_synapse = Synapses(East_shifting_neurons,IPI_Neurons, 'w_IPI_E:1',on_pre='v_post+=w_IPI_E', name = 'IPI_E_synapse')
    for y in range(N_y_axis):
        IPI_E_synapse.connect(i=list(range(y*N_x_axis,(y+1)*N_x_axis-1)),j=list(range(y*N_x_axis+1,(y+1)*N_x_axis)))
    IPI_E_synapse.w_IPI_E = shift_IPI      
      
     #### West - IPI #### -- exhibit synapse up shift  -x
    
    IPI_W_synapse = Synapses(West_shifting_neurons,IPI_Neurons, 'w_IPI_W:1',on_pre='v_post+=w_IPI_W', name = 'IPI_W_synapse')
    for y in range(N_y_axis):
        IPI_W_synapse.connect(i=list(range(y*N_x_axis+1,(y+1)*N_x_axis)),j=list(range(y*N_x_axis,(y+1)*N_x_axis-1)))
    IPI_W_synapse.w_IPI_W = shift_IPI       
    
    
    ####NorthEast - IPI #### -- inhibit synapse +x,+y
    IPI_NE_synapse = Synapses(NE_shifting_neurons,IPI_Neurons, 'w_IPI_NE:1',on_pre='v_post+=w_IPI_NE', name = 'IPI_NE_synapse')
    for y in range(N_y_axis-1):
        IPI_NE_synapse.connect(i=list(range(y*N_x_axis,(y+1)*N_x_axis-1)),j=list(range((y+1)*N_x_axis+1,(y+2)*N_x_axis)))
    IPI_NE_synapse.w_IPI_NE = shift_IPI 

    
    ####SouthEast - IPI #### -- inhibit synapse +x,-y
    IPI_SE_synapse = Synapses(SE_shifting_neurons,IPI_Neurons, 'w_IPI_SE:1',on_pre='v_post+=w_IPI_SE', name = 'IPI_SE_synapse')
    for y in range(1,N_y_axis):
        IPI_SE_synapse.connect(i=list(range(y*N_x_axis,(y+1)*N_x_axis-1)),j=list(range((y-1)*N_x_axis+1,y*N_x_axis)))
    IPI_SE_synapse.w_IPI_SE = shift_IPI    
    
    
    ####SouthWest - IPI #### -- inhibit synapse -x,-y
    IPI_WS_synapse = Synapses(WS_shifting_neurons,IPI_Neurons, 'w_IPI_WS:1',on_pre='v_post+=w_IPI_WS', name = 'IPI_WS_synapse')
    for y in range(1,N_y_axis):
        IPI_WS_synapse.connect(i=list(range(y*N_x_axis+1,(y+1)*N_x_axis)),j=list(range((y-1)*N_x_axis,y*N_x_axis-1)))
    IPI_WS_synapse.w_IPI_WS = shift_IPI       
    
    
    ####NorthWest - IPI #### -- inhibit synapse -x,+y
    IPI_WN_synapse = Synapses(WN_shifting_neurons,IPI_Neurons, 'w_IPI_WN:1',on_pre='v_post+=w_IPI_WN', name = 'IPI_WN_synapse')
    for y in range(N_y_axis-1):
        IPI_WN_synapse.connect(i=list(range(y*N_x_axis+1,(y+1)*N_x_axis)),j=list(range((y+1)*N_x_axis,(y+2)*N_x_axis-1)))
    IPI_WN_synapse.w_IPI_WN = shift_IPI       
     
    ####Stay - IPI #### -- inhibit synapse -x,+y
    IPI_stay_synapse = Synapses(Stay_layer, IPI_Neurons, 'w_IPI_stay:1',on_pre='v_post+=w_IPI_stay', name = 'IPI_stay_synapse')
    IPI_stay_synapse.connect(j='i')
    IPI_stay_synapse.w_IPI_stay = shift_IPI       
 

                                #### --------------------- ####
                                 #### IPI and PI synapse ####
                                #### --------------------- ####
                                
    IPI_PI_synapse = Synapses(IPI_Neurons,PI_Neurons,'w_IPI_PI:1', on_pre='v_post+=w_IPI_PI',name='IPI_PI_synapse')
    IPI_PI_synapse.connect(j='i')
    IPI_PI_synapse.w_IPI_PI = 15
    
    
                                #### --------------------- ####
                                     #### Reset - IPI ####
                                #### --------------------- ####
                                
    PI_Reset_synapse = Synapses(Poisson_PI,IPI_Neurons,'w_poi_PI:1', on_pre='v_post+=w_poi_PI',name='IPI_Reset_synapse')
    PI_Reset_synapse.connect(j='i')
    PI_Reset_synapse.w_poi_PI=np.array(gaussian_spike(N_PI,495,20,0.03))  #initial condition: 495 is in the middle  #WTA
 
    #PI --> IPI
     
    ##--------------------Spike monitor----------------------##
    
    #spikemon
    spikemon_PI = SpikeMonitor(PI_Neurons)
    
    spikemon_IPI = SpikeMonitor(IPI_Neurons) #for debugging
    

    spikemon_HD = SpikeMonitor(HD_Neuron)
    
    spikemon_IHD = SpikeMonitor(IHD_Neuron)  # for debugging
    
    return IHD_in_synapse, str_inh_synapse,Poisson_straight,Go_straight_neuron,Go_straight,HD_str_synapse,str_IHD_synapse, sti_PI,sti,Poisson_Left,Poisson_Right,Poisson_Compass,Left_drive,Right_drive, HD_Neuron,IHD_Neuron, Left_shift_neuron, Left_shift_speed_neuron, Right_shift_neuron, Right_shift_speed_neuron, HD_Left_synapse, Left_IHD_synapse, HD_Right_synapse,Left_drive_synapse,Right_drive_synapse, Right_IHD_synapse, HD_IHD_synapse, Reset_synapse, HD_ex_synapse, HD_in_synapse,spikemon_HD,spikemon_IHD,Poisson_PI,PI_Neurons,IPI_Neurons,Directional_Neurons,HD_directional_synapse,directional_PI_synapse,PI_shifting_neurons,North_shifting_neurons,South_shifting_neurons,East_shifting_neurons,West_shifting_neurons,PI_ex_synapse,PI_in_synapse , PI_N_synapse   ,PI_S_synapse    ,PI_E_synapse,PI_W_synapse,IPI_N_synapse,IPI_S_synapse,IPI_E_synapse,IPI_W_synapse,IPI_PI_synapse,PI_Reset_synapse,spikemon_PI,spikemon_IPI,NE_shifting_neurons,SE_shifting_neurons,WS_shifting_neurons,WN_shifting_neurons ,PI_NE_synapse,PI_SE_synapse,PI_WS_synapse,PI_WN_synapse, IPI_NE_synapse,IPI_SE_synapse,IPI_WS_synapse,IPI_WN_synapse, Left_inh_synapse  ,   Right_IHD_synapse,IPI_in_synapse , IPI_stay_synapse,Stay_stay_layer ,Stay_layer ,Stay,PI_stay_synapse    


