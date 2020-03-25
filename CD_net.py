

from brian2 import *
import numpy as np

# Collision network architecture
def CD_net(N_CD,N_vision,N_VD):
    # Poisson stimuli group for collision sensor
    Poisson_group = PoissonGroup(N_CD, rates=700 * Hz, name='CD_poissonGroup')
    Poisson_non_collision = PoissonGroup(1, rates=700 * Hz, name='NonCollision_poissonGroup')

    # Poisson stimuli for vision sensor
    Poisson_vision =PoissonGroup(N_VD, rates=700 * Hz, name='Vision_poisson')

    # Collision detection Neurons and its equation
    eqs = '''dv/dt = (-v) / (10*ms) : 1'''
    eqs2 = '''dv/dt = (-v) / (5*ms) : 1'''



    ##--------------------Neurons-----------------------##
    Collision_neuron = NeuronGroup(N_CD, eqs, threshold='v>1', reset='v=0', method='linear', name='CD_CollisionNeuron')
    Collision_or_not = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear', name='Collision_or_not')
    Non_collision_neuron = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear', name='non_Collision_Neuron')

    # non landmark neuron for later mismatch network
    Non_landmark_neuron = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear', name='non_landmark_neuron')

    ##--------------vision sensor related neurons-------#
    Color =  NeuronGroup(N_VD, eqs, threshold='v>1', reset='v=0', method='linear', name='VD_Color_Neuron')

    ##-------------- wall and landmark neurons-------##

    Wall = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear', name='Wall_Neuron')
    Landmark = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='linear', name='Landmark_Neuron')

    ##-------------- estimated landmark neurons-------##
    Estimated_landmark_neuron = NeuronGroup(1, eqs2, threshold='v>1', reset='v=0', method='linear', name='Estimated_landmark_Neuron')

    #############--------------------Synapse-----------------------############

    # Poisson 1-1 synapse with Neurons
    Poisson_synapse = Synapses(Poisson_group, Collision_neuron, 'w_poi:1', on_pre='v_post+=w_poi',
                               name='CD_poissonSynapse')
    Poisson_synapse.connect(j='i')
    #the poisson synapse weight will be determined by sensors
    # non collision possion with non collision neuron 1-1 synapses
    Poisson_non_synapse = Synapses(Poisson_non_collision, Non_collision_neuron, 'w_non_poi:1', on_pre='v_post+=w_non_poi',
                               name='nonCD_poissonSynapse')
    Poisson_non_synapse.connect()
    Poisson_non_synapse.w_non_poi = 5

    # non collision possion with non landmark neuron 1-1 synapses
    Poisson_nonlandmark_synapse = Synapses(Poisson_non_collision, Non_landmark_neuron, 'w_nonlandmark_poi:1',
                                   on_pre='v_post+=w_nonlandmark_poi',
                                   name='nonLandmark_poissonSynapse')
    Poisson_nonlandmark_synapse.connect()
    Poisson_nonlandmark_synapse.w_nonlandmark_poi = 1


    #poisson vision sensor group with vision sensor related neurons
    PoissonVision_synapse = Synapses(Poisson_vision, Color, 'w_vision_poi:1', on_pre='v_post+=w_vision_poi',
                                         name='vision_poi_synapse')
    PoissonVision_synapse.connect(j="i")

    # Self excitatory synapse
    Self_ex_synapse = Synapses(Collision_neuron, Collision_neuron, 'w_self:1', on_pre='v_post+=w_self',
                               name='CD_SelfExSynapse')

    Self_ex_synapse.connect(j="i")
    Self_ex_synapse.w_self = 0.6

    #COLOR selfex
    Self_ex_synapse_color = Synapses(Color, Color, 'w_self_color:1', on_pre='v_post+=w_self_color',
                               name='Color_SelfExSynapse')
    #Self_ex_synapse_color.connect(condition='abs(i-j)<1')
    #Self_ex_synapse_color.connect(i=[0, N_VD - 1], j=[N_VD - 1, 0])
    Self_ex_synapse_color.connect(j="i")
    Self_ex_synapse_color.w_self_color = 0.5

    # self_inhibitory
    Self_in_synapse = Synapses(Collision_neuron, Collision_neuron, 'w_nself:1', on_pre='v_post+=w_nself',
                               name='CD_SelfInSynapse')
    Self_in_synapse.connect(condition='abs(i-j)>=1')
    Self_in_synapse.w_nself = -0.6

    #COLOR self-in
    Self_in_synapse_color = Synapses(Color, Color, 'w_selfn_color:1', on_pre='v_post+=w_selfn_color',
                              name='Color_SelfInSynapse')
    Self_in_synapse_color.connect(condition='i!=j')
    #Self_in_synapse_color.connect(i=[0, N_VD - 1], j=[N_VD - 1, 0])
    Self_in_synapse_color.w_selfn_color = -0.7

    # CN to collision or not neuron
    Collide_or_not_synapse = Synapses(Collision_neuron, Collision_or_not, 'w_collide:1', on_pre='v_post+=w_collide',
                                      name='CD_collide_or_not')
    Collide_or_not_synapse.connect()
    Collide_or_not_synapse.w_collide = 1.4

    # CON to wall and landmark
    CON_wall_ex = Synapses(Collision_or_not, Wall, 'w_con_wall:1', on_pre='v_post+=w_con_wall', name='con_wall_ex')
    CON_wall_ex.connect()
    CON_wall_ex.w_con_wall = 0.55 #0.6
    CON_landmark_ex = Synapses(Collision_or_not, Landmark, 'w_con_landmark:1', on_pre='v_post+=w_con_landmark',
                               name='con_landmark_ex')
    CON_landmark_ex.connect()
    CON_landmark_ex.w_con_landmark = 0.4

    # collision or not neuron to non collision neuron
    CON_noncollision_synapse = Synapses(Collision_or_not,Non_collision_neuron,'w_non_collide:1',on_pre='v_post+=w_non_collide',
                                        name='CON_Noncollide_synapse')
    CON_noncollision_synapse.connect()
    CON_noncollision_synapse.w_non_collide = -10

    # non collision neuron to color neurons
    Non_collision_color_synapses = Synapses(Non_collision_neuron,Color,'w_noncollide_color:1',on_pre='v_post+=w_noncollide_color',
                                            name="Noncollide_color_synapse")
    Non_collision_color_synapses.connect()
    Non_collision_color_synapses.w_noncollide_color=-10

    #####red and not red neuron, synapses btw color and red/ notred
    Red = NeuronGroup(1,eqs,threshold='v>1', reset='v=0', method='linear',name="red")
    Notred = NeuronGroup(1,eqs,threshold='v>1', reset='v=0',method='linear',name='notred')

    color_red_synapses = Synapses(Color,Red,'w_color_red:1',on_pre='v_post+=w_color_red',name='color_red_synapse')
    color_red_synapses.connect(i=np.arange(5), j=0)
    color_red_synapses.w_color_red = 1.5

    color_notred_synapses = Synapses(Color,Notred,'w_color_notred:1',on_pre='v_post+=w_color_notred',name='color_notred_synapses')
    color_notred_synapses.connect(i=np.arange(5,10),j=0)
    color_notred_synapses.w_color_notred = 1.2

    # color re/notred excitory to wall and landmark
    Red_landmark_ex = Synapses(Red, Landmark, 'w_red_landmark:1', on_pre='v_post+=w_red_landmark',name='red_landmark_ex_synapse')
    Red_landmark_ex.connect()
    Red_landmark_ex.w_red_landmark = 0.8
    Notred_wall_ex = Synapses(Notred, Wall, 'w_notred_wall:1', on_pre='v_post+=w_notred_wall',name='notred_wall_ex_synapse')
    Notred_wall_ex.connect()
    Notred_wall_ex.w_notred_wall = 0.5 #0.4


    # color red/notred inhibitory to wall and landmark
    Red_wall_inh = Synapses(Red, Wall, 'w_red_wall:1', on_pre='v_post+=w_red_wall', name='red_wall_inh_synapse')
    Red_wall_inh.connect()
    Red_wall_inh.w_red_wall = -10

    Notred_landmark_inh = Synapses(Notred, Landmark, 'w_notred_landmark:1', on_pre='v_post+=w_notred_landmark',
                                   name='notred_landmark_inh_synapse')
    Notred_landmark_inh.connect()
    Notred_landmark_inh.w_notred_landmark = -10

    Notred_red_inh = Synapses(Notred, Red, 'w_notred_red:1', on_pre='v_post+=w_notred_red',
                                   name='notred_red_inh_synapse')
    Notred_red_inh.connect()
    Notred_red_inh.w_notred_red = -10


    # wall and landmark inter inhibit synapses
    wall_landmark_synapse = Synapses(Wall, Landmark, 'w_wall_landmark:1', on_pre='v_post+=w_wall_landmark')
    wall_landmark_synapse.connect()
    wall_landmark_synapse.w_wall_landmark = -4
    landmark_wall_synapse = Synapses(Landmark, Wall, 'w_landmark_wall:1', on_pre='v_post+=w_landmark_wall')
    landmark_wall_synapse.connect()
    landmark_wall_synapse.w_landmark_wall = -4

    #### synapse btw Landmark and non_landmark neuron
    Landmark_nonlandmark_synapse = Synapses(Landmark, Non_landmark_neuron, 'w_landmark_non:1',
                                           on_pre='v_post+=w_landmark_non',
                                           name='landmark_nonlandmarkSynapse')
    Landmark_nonlandmark_synapse.connect()
    Landmark_nonlandmark_synapse.w_landmark_non = -3

    ##--------------------Spike monitor----------------------##
    # spike monitor
    spikemon_CD = SpikeMonitor(Collision_neuron, name='spikemon_CD')
    spikemon_collision = SpikeMonitor(Collision_or_not, name='spikemon_Collide_or_not')
    spikemon_landmark = SpikeMonitor(Landmark, name='spikemon_landmark')
    spikemon_wall = SpikeMonitor(Wall, name='spikemon_wall')
    spikemon_poisson = SpikeMonitor(Poisson_vision, name='spikemom_pos_red')
    spikemon_noncollison = SpikeMonitor(Non_collision_neuron,name='non_collision_neuron')
    #statemon_estimatedlandmark = StateMonitor(Estimated_landmark_neuron,'v', record=0)
    #statemon_color = StateMonitor(Color, 'v',record=[0,5])
    spikemon_nonlandmark = SpikeMonitor(Non_landmark_neuron,name='spikemon_nonlandmark')
    spikemon_estimatedlandmark = SpikeMonitor(Estimated_landmark_neuron,name="spikemon_estimatedlandmark")
    spikemon_red = SpikeMonitor(Red,name="spikemon_red")
    spikemon_notred = SpikeMonitor(Notred,name="spikemon_notred")



    return Notred_red_inh,spikemon_red,spikemon_notred,spikemon_estimatedlandmark,spikemon_nonlandmark,wall_landmark_synapse, landmark_wall_synapse,Poisson_group,Poisson_vision,Collision_neuron,Collision_or_not,Color,Wall,Landmark,Poisson_synapse,PoissonVision_synapse,Self_ex_synapse,Self_ex_synapse_color,Self_in_synapse,Self_in_synapse_color,Collide_or_not_synapse,Red_landmark_ex,Red_wall_inh,Notred_landmark_inh,spikemon_CD,spikemon_collision,spikemon_landmark,spikemon_poisson,spikemon_wall,Poisson_non_collision,Non_collision_neuron,Poisson_non_synapse,CON_noncollision_synapse,Non_collision_color_synapses,spikemon_noncollison,Estimated_landmark_neuron, Non_landmark_neuron,Poisson_nonlandmark_synapse,Landmark_nonlandmark_synapse,CON_landmark_ex,CON_wall_ex,Notred_wall_ex,Red,Notred,color_notred_synapses,color_red_synapses
    #return spikemon_estimatedlandmark,wall_landmark_synapse, landmark_wall_synapse,Poisson_group,Poisson_vision,Collision_neuron,Collision_or_not,Color,Wall,Landmark,Poisson_synapse,PoissonVision_synapse,Self_ex_synapse,Self_ex_synapse_color,Self_in_synapse,Self_in_synapse_color,Collide_or_not_synapse,Red_landmark_ex,Red_wall_inh,Notred_landmark_inh,spikemon_CD,spikemon_collision,spikemon_landmark,spikemon_poisson,spikemon_wall,Poisson_non_collision,Non_collision_neuron,Poisson_non_synapse,CON_noncollision_synapse,Non_collision_color_synapses,spikemon_noncollison,Estimated_landmark_neuron,Landmark_estLanmark_synapse, Non_landmark_neuron,Poisson_nonlandmark_synapse,Landmark_nonlandmark_synapse,CON_landmark_ex,CON_wall_ex,Notred_wall_ex,Red,Notred,color_notred_synapses,color_red_synapses
