

from brian2 import *


def fusi_wall(PI_Neurons,Wall):
    
    plastic_syn_eq = '''
                    
                    dCa/dt = (-Ca/tau_ca) : 1 (clock-driven)                     #Calcium Potential
                    updrift = 1.0*(w_fusi_wall>theta_w) : 1
                    downdrift = 1.0*(w_fusi_wall<=theta_w) : 1
                    dw/dt = (alpha*updrift)-(beta*downdrift) : 1 (clock-driven) # internal weight variable
                    
                    w_fusi_wall:1
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
    
    preEq = '''
            up = 1. * (Ca>theta_upl) * (Ca<theta_uph)
            down = 1. *  (Ca>theta_downl) * (Ca<theta_downh)
            w_fusi_wall += wplus * up - wminus * down
            w_fusi_wall = clip(w_fusi_wall,w_min,w_max)
            '''  
            
    postEq = '''Ca += w_ca'''
    

    fusi_wall_synapse = Synapses(PI_Neurons,Wall,plastic_syn_eq,on_pre =preEq ,on_post = postEq)
    fusi_wall_synapse.connect()
    fusi_wall_synapse.wplus = 0.5
    fusi_wall_synapse.wminus = 0.3
    fusi_wall_synapse.theta_upl = 1.1
    fusi_wall_synapse.theta_uph = 10
    fusi_wall_synapse.theta_downh = 1
    fusi_wall_synapse.theta_downl = 0
    fusi_wall_synapse.alpha = 0.1*1/second
    fusi_wall_synapse.beta = 0.1*1/second
    fusi_wall_synapse.tau_ca = 4*ms
    fusi_wall_synapse.w_ca = 1
    fusi_wall_synapse.w_min = 0
    fusi_wall_synapse.w_max = 1
    fusi_wall_synapse.theta_w = 0.5

    
    return fusi_wall_synapse