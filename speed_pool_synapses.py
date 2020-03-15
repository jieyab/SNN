from brian2 import *

def stdp_errorCW(CW_neuron,Speed_Neuron_right):
    taupre = taupost = 10 * ms
    wmax = 10
    wmin = 0
    Apre = 10
    Apost = -Apre * taupre / taupost * 1.05
    cw_rightspeed_synapses = Synapses(Speed_Neuron_right, CW_neuron,
                                        '''w:1
                                        dapre/dt = -apre/taupre : 1 (clock-driven) 
                                        dapost/dt = -apost/taupost : 1 (clock-driven) ''',
                                        on_pre='''
                                        v_post += w
                                        apre += Apre
                                         w = clip(w+apost, 0, wmax)
                                        ''',
                                        on_post='''
                                        apost += Apost
                                        w = clip(w+apre, wmin, wmax)
                                        ''', method='linear')
    cw_rightspeed_synapses.connect(j="i")

    return cw_rightspeed_synapses

def stdp_errorCCW(CCW_neuron,Speed_Neuron_left):
    taupre = taupost = 10 * ms
    wmax = 10
    Apre = 10
    Apost = -Apre * taupre / taupost * 1.05
    cw_leftspeed_synapses = Synapses(Speed_Neuron_left, CCW_neuron,
                                        '''w:1
                                        dapre/dt = -apre/taupre : 1 (clock-driven) 
                                        dapost/dt = -apost/taupost : 1 (clock-driven) ''',
                                        on_pre='''
                                        v_post += w
                                        apre += Apre
                                         w = clip(w+apost, wmin, wmax)
                                        ''',
                                        on_post='''
                                        apost += Apost
                                        w = clip(w+apre, 0, wmax)
                                        ''', method='linear')
    cw_leftspeed_synapses.connect(j="i")

    return cw_leftspeed_synapses


