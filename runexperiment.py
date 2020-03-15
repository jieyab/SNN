from mainold import *
from mainExtendModel import*

def main():
    sim_number = 1
    # Simulation time for each experiment
    sim_time = 12000
    #calling training function for only use IMU to calibrate
    clientID  = run_training("_onlyIMU",sim_time)
    #clientID  = run_extendedmodel_training("_IMUHC",sim_time)
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

if __name__== "__main__":
    main()




