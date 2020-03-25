from mainold import *
from mainExtendModel import *
from mainOnlyIMU import *
from mainOnlyHC import *

def main():
    sim_number = 1
    # Simulation time for each experiment
    sim_time = 12000

    #--------- calling training function for benchmark model--------#
    #clientID  = run_benchmark_training("_noIMUnoHC",sim_time)

    #--------- calling training function for only use IMU to calibrate --------#
    #clientID  = run_onlyimu_training("_onlyIMU",sim_time)

    #--------- calling training function for only use head direction correction to calibrate --------#
    #clientID  = run_onlyhc_training("_onlyHC",sim_time)

    #--------- calling training function for using both IMU and head direction correction model
    clientID  = run_extendedmodel_training("_IMUHC",sim_time)

    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)

if __name__== "__main__":
    main()




