import sys
import time
import optparse
import random
import numpy as np
import traci

# Constraints
X = 0
Y = 1
MAIN_JUNCTION = '0'
MAIN_SEMAPHORE = '0'

LEFT_EDGE = '1i'
RIGHT_EDGE = '2i'
BOTTOM_EDGE = '3i'
UPPER_EDGE = '4i'

VGHR = 0  # vertical green  horizontal red
VYHR = 1  # vertical yellow horizontal red
VRHG = 2  # vertical red    horizontal green
VRHY = 3  # vertical red    horizontal yellow

RED = 0
GREEN = 1

HORIZONTAL_GREEN = 0
VERTICAL_GREEN = 1


class NewSumoAgent:
    def __init__(self, vehicle_generation_prababilities, episode_timesteps, seed=None):
        self.vehicle_generation_probabilities = vehicle_generation_prababilities
        self.episode_timesteps = episode_timesteps  # Number of time steps per episode
        if seed is not None:
            random.seed(seed)  # Make tests reproducible

        # We need to import python modules from the $SUMO_HOME/tools directory
        try:
            from sumolib import checkBinary
        except ImportError:
            sys.exit("Please declare environment variable 'SUMO_HOME' as the root directory of your sumo " +
                     "installation (it should contain folders 'bin', 'tools' and 'docs')")

        self.options = self.get_options()
        if self.options.nogui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')
        self.generate_routefile()

        self.yellow_light_time = 6
        self.green_light_time = 10

        self.horizontal_light_state = None
        self.state = None

        self.waiting_time = 0
        self.reward_moving = 0
        self.reward_halting = 0

    def generate_routefile(self):
        # Demand per second from different directions
        p_right = self.vehicle_generation_probabilities['right']
        p_left = self.vehicle_generation_probabilities['left']
        p_up = self.vehicle_generation_probabilities['up']
        p_down = self.vehicle_generation_probabilities['down']

        with open("sim/cross/cross.rou.xml", "w") as routes:
            print('''<routes>''', file=routes)
            print('''\t<vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>''', file=routes)
            print('''\t\t<route id="right" edges="51o 1i 2o 52i"/>''', file=routes)
            print('''\t\t<route id="left"  edges="52o 2i 1o 51i"/>''', file=routes)
            print('''\t\t<route id="down"  edges="54o 4i 3o 53i"/>''', file=routes)
            print('''\t\t<route id="up"    edges="53o 3i 4o 54i"/>\n''', file=routes)

            vehicle_n = 0
            # Generates for each timestep vehicles entering
            for i in range(self.episode_timesteps):
                if random.uniform(0, 1) < p_right:
                    print('\t\t<vehicle id="right_%i"\ttype="SUMO_DEFAULT_TYPE" route="right"\tdepart="%i" />' % (
                        vehicle_n, i), file=routes)
                    vehicle_n += 1
                if random.uniform(0, 1) < p_left:
                    print('\t\t<vehicle id="left_%i"\ttype="SUMO_DEFAULT_TYPE" route="left"\tdepart="%i" />' % (
                        vehicle_n, i), file=routes)
                    vehicle_n += 1
                if random.uniform(0, 1) < p_down:
                    print('\t\t<vehicle id="up_%i"  \ttype="SUMO_DEFAULT_TYPE" route="up"\t\tdepart="%i"/>' % (
                        vehicle_n, i), file=routes)
                    vehicle_n += 1
                if random.uniform(0, 1) < p_up:
                    print('\t\t<vehicle id="down_%i"\ttype="SUMO_DEFAULT_TYPE" route="down"\tdepart="%i"/>' % (
                        vehicle_n, i), file=routes)
                    vehicle_n += 1
            print("</routes>", file=routes)

    def get_options(self):
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
        options, args = opt_parser.parse_args()
        return options

    def start_sim(self):
        self.waiting_time = 0
        self.reward_moving = 0
        self.reward_halting = 0

        traci.start([self.sumoBinary, "-c", "sim/cross/cross.sumocfg", '--start', '--quit-on-end'])
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 200)

    def get_state(self):
        n_lanes = 8
        max_cell_dist = 8
        position_matrix = np.zeros((n_lanes, max_cell_dist))
        velocity_matrix = np.zeros((n_lanes, max_cell_dist))

        cell_length = 7
        offset = 11  # Junction radius
        speed_limit = 14

        junction_position = traci.junction.getPosition(MAIN_JUNCTION)[X]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs(LEFT_EDGE)
        vehicles_road2 = traci.edge.getLastStepVehicleIDs(RIGHT_EDGE)
        vehicles_road3 = traci.edge.getLastStepVehicleIDs(BOTTOM_EDGE)
        vehicles_road4 = traci.edge.getLastStepVehicleIDs(UPPER_EDGE)

        for v in vehicles_road1:
            ind = int(abs(junction_position - offset - traci.vehicle.getPosition(v)[X]) / cell_length)
            if ind < max_cell_dist:
                position_matrix[traci.vehicle.getLaneIndex(v)][7 - ind] = 1
                velocity_matrix[traci.vehicle.getLaneIndex(v)][7 - ind] = traci.vehicle.getSpeed(v) / speed_limit

        for v in vehicles_road2:
            ind = int(abs((junction_position - traci.vehicle.getPosition(v)[X] + offset)) / cell_length)
            if ind < max_cell_dist:
                position_matrix[2 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocity_matrix[2 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speed_limit

        junction_position = traci.junction.getPosition(MAIN_SEMAPHORE)[Y]
        for v in vehicles_road3:
            ind = int(abs((junction_position - traci.vehicle.getPosition(v)[Y] - offset)) / cell_length)
            if ind < max_cell_dist:
                position_matrix[6 + traci.vehicle.getLaneIndex(v)][7 - ind] = 1
                velocity_matrix[6 + traci.vehicle.getLaneIndex(v)][7 - ind] = traci.vehicle.getSpeed(v) / speed_limit

        for v in vehicles_road4:
            ind = int(abs((junction_position - traci.vehicle.getPosition(v)[Y] + offset)) / cell_length)
            if ind < max_cell_dist:
                position_matrix[4 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocity_matrix[4 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speed_limit

        light = []
        if traci.trafficlight.getPhase(MAIN_SEMAPHORE) == VRHG:
            light = [1, 0]
        else:
            light = [0, 1]

        position = np.array(position_matrix)
        position = position.reshape(1, 8, 8, 1)

        velocity = np.array(velocity_matrix)
        velocity = velocity.reshape(1, 8, 8, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1)

        self.state = [position, velocity, lgts]
        return self.state

    def act_semaphore(self, action, horizontal_light_state=None):
        steps = 0
        if horizontal_light_state is None:
            horizontal_light_state = self.state[2][0][0][0]

        waiting_time_now = 0
        stepz_elapsed = 0

        if horizontal_light_state == RED and action == HORIZONTAL_GREEN:
            # Vertical green -> horizontal green
            waiting_time_now, reward_moving, reward_halting, stepz_elapsed = self._act_semaphore(VRHG, True, VYHR)
        elif horizontal_light_state == GREEN and action == HORIZONTAL_GREEN:
            # Horizontal green -> horizontal green
            waiting_time_now, reward_moving, reward_halting, stepz_elapsed = self._act_semaphore(VRHG, True)
        elif horizontal_light_state == RED and action == VERTICAL_GREEN:
            # Vertical green -> vertical green
            waiting_time_now, reward_moving, reward_halting, stepz_elapsed = self._act_semaphore(VGHR, False)
        elif horizontal_light_state == GREEN and action == VERTICAL_GREEN:
            # Horizontal green -> vertical green
            waiting_time_now, reward_moving, reward_halting, stepz_elapsed = self._act_semaphore(VGHR, False, VRHY)

        self.waiting_time += waiting_time_now
        steps += stepz_elapsed

        return waiting_time_now, reward_moving, reward_halting, stepz_elapsed

    def _act_semaphore(self, phase, moving_horizontal, yellow_phase=None):
        waiting_time = 0
        stepz = 0

        if yellow_phase is not None:
            # Sets vertical yellow (transition phase) for 6 seconds
            for i in range(self.yellow_light_time):
                traci.trafficlight.setPhase(MAIN_SEMAPHORE, yellow_phase)

                waiting_time += get_waiting_time()

                traci.simulationStep()
                stepz += 1

        # Calculates reward using the halting cars in the halted edges and all the cars in the moving edges
        reward_moving = get_num_of_moving_vehicles(moving_horizontal)
        reward_halting = get_num_of_halting_vehicles(not moving_horizontal)

        for i in range(self.green_light_time):
            traci.trafficlight.setPhase(MAIN_SEMAPHORE, phase)

            waiting_time += get_waiting_time()
            # Updates reward
            reward_moving += get_num_of_moving_vehicles(moving_horizontal)
            reward_halting += get_num_of_halting_vehicles(not moving_horizontal)

            traci.simulationStep()
            stepz += 1

        return waiting_time, reward_moving, reward_halting, stepz

    def calculate_reward(self):
        return self.reward_moving - self.reward_halting

    def end_sim(self):
        traci.close(wait=False)

        return self.waiting_time


def get_waiting_time():
    return (traci.edge.getLastStepHaltingNumber(LEFT_EDGE) +
            traci.edge.getLastStepHaltingNumber(RIGHT_EDGE) +
            traci.edge.getLastStepHaltingNumber(BOTTOM_EDGE) +
            traci.edge.getLastStepHaltingNumber(UPPER_EDGE))


def get_num_of_moving_vehicles(horizontal=True):
    if horizontal:
        return (traci.edge.getLastStepVehicleNumber(LEFT_EDGE) +
                traci.edge.getLastStepVehicleNumber(RIGHT_EDGE))
    else:
        return (traci.edge.getLastStepVehicleNumber(UPPER_EDGE) +
                traci.edge.getLastStepVehicleNumber(BOTTOM_EDGE))


def get_num_of_halting_vehicles(horizontal=True):
    if horizontal:
        return (traci.edge.getLastStepHaltingNumber(LEFT_EDGE) +
                traci.edge.getLastStepHaltingNumber(RIGHT_EDGE))
    else:
        return (traci.edge.getLastStepHaltingNumber(BOTTOM_EDGE) +
                traci.edge.getLastStepHaltingNumber(UPPER_EDGE))
