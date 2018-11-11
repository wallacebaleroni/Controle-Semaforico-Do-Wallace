from sumolib import checkBinary

import os
import sys
import time
import optparse
from collections import deque
import random
import numpy as np
import keras
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model
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


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.0002
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.action_size = 2

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        # Position P
        input_1 = Input(shape=(12, 12, 1))
        # First layer P
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        # Second layer P
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        # Part of the third layer
        x1 = Flatten()(x1)

        # Speed V
        input_2 = Input(shape=(12, 12, 1))
        # First layer V
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        # Second layer V
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        # Part of the third layer
        x2 = Flatten()(x2)

        # Latest traffic signal state L
        input_3 = Input(shape=(2, 1))
        # Part of the third layer
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        # Third layer
        x = Dense(128, activation='relu')(x)
        # Forth layer
        x = Dense(64, activation='relu')(x)
        # Output layer
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class SumoIntersection:
    def __init__(self):
        # We need to import python modules from the $SUMO_HOME/tools directory
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))  # Tutorial in tests
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))  # Tutorial in docs
            from sumolib import checkBinary  # noqa
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def generate_routefile(self):
        random.seed(42)  # Make tests reproducible
        episode_timesteps = 3600  # Number of time steps per episode
        # Demand per second from different directions
        p_right = 1. / 15
        p_left = 1. / 15
        p_up = 1. / 15
        p_down = 1. / 15

        with open("cross.rou.xml", "w") as routes:
            print('''<routes>
\t<vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
\t\t<route id="right" edges="51o 1i 2o 52i" />
\t\t<route id="left"  edges="52o 2i 1o 51i" />
\t\t<route id="down"  edges="54o 4i 3o 53i" />
\t\t<route id="up"    edges="53o 3i 4o 54i" />\n''', file=routes)

            vehicle_n = 0
            # Generates for each timestep vehicles entering
            for i in range(episode_timesteps):
                if random.uniform(0, 1) < p_right:
                    print('\t\t<vehicle id="right_%i"\ttype="SUMO_DEFAULT_TYPE" route="right"\tdepart="%i" />' % (
                        vehicle_n, i), file=routes)
                    vehicle_n += 1
                if random.uniform(0, 1) < p_left:
                    print('\t\t<vehicle id="left_%i"\ttype="SUMO_DEFAULT_TYPE" route="left"\tdepart="%i" />' % (
                        vehicle_n, i), file=routes)
                    vehicle_n += 1
                if random.uniform(0, 1) < p_down:
                    print('\t\t<vehicle id="up_%i"\ttype="SUMO_DEFAULT_TYPE" route="up"\tdepart="%i"/>' % (
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

    def get_state(self):
        position_matrix = np.zeros((12, 12))
        velocity_matrix = np.zeros((12, 12))

        cell_length = 7
        offset = 5  # Junction radius
        speed_limit = 14

        junction_position = traci.junction.getPosition(MAIN_JUNCTION)[X]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1i')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4i')

        for v in vehicles_road1:
            ind = int(abs(junction_position - offset - traci.vehicle.getPosition(v)[X]) / cell_length)
            if ind < 4:
                position_matrix[traci.vehicle.getLaneIndex(v)][3 - ind] = 1
                velocity_matrix[traci.vehicle.getLaneIndex(v)][3 - ind] = traci.vehicle.getSpeed(v) / speed_limit

        for v in vehicles_road2:
            ind = int(abs((junction_position - traci.vehicle.getPosition(v)[X] + offset)) / cell_length)
            if ind < 4:
                position_matrix[traci.vehicle.getLaneIndex(v)][ind] = 1
                velocity_matrix[traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speed_limit

        junction_position = traci.junction.getPosition(MAIN_SEMAPHORE)[Y]
        for v in vehicles_road3:
            ind = int(abs((junction_position - traci.vehicle.getPosition(v)[Y] - offset)) / cell_length)
            if ind < 4:
                position_matrix[traci.vehicle.getLaneIndex(v)][3 - ind] = 1
                velocity_matrix[traci.vehicle.getLaneIndex(v)][3 - ind] = traci.vehicle.getSpeed(v) / speed_limit

        for v in vehicles_road4:
            ind = int(abs((junction_position - traci.vehicle.getPosition(v)[Y] + offset)) / cell_length)
            if ind < 4:
                position_matrix[traci.vehicle.getLaneIndex(v)][ind] = 1
                velocity_matrix[traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speed_limit

        light = []
        if traci.trafficlight.getPhase(MAIN_SEMAPHORE) == VRHG:
            light = [1, 0]
        else:
            light = [0, 1]

        position = np.array(position_matrix)
        position = position.reshape(1, 12, 12, 1)

        velocity = np.array(velocity_matrix)
        velocity = velocity.reshape(1, 12, 12, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1)

        return [position, velocity, lgts]


if __name__ == '__main__':
    sumoInt = SumoIntersection()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    options = sumoInt.get_options()

    #if options.nogui:
    if True:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    sumoInt.generate_routefile()

    episodes = 100
    batch_size = 32

    tg = 10
    ty = 6
    agent = DQNAgent()
    try:
        agent.load('Models/reinf_traf_control.h5')
    except:
        print('No models found')

    time_mean = 0
    sim_start_time = time.clock()

    for e in range(episodes):
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network
        step = 0
        waiting_time = 0
        reward_moving = 0
        reward_halting = 0
        stepz = 0
        action = 0

        log = open('log.txt', 'a')
        epi_start_time = time.clock()

        traci.start([sumoBinary, "-c", "cross.sumocfg", '--start', '--quit-on-end'])
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 200)

        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 3600:
            traci.simulationStep()
            state = sumoInt.get_state()
            action = agent.act(state)
            horizontal_light_state = state[2][0][0][0]

            # Vertical green -> horizontal green
            if horizontal_light_state == 0 and action == 0:
                # Set vertical yellow (transition phase) for 6 timesteps
                for i in range(6):
                    traci.trafficlight.setPhase(MAIN_SEMAPHORE, VYHR)  # Vertical yellow, horizontal red

                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                                   + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))

                    traci.simulationStep()
                    stepz += 1

                # Calculates reward using the halting cars in the halted edges and all the cars in the moving edges
                reward_moving = traci.edge.getLastStepVehicleNumber(LEFT_EDGE) + traci.edge.getLastStepVehicleNumber(RIGHT_EDGE)
                reward_halting = traci.edge.getLastStepHaltingNumber(BOTTOM_EDGE) + traci.edge.getLastStepHaltingNumber(UPPER_EDGE)

                # Set horizontal green (action execution) for 10 timesteps
                for i in range(10):
                    traci.trafficlight.setPhase(MAIN_SEMAPHORE, VRHG)  # Vertical red, horizontal green

                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                                   + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    # Updates reward
                    reward_moving += traci.edge.getLastStepVehicleNumber(LEFT_EDGE) + traci.edge.getLastStepVehicleNumber(RIGHT_EDGE)
                    reward_halting += traci.edge.getLastStepHaltingNumber(BOTTOM_EDGE) + traci.edge.getLastStepHaltingNumber(UPPER_EDGE)

                    traci.simulationStep()
                    stepz += 1

            # Horizontal green -> horizontal green
            elif horizontal_light_state == 1 and action == 0:
                reward_moving = traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                reward_halting = traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')

                # Horizontal green
                for i in range(10):
                    traci.trafficlight.setPhase(MAIN_SEMAPHORE, VRHG)

                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                                   + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))

                    reward_moving += traci.edge.getLastStepVehicleNumber('1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward_halting += traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i')

                    traci.simulationStep()
                    stepz += 1
            # Vertical green -> vertical green
            elif horizontal_light_state == 0 and action == 1:
                reward_moving = traci.edge.getLastStepVehicleNumber('4i') + traci.edge.getLastStepVehicleNumber('3i')
                reward_halting = traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('1i')

                # Vertical green
                for i in range(10):
                    traci.trafficlight.setPhase(MAIN_SEMAPHORE, VGHR)

                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                                   + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))

                    reward_moving += traci.edge.getLastStepVehicleNumber('4i') + traci.edge.getLastStepVehicleNumber('3i')
                    reward_halting += traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('1i')

                    traci.simulationStep()
                    stepz += 1
            # Horizontal green -> vertical green
            elif horizontal_light_state == 1 and action == 1:
                # Horizontal yellow
                for i in range(6):
                    traci.trafficlight.setPhase(MAIN_SEMAPHORE, VRHY)
                    traci.simulationStep()
                    stepz += 1

                reward_moving = traci.edge.getLastStepVehicleNumber('4i') + traci.edge.getLastStepVehicleNumber('3i')
                reward_halting = traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('1i')

                # Vertical green
                for i in range(10):
                    traci.trafficlight.setPhase(MAIN_SEMAPHORE, VGHR)

                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber('2i')
                                   + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))

                    reward_moving += traci.edge.getLastStepVehicleNumber('4i') + traci.edge.getLastStepVehicleNumber('3i')
                    reward_halting += traci.edge.getLastStepHaltingNumber('2i') + traci.edge.getLastStepHaltingNumber('1i')

                    traci.simulationStep()
                    stepz += 1

            new_state = sumoInt.get_state()
            reward = reward_moving - reward_halting
            agent.remember(state, action, reward, new_state, False)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))

        epi_end_time = time.clock()
        epi_time = epi_end_time - epi_start_time
        time_mean = ((time_mean * e) + epi_time) / (e + 1)

        print("Episode: %d\tTotal waiting time: %d\n\tEpisode lenght: %d seconds\tExpected sim end in: %d seconds" % (e, waiting_time, epi_time, (time_mean * ((episodes - e) - 1))))

        log.write("Episode: %d\tTotal waiting time: %d\n" % (e, waiting_time))
        log.close()

        traci.close(wait=False)

    sim_end_time = time.clock()

    print("Total simulation time: %d" % (sim_end_time - sim_start_time))

sys.stdout.flush()
