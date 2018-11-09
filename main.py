from sumolib import checkBinary

import os
import sys
import optparse
import random
import traci
import numpy as np
import keras
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model
from collections import deque

# Constantes
X = 0
Y = 0
JUNCAO_PRINCIPAL = '0'
SEMAFORO_PRINCIPAL = '0'

HRVGLG = 0  # w: horizontal red, vertical green and left turn green
HRVYLG = 1  # w: horizontal red, vertical yellow and left turn green
HRVRLG = 2  # w: horizontal red, vertical red and left turn green
HRVRLY = 3  # w: horizontal red, vertical red and left turn yellow

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

        # w: position P
        input_1 = Input(shape=(12, 12, 1))
        # w: primeira camada P
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        # w: segunda camada P
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        # w: parte da terceira camada
        x1 = Flatten()(x1)

        # w: speed V
        input_2 = Input(shape=(12, 12, 1))
        # w: primeira camada V
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        # w: segunda camada V
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        # w: parte da terceira camada
        x2 = Flatten()(x2)

        # w: latest traffic signal state L
        input_3 = Input(shape=(2, 1))
        # w: parte da terceira camada
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        # w: terceira camada
        x = Dense(128, activation='relu')(x)
        # w: quarta camada
        x = Dense(64, activation='relu')(x)
        # w: camada de saída
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

        return np.argmax(act_values[0])  # returns action

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
        # we need to import python modules from the $SUMO_HOME/tools directory
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
            from sumolib import checkBinary  # noqa
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def generate_routefile(self):
        random.seed(42)  # make tests reproducible
        N = 3600  # number of time steps
        # demand per second from different directions
        pRight = 1. / 15
        pLeft = 1. / 15
        pUp = 1. / 15
        pDown = 1. / 15
        with open("cross.rou.xml", "w") as routes:
            print('''<routes>
        <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left"  edges="52o 2i 1o 51i" />
        <route id="down"  edges="54o 4i 3o 53i" />
        <route id="up"    edges="53o 3i 4o 54i" />

    ''', file=routes)
            lastVeh = 0
            vehNr = 0
            # w: pra cada timestep gera probalisticamente se um veículo vai vir de uma das direções
            for i in range(N):
                if random.uniform(0, 1) < pRight:
                    print('    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                    lastVeh = i
                if random.uniform(0, 1) < pLeft:
                    print('    <vehicle id="left_%i"  type="SUMO_DEFAULT_TYPE" route="left"  depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pDown:
                    print('    <vehicle id="up_%i"    type="SUMO_DEFAULT_TYPE" route="up"    depart="%i"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pUp:
                    print('    <vehicle id="down_%i"  type="SUMO_DEFAULT_TYPE" route="down"  depart="%i"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
            print("</routes>", file=routes)

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def getState(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14

        junctionPosition = traci.junction.getPosition('0')[X]
        list = traci.edge.getIDList()
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1i')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4i')

        # w: inicializa matriz 12x12 com 0
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(12):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        for v in vehicles_road1:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[X] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[X] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('0')[Y]
        for v in vehicles_road3:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[Y] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[6 + 2 -
                               traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road4:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[Y] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        light = []
        if(traci.trafficlight.getPhase(SEMAFORO_PRINCIPAL) == 4):
            light = [1, 0]
        else:
            light = [0, 1]

        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 12, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 12, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1)

        return [position, velocity, lgts]

if __name__ == '__main__':
    sumoInt = SumoIntersection()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    options = sumoInt.get_options()

    if options.nogui:
    #if True:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    sumoInt.generate_routefile()

    # Main logic
    # parameters
    episodes = 2000
    batch_size = 32

    tg = 10
    ty = 6
    agent = DQNAgent()
    try:
        agent.load('Models/reinf_traf_control.h5')
    except:
        print('No models found')

    for e in range(episodes):
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network
        #log = open('log.txt', 'a')
        step = 0
        waiting_time = 0
        reward1 = 0
        reward2 = 0
        total_reward = reward1 - reward2
        stepz = 0
        action = 0

        traci.start([sumoBinary, "-c", "cross3ltl.sumocfg", '--start'])
        traci.trafficlight.setPhase("0", HRVGLG)
        traci.trafficlight.setPhaseDuration("0", 200)
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 3500:
            traci.simulationStep()
            state = sumoInt.getState()
            action = agent.act(state)
            light = state[2]

        traci.trafficlight.setPhase('0', 0)
        traci.simulationStep()

        traci.close(wait=False)

'''
            if(action == 0 and light[0][0][0] == 0):  # w: light = 1x2x1, light[0][0][0] == 0 significa que o semáforo horizontal está fechado
                # Transition Phase
                for i in range(6):
                    stepz += 1
                    traci.trafficlight.setPhase(SEMAFORO_PRINCIPAL, HRVYLG)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()
                for i in range(10):
                    stepz += 1
                    traci.trafficlight.setPhase(SEMAFORO_PRINCIPAL, HRVRLG)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()
                for i in range(6):
                    stepz += 1
                    traci.trafficlight.setPhase(SEMAFORO_PRINCIPAL, HRVRLY)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()

                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1i') + traci.edge.getLastStepVehicleNumber('2i')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '3i') + traci.edge.getLastStepHaltingNumber('4i')
                for i in range(10):
                    stepz += 1
                    traci.trafficlight.setPhase(SEMAFORO_PRINCIPAL, HGVRLG)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '3i') + traci.edge.getLastStepHaltingNumber('4i')
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()

            if(action == 0 and light[0][0][0] == 1):
                # Action Execution, no state change
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1i') + traci.edge.getLastStepVehicleNumber('2i')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '3i') + traci.edge.getLastStepHaltingNumber('4i')
                for i in range(10):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 4)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1i') + traci.edge.getLastStepVehicleNumber('2i')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '3i') + traci.edge.getLastStepHaltingNumber('4i')
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()

            if(action == 1 and light[0][0][0] == 0):
                # Action Execution, no state change
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4i') + traci.edge.getLastStepVehicleNumber('3i')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('1i')
                for i in range(10):
                    stepz += 1
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4i') + traci.edge.getLastStepVehicleNumber('3i')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('1i')
                    traci.trafficlight.setPhase('0', 0)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()

            if(action == 1 and light[0][0][0] == 1):
                for i in range(6):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 5)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()
                for i in range(10):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 6)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()
                for i in range(6):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 7)
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()

                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4i') + traci.edge.getLastStepVehicleNumber('3i')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '2i') + traci.edge.getLastStepHaltingNumber('1i')
                for i in range(10):
                    stepz += 1
                    traci.trafficlight.setPhase('0', 0)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4i') + traci.edge.getLastStepVehicleNumber('3i')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('1i')
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1i') + traci.edge.getLastStepHaltingNumber(
                        '2i') + traci.edge.getLastStepHaltingNumber('3i') + traci.edge.getLastStepHaltingNumber('4i'))
                    traci.simulationStep()

            new_state = sumoInt.getState()
            reward = reward1 - reward2
            agent.remember(state, action, reward, new_state, False)
            # Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
            if(len(agent.memory) > batch_size):
                agent.replay(batch_size)
            

        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))
        #log.write('episode - ' + str(e) + ', total waiting time - ' +
        #          str(waiting_time) + ', static waiting time - 338798 \n')
        #log.close()
        print('episode - ' + str(e) + ' total waiting time - ' + str(waiting_time))
        #agent.save('reinf_traf_control_' + str(e) + '.h5')
        traci.close(wait=False)'''

sys.stdout.flush()
