import sys
import optparse
import random
import numpy as np
import traci
import logging

# Constraints
VGHR = 0  # vertical green  horizontal red
VYHR = 1  # vertical yellow horizontal red
VRHG = 2  # vertical red    horizontal green
VRHY = 3  # vertical red    horizontal yellow

RED = 0
GREEN = 1

HORIZONTAL_GREEN = 0
VERTICAL_GREEN = 1


class SumoAgent:
    def __init__(self, episode_timesteps, controlled_tls_id, monitored_tls_ids=(), seed=None):
        logging.info('Controlled TLS: %s' % controlled_tls_id)
        logging.info('Monitored TLS ' + str(monitored_tls_ids))

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

        self.controlled_tls = {'id': controlled_tls_id, 'waiting_time': 0,
                               'horizontal_edge': None, 'vertical_edge': None}

        self.monitored_tls = []
        if type(monitored_tls_ids) == str:
            self.monitored_tls.append({'id': monitored_tls_ids, 'waiting_time': 0,
                                       'horizontal_edge': None, 'vertical_edge': None})
        else:
            for monitored_tls_id in monitored_tls_ids:
                self.monitored_tls.append({'id': monitored_tls_id, 'waiting_time': 0,
                                           'horizontal_edge': None, 'vertical_edge': None})

        self.yellow_light_time = 6
        self.green_light_time = 10

        self.horizontal_light_state = None
        self.state = None

        self.reward_moving = 0
        self.reward_halting = 0

    @staticmethod
    def generate_routefile():
        number_of_vehicles = {'UFF__UFF__retorno': 60,
                              'UFF__icarai_meio': 160,
                              'UFF__praia_icarai': 40,
                              'centro__praia_icarai': 720,
                              'icarai_meio__centro': 200,
                              'icarai_meio__icarai_praia': 80,
                              'praia_icarai__UFF': 400,
                              'praia_icarai__centro': 720,
                              'MAC__icarai_meio': 100,
                              'MAC__centro': 100,
                              'icarai_meio__MAC': 100}
        multiplier = 1

        with open("../sim/inga/inga.rou.xml", "w") as routes:
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">''', file=routes)
            print('''\t<route id="UFF__UFF__retorno" edges="Presidente_Pedreira#1 Presidente_Pedreira#2 Presidente_Pedreira#3 Nilo_Pecanha#1 Tiradentes#3 Tiradentes#4" color="127,255,212"/>''', file=routes)
            print('''\t<route id="UFF__icarai_meio" edges="Presidente_Pedreira#1 Presidente_Pedreira#2 Presidente_Pedreira#3 Presidente_Pedreira#4 Presidente_Pedreira#5 Presidente_Pedreira#6L Paulo_Alves#4 Paulo_Alves#5 Fagundes_Varela#1" color="cyan"/>''', file=routes)
            print('''\t<route id="UFF__praia_icarai" edges="Presidente_Pedreira#1 Presidente_Pedreira#2 Presidente_Pedreira#3 Presidente_Pedreira#4 Presidente_Pedreira#5 Presidente_Pedreira#6R Paulo_Alves#-1" color="blue"/>''', file=routes)
            print('''\t<route id="centro__praia_icarai" edges="Visconde_de_Morais#1 Visconde_de_Morais#2 Presidente_Pedreira#3 Presidente_Pedreira#4 Presidente_Pedreira#5 Presidente_Pedreira#6R Paulo_Alves#-1" color="red"/>''', file=routes)
            print('''\t<route id="icarai_meio__centro" edges="Fagundes_Varela#-1 Sao_Sebastiao#1" color="magenta"/>''', file=routes)
            print('''\t<route id="icarai_meio__icarai_praia" edges="Fagundes_Varela#-1 Tiradentes#1 Pereira_Nunes#1 Presidente_Pedreira#5 Presidente_Pedreira#6R Paulo_Alves#-1" color="250,235,215"/>''', file=routes)
            print('''\t<route id="praia_icarai__UFF" edges="Paulo_Alves#1 Paulo_Alves#2 Paulo_Alves#3 Paulo_Alves#4 Paulo_Alves#5 Tiradentes#1 Tiradentes#2 Tiradentes#3 Tiradentes#4" color="yellow"/>''', file=routes)
            print('''\t<route id="praia_icarai__centro" edges="Paulo_Alves#1 Paulo_Alves#2 Paulo_Alves#3 Paulo_Alves#4 Paulo_Alves#5 Sao_Sebastiao#1" color="green"/>''', file=routes)
            print('''\t<route id="MAC__icarai_meio" edges="Nilo_Pecanha#0 Presidente_Pedreira#4 Presidente_Pedreira#5 Presidente_Pedreira#6L Paulo_Alves#4 Paulo_Alves#5 Fagundes_Varela#1" color="255,255,255"/>''', file=routes)
            print('''\t<route id="MAC__centro" edges="Nilo_Pecanha#0 Presidente_Pedreira#4 Presidente_Pedreira#5 Presidente_Pedreira#6L Paulo_Alves#4 Paulo_Alves#5 Sao_Sebastiao#1" color="128,0,128"/>''', file=routes)
            print('''\t<route id="icarai_meio__MAC" edges="Fagundes_Varela#-1 Tiradentes#1 Pereira_Nunes#1 Pereira_Nunes#2" color="127,127,127"/>''', file=routes)
            print("")

            for flow in number_of_vehicles.keys():
                probability = (number_of_vehicles[flow] / 3600) * multiplier
                print('''\t<flow id="flow_''' + flow +
                      '''" type="DEFAULT_VEHTYPE" begin="0.00" end="3600.00" probability="''' +
                      str(probability) + '''"  route="''' + flow + '''"/>''', file=routes)
            print("</routes>", file=routes)

    @staticmethod
    def get_options():
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
        options, args = opt_parser.parse_args()
        return options

    def start_sim(self):
        traci.start([self.sumoBinary, "-c", "../sim/inga/inga.sumocfg", '--start', '--quit-on-end'])

        self.controlled_tls['waiting_time'] = 0
        self.__set_influenced_edges(self.controlled_tls)
        for tls in self.monitored_tls:
            tls['waiting_time'] = 0
            self.__set_influenced_edges(tls)

    @staticmethod
    def __set_influenced_edges(tls):
        edges = []

        for lane in traci.trafficlight.getControlledLanes(tls['id']):
            if lane[0] is ':':  # Walking areas ids starts with ':'
                continue

            if lane[:-2] not in edges:
                edges.append(lane[:-2])

        if len(edges) > 2:
            print("WARNING: TLS %s influences %d edges, it should influence 2" % (tls['id'], len(edges)))

        tls['horizontal_edge'] = edges[0]
        tls['vertical_edge'] = edges[1]

    def get_state(self):
        n_lanes = 8
        max_cell_dist = 8

        position_matrix = np.zeros((n_lanes, max_cell_dist))
        velocity_matrix = np.zeros((n_lanes, max_cell_dist))

        cell_length = 7
        speed_limit = 14

        vehicles_road1 = traci.edge.getLastStepVehicleIDs(self.controlled_tls['vertical_edge'])
        vehicles_road2 = traci.edge.getLastStepVehicleIDs(self.controlled_tls['horizontal_edge'])

        for v in vehicles_road1:
            next_tls_distance = self.__get_next_tls_distance(v)
            ind = int(next_tls_distance / cell_length)
            if ind < max_cell_dist:
                position_matrix[traci.vehicle.getLaneIndex(v)][(max_cell_dist - 1) - ind] = 1
                velocity_matrix[traci.vehicle.getLaneIndex(v)][(max_cell_dist - 1) - ind] = traci.vehicle.getSpeed(v) / speed_limit

        for v in vehicles_road2:
            next_tls_distance = self.__get_next_tls_distance(v)
            ind = int(next_tls_distance / cell_length)
            if ind < max_cell_dist:
                position_matrix[2 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocity_matrix[2 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speed_limit

        if traci.trafficlight.getPhase(self.controlled_tls['id']) == VRHG:
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
        if horizontal_light_state is None:
            horizontal_light_state = self.state[2][0][0][0]

        if horizontal_light_state == RED and action == HORIZONTAL_GREEN:
            # Vertical green -> horizontal green
            return self.__act_semaphore(VRHG, True, VYHR)
        elif horizontal_light_state == GREEN and action == HORIZONTAL_GREEN:
            # Horizontal green -> horizontal green
            return self.__act_semaphore(VRHG, True)
        elif horizontal_light_state == RED and action == VERTICAL_GREEN:
            # Vertical green -> vertical green
            return self.__act_semaphore(VGHR, False)
        elif horizontal_light_state == GREEN and action == VERTICAL_GREEN:
            # Horizontal green -> vertical green
            return self.__act_semaphore(VGHR, False, VRHY)

    def __act_semaphore(self, phase, moving_horizontal, yellow_phase=None):
        steps_elapsed = 0

        if yellow_phase is not None:
            # Sets vertical yellow (transition phase) for 6 seconds
            for i in range(self.yellow_light_time):
                traci.trafficlight.setPhase(self.controlled_tls['id'], yellow_phase)

                self.__update_waiting_times()

                traci.simulationStep()
                steps_elapsed += 1

        # Calculates reward using the halting cars in the halted edges and all the cars in the moving edges
        self.reward_moving = self.__get_num_of_moving_vehicles(not moving_horizontal)
        self.reward_halting = self.__get_num_of_halting_vehicles(moving_horizontal)

        for i in range(self.green_light_time):
            traci.trafficlight.setPhase(self.controlled_tls['id'], phase)

            self.__update_waiting_times()

            # Updates reward
            self.reward_moving += self.__get_num_of_moving_vehicles(not moving_horizontal)
            self.reward_halting += self.__get_num_of_halting_vehicles(moving_horizontal)

            traci.simulationStep()
            steps_elapsed += 1

        return steps_elapsed

    def calculate_reward(self):
        return self.reward_moving - self.reward_halting

    @staticmethod
    def num_of_vehicles_still_in_simulation():
        return traci.simulation.getMinExpectedNumber()

    def __get_next_tls_distance(self, vehicle):
        next_tls = traci.vehicle.getNextTLS(vehicle)[0]

        if next_tls[0] != self.controlled_tls['id']:
            return "Incorrect semaphore: The first position from getNextTLS isn't returning the expected TLS"

        return next_tls[2]

    def end_sim(self):
        traci.close(wait=False)

        return self.controlled_tls, self.monitored_tls

    def __update_waiting_times(self):
        self.controlled_tls['waiting_time'] += (traci.edge.getLastStepHaltingNumber(self.controlled_tls['vertical_edge']) +
                                                traci.edge.getLastStepHaltingNumber(self.controlled_tls['horizontal_edge']))
        for tls in self.monitored_tls:
            tls['waiting_time'] += (traci.edge.getLastStepHaltingNumber(tls['vertical_edge']) +
                                    traci.edge.getLastStepHaltingNumber(tls['horizontal_edge']))

    def __get_num_of_moving_vehicles(self, horizontal=True):
        if horizontal:
            return traci.edge.getLastStepVehicleNumber(self.controlled_tls['horizontal_edge'])
        else:
            return traci.edge.getLastStepVehicleNumber(self.controlled_tls['vertical_edge'])

    def __get_num_of_halting_vehicles(self, horizontal=True):
        if horizontal:
            return traci.edge.getLastStepVehicleNumber(self.controlled_tls['horizontal_edge'])
        else:
            return traci.edge.getLastStepVehicleNumber(self.controlled_tls['vertical_edge'])
