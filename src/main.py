import sys
import time
import optparse
import random
import numpy as np
import traci

from src.DeepQNetworkAgent import DeepQNetworkAgent
from src.SumoAgent import SumoAgent

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


def print_progress_bar(current, total, total_bars):
    percentage = int((current * 100) / total)
    bars = int((current / total) * total_bars)
    hifens = total_bars - bars - 1

    sys.stdout.write("\r" + "|" + ("|" * bars) + ("-" * hifens) + "| " + str(percentage) + "%")


if __name__ == '__main__':
    # We need to import python modules from the $SUMO_HOME/tools directory
    try:
        from sumolib import checkBinary
    except ImportError:
        sys.exit("Please declare environment variable 'SUMO_HOME' as the root directory of your sumo " +
                 "installation (it should contain folders 'bin', 'tools' and 'docs')")

    # Simulation parameters
    vehicle_generation_seed = 42
    vehicle_generation_probabilities = {'right': 1. / 15, 'left': 1. / 15, 'up': 1. / 15, 'down': 1. / 15}
    episode_timesteps = 3600
    episodes = 25
    batch_size = 32

    # DNN Agent
    # Initialize DNN with random weights
    # Initialize target network with same weights as DNN Network
    network_agent = DeepQNetworkAgent()
    sumo_agent = SumoAgent(vehicle_generation_probabilities, episode_timesteps, vehicle_generation_seed)

    time_mean = 0
    sim_start_time = time.clock()
    for episode_num in range(episodes):
        log = open('log.txt', 'a')
        epi_start_time = time.clock()

        steps = 0
        action = 0

        sumo_agent.start_sim()

        while sumo_agent.num_of_vehicles_still_in_simulation() > 0 and steps < episode_timesteps:
            # Gets current simulation state
            state = sumo_agent.get_state()
            # Gets an action accordingly to current state
            action = network_agent.act(state)
            # Executes action on simulation
            steps += sumo_agent.act_semaphore(action)
            # Gets next state
            next_state = sumo_agent.get_state()
            # Gets reward from last action
            reward = sumo_agent.calculate_reward()
            # Stores states for network tunning
            network_agent.remember(state, action, reward, next_state, False)

            if len(network_agent.memory) > batch_size:
                network_agent.replay(batch_size)

            print_progress_bar(steps, episode_timesteps, 20)
        print_progress_bar(episode_timesteps, episode_timesteps, 19)

        mem = network_agent.memory[-1]
        del network_agent.memory[-1]
        network_agent.memory.append((mem[0], mem[1], reward, mem[3], True))

        epi_end_time = time.clock()
        epi_time = epi_end_time - epi_start_time
        time_mean = ((time_mean * episode_num) + epi_time) / (episode_num + 1)

        waiting_time = sumo_agent.end_sim()

        print("\nEpisode: %d\n"
              "\tTotal waiting time: %d seconds\n"
              "\tEpisode length: %d seconds\n"
              "\tExpected sim end in: %d minutes" %
              (episode_num + 1, waiting_time, epi_time, (time_mean * ((episodes - episode_num) - 1)) / 60))

        log.write("Episode: %d \tTotal waiting time: %d\n" % (episode_num + 1, waiting_time))
        log.close()

    sim_end_time = time.clock()

    print("Total simulation time: %d minutes" % ((sim_end_time - sim_start_time) / 60))
