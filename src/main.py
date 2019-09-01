import sys
import time

from src.DeepQNetworkAgent import DeepQNetworkAgent
from src.SumoAgent import SumoAgent


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
    episode_timesteps = 3600
    episodes = 25
    batch_size = 32
    use_memory_palace = True

    # DNN Agent
    # Initialize DNN with random weights
    # Initialize target network with same weights as DNN Network
    network_agent = DeepQNetworkAgent(use_memory_palace)
    sumo_agent = SumoAgent(episode_timesteps)

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

            if use_memory_palace:
                if network_agent.get_memory_size() > batch_size:
                    network_agent.replay(batch_size)
            else:
                if len(network_agent.memory) > batch_size:
                    network_agent.replay(batch_size)

            print_progress_bar(steps, episode_timesteps, 20)
        print_progress_bar(episode_timesteps, episode_timesteps, 19)

        epi_end_time = time.clock()
        epi_time = epi_end_time - epi_start_time
        time_mean = ((time_mean * episode_num) + epi_time) / (episode_num + 1)

        waiting_time = sumo_agent.end_sim()

        print("\nEpisode: %d\n"
              "\tTotal waiting time: %d seconds\n"
              "\tEpisode length: %d seconds\n"
              "\tExpected sim end in: %d minutes" %
              (episode_num + 1, waiting_time, epi_time, (time_mean * ((episodes - episode_num) - 1)) / 60))

        log.write("Episode:\t%d\tTotal waiting time:\t%d\n" % (episode_num + 1, waiting_time))
        log.close()

    sim_end_time = time.clock()

    print("Total simulation time: %d minutes" % ((sim_end_time - sim_start_time) / 60))
