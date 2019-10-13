import sys
import time
import logging

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

    # Configure log
    logging.basicConfig(level=logging.DEBUG, filename='log.l', filemode='a', format='%(levelname)s - %(message)s')
    logging.info('Starting simulation')

    # Simulation parameters
    episode_timesteps = 3600
    episodes = 25
    batch_size = 32
    use_memory_palace = True

    # Network parameters
    gamma = 0.95
    epsilon = 0.1
    learning_rate = 0.0002

    # Log parameters
    logging.info('Batch size: %d' % batch_size)

    # DNN Agent
    # Initialize DNN with random weights
    # Initialize target network with same weights as DNN Network
    network_agent = DeepQNetworkAgent(gamma, epsilon, learning_rate, use_memory_palace)
    sumo_agent = SumoAgent(episode_timesteps,
                           "Presidente_Pedreira__Pereira_Nunes",
                           ("Presidente_Pedreira__Nilo_Pecanha", "Presidente_Pedreira__Paulo_Alves"))

    time_mean = 0
    sim_start_time = time.clock()

    for episode_num in range(episodes):
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

        controlled_tls, monitored_tls = sumo_agent.end_sim()

        print("\nEpisode: %d\n"
              "\tEpisode length: %d seconds\n"
              "\tExpected sim end in: %d minutes\n"
              "\tControlled TLS: %s\n"
              "\t\tTotal waiting time: %d seconds" %
              (episode_num + 1, epi_time,
               (time_mean * ((episodes - episode_num) - 1)) / 60,
               controlled_tls['id'], controlled_tls['waiting_time']))
        print("\tMonitored TLSs:")
        for tls in monitored_tls:
            print("\t\t%s: %d seconds" % (tls['id'], tls['waiting_time']))

        main_log_message = "%d\t%d" % (episode_num + 1, controlled_tls['waiting_time'])
        secondary_log_message = ""
        for tls in monitored_tls:
            secondary_log_message += "\t%d" % tls['waiting_time']
        logging.info(main_log_message + secondary_log_message)

    sim_end_time = time.clock()

    print("Total simulation time: %d minutes" % ((sim_end_time - sim_start_time) / 60))
