import matplotlib.pyplot as plt
import numpy as np

def plot_episodes(env):
    """ Plot function to study signals involved during the training of an environment env.

        Args:
            env (gymnasium.Env): environment we wish to analyze. """
        
    episode_rewards = env.episode_reward
    episode_lengths = env.episode_length
    episode_times = env.episode_time
    episode_initial_angle = env.initial_angle
    episode_final_angle = env.final_angle
    delays = env.delays
    steps_before_1st_measure = env.steps_before_1st_measure
    x_initial = env.initial_x
    y_initial = env.initial_y
    x_final = env.final_x
    y_final = env.final_y
    
    number_of_episode = np.arange(1.,len(episode_times)+1)
    
    plt.ion()
    fig, axs = plt.subplots(3, 3, figsize=(15, 9))

    axs[0, 0].set_xlabel('N° Episode')
    axs[0, 0].set_ylabel('Episode Reward')
    axs[0, 0].set_title('Episodes Total Reward')
    axs[0, 0].plot(number_of_episode, episode_rewards, '.')

    axs[0, 1].set_xlabel('N° Episode')
    axs[0, 1].set_ylabel('Episode Length')
    axs[0, 1].set_title('Episodes Length')
    axs[0, 1].plot(number_of_episode, episode_lengths, '.')

    axs[0, 2].set_xlabel('N° Episode')
    axs[0, 2].set_ylabel('Episode Time')
    axs[0, 2].set_title('Episodes Times')
    axs[0, 2].plot(number_of_episode, episode_times, '.')

    axs[1, 0].set_xlabel('N° Episode')
    axs[1, 0].set_ylabel('Initial Angle')
    axs[1, 0].set_title('Episodes Initial Angle')
    axs[1, 0].plot(number_of_episode, env.threshold_angle * np.ones(len(number_of_episode)), 'r')
    axs[1, 0].plot(number_of_episode, -env.threshold_angle * np.ones(len(number_of_episode)), 'r')
    axs[1, 0].plot(episode_initial_angle, 'o--')

    axs[1, 1].set_xlabel('N° Episode')
    axs[1, 1].set_ylabel('Final Angle')
    axs[1, 1].set_title('Episodes Final Angle')
    axs[1, 1].plot(number_of_episode, env.threshold_angle * np.ones(len(number_of_episode)), 'r')
    axs[1, 1].plot(number_of_episode, -env.threshold_angle * np.ones(len(number_of_episode)), 'r')
    axs[1, 1].plot(episode_final_angle, 'o--')

    axs[1, 2].set_xlabel('N° Episode')
    axs[1, 2].set_ylabel('Delay')
    axs[1, 2].set_title('Delay between reset and 1st state measurement')
    axs[1, 2].plot(delays)

    axs[2, 0].set_xlabel('N° Episode')
    axs[2, 0].set_ylabel('N° of steps before 1st measure after reset')
    axs[2, 0].set_title('Steps before 1st measure')
    axs[2, 0].plot(steps_before_1st_measure)

    axs[2, 1].set_xlabel('N° Episode')
    axs[2, 1].set_ylabel('X_inicial / Y_inicial')
    axs[2, 1].set_title('Posición Inicial')
    axs[2, 1].plot(env.threshold_position * np.ones(len(number_of_episode)), 'r')
    axs[2, 1].plot(-env.threshold_position * np.ones(len(number_of_episode)), 'r')
    axs[2, 1].plot(x_initial, 'o--')
    axs[2, 1].plot(y_initial, 'o--')

    axs[2, 2].set_xlabel('N° Episode')
    axs[2, 2].set_ylabel('X_final / Y_final')
    axs[2, 2].set_title('Posición Final')
    axs[2, 2].plot(env.threshold_position * np.ones(len(number_of_episode)), 'r', label='Threshold')
    axs[2, 2].plot(-env.threshold_position * np.ones(len(number_of_episode)), 'r', label='-Threshold')
    axs[2, 2].plot(x_final, 'o--', label='x_final')
    axs[2, 2].plot(y_final, 'o--', label='y_final')
    axs[2, 2].legend()

    # plt.show(block=True)


def plot_debugging(env):
    step_time = env.step_time
    step_angle = env.step_angle
    step_action = env.step_action
    step_done = env.step_done
    measures = env.measures
    measure_times = env.measure_times

    plt.ion()
    plt.figure()
    plt.plot(measure_times, measures, '.--')
    plt.xlabel('Time')
    plt.ylabel('Angle Measured')
    plt.title('Measures')

    

    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))

    axs[0, 0].set_xlabel('N° Step')
    axs[0, 0].set_ylabel('time')
    axs[0, 0].set_title('Step time')
    aux = np.diff(step_time)
    axs[0, 0].plot(aux)

    axs[0, 1].set_xlabel('Times')
    axs[0, 1].set_ylabel('Angle')
    axs[0, 1].set_title('Angle')
    axs[0, 1].plot(measure_times, measures, '--')
    axs[0, 1].scatter(step_time, step_angle)

    axs[1, 0].set_xlabel('Times')
    axs[1, 0].set_ylabel('Action')
    axs[1, 0].set_title('Action')
    axs[1, 0].vlines(measure_times, -1, 1, colors='r')
    axs[1, 0].plot(step_time, step_action, '.--')

    axs[1, 1].set_xlabel('Times')
    axs[1, 1].set_ylabel('Done')
    axs[1, 1].set_title('Done')
    axs[1, 1].plot(step_time, step_done)

    # plt.show(block=True)