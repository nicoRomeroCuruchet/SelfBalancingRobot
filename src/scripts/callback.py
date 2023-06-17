from stable_baselines3.common.callbacks import BaseCallback
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

class PlotCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=1, **kwargs):
        super().__init__(verbose)
        self._polt1 = None
        self._polt2 = None
        self.step = 0
        try:
            self.check_freq = check_freq
        except:
            self.check_freq = 10

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # This last update is done to include the last episode reward, which isn't available yet during the last _on_step callback

        episode_rewards = self.training_env.envs[0].get_episode_rewards()
        episode_lengths = self.training_env.envs[0].get_episode_lengths()
        episode_times   = self.training_env.envs[0].get_episode_times()
        number_of_episode = np.arange(1.,len(episode_rewards)+1)
        print("Episodios totales: ", number_of_episode[-1])

        self._polt1[0].set_data(number_of_episode,episode_rewards)
        self._polt1[-2].relim()
        self._polt1[-2].autoscale_view(True,True,True)
        self._polt1[-1].canvas.draw()
        self._polt2[0].set_data(number_of_episode,episode_lengths)
        self._polt2[-2].relim()
        self._polt2[-2].autoscale_view(True,True,True)
        self._polt2[-1].canvas.draw()
        # plt.savefig('logs/rewards_and_lengths_final.svg', format='svg')
        plt.show(block=True)

    def _on_step(self) -> bool:
        self.step += 1
        if self.step % self.check_freq == 0:
            episode_rewards = self.training_env.envs[0].get_episode_rewards()
            episode_lengths = self.training_env.envs[0].get_episode_lengths()
            episode_timestamps   = self.training_env.envs[0].get_episode_times()
            t_previous = 0
            episode_times = []
            for t in episode_timestamps:
                episode_times += [t - t_previous]
                t_previous = t
            number_of_episode = np.arange(1.,len(episode_rewards)+1)
            
            if self._polt1 is None: # make the plot
                plt.ion()
                fig = plt.figure(figsize=(12,4))
                ax1 = fig.add_subplot(131)
                ax1.set_xlabel('N° Episode')
                ax1.set_ylabel('Episode Reward')
                ax1.set_title('Episodes Total Reward')
                line1, = ax1.plot(number_of_episode,episode_rewards)
                self._polt1 = (line1, ax1, fig)
                ax2 = fig.add_subplot(132)
                ax2.set_xlabel('N° Episode')
                ax2.set_ylabel('Episode Length')
                ax2.set_title('Episodes Length')
                line2, = ax2.plot(number_of_episode,episode_lengths)
                self._polt2 = (line2, ax2, fig)
                ax3 = fig.add_subplot(133)
                ax3.set_xlabel('N° Episode')
                ax3.set_ylabel('Episode Time')
                ax3.set_title('Episodes Times')
                line3, = ax3.plot(number_of_episode,episode_times)
                self._polt3 = (line3, ax3, fig)
                plt.savefig('logs/rewards_and_lengths.svg', format='svg')
                plt.show()
            else: # update and rescale the plot
                self._polt1[0].set_data(number_of_episode,episode_rewards)
                self._polt1[-2].relim()
                self._polt1[-2].autoscale_view(True,True,True)
                self._polt1[-1].canvas.draw()
                self._polt2[0].set_data(number_of_episode,episode_lengths)
                self._polt2[-2].relim()
                self._polt2[-2].autoscale_view(True,True,True)
                self._polt2[-1].canvas.draw()
                self._polt3[0].set_data(number_of_episode,episode_times)
                self._polt3[-2].relim()
                self._polt3[-2].autoscale_view(True,True,True)
                self._polt3[-1].canvas.draw()
                plt.savefig('logs/rewards_and_lengths.svg', format='svg')



class DebuggingPlotCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=1, **kwargs):
        super().__init__(verbose)
        self._polt1 = None
        self._polt2 = None
        self.step = 0
        try:
            self.check_freq = check_freq
        except:
            self.check_freq = 10

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # This last update is done to include the last episode reward, which isn't available yet during the last _on_step callback

        episode_initial_angle = self.training_env.envs[0].initial_angle
        episode_final_angle   = self.training_env.envs[0].final_angle
        delays                = self.training_env.envs[0].delays
        number_of_episode = np.arange(1.,len(episode_rewards)+1)
        print("Episodios totales: ", number_of_episode[-1])

        self._polt1[0].set_data(number_of_episode,episode_initial_angle)
        self._polt1[-2].relim()
        self._polt1[-2].autoscale_view(True,True,True)
        self._polt1[-1].canvas.draw()
        self._polt2[0].set_data(number_of_episode,episode_final_angle)
        self._polt2[-2].relim()
        self._polt2[-2].autoscale_view(True,True,True)
        self._polt2[-1].canvas.draw()
        self._polt3[0].set_data(number_of_episode,delays)
        self._polt3[-2].relim()
        self._polt3[-2].autoscale_view(True,True,True)
        self._polt3[-1].canvas.draw()
        # plt.savefig('logs/debugging_final.svg', format='svg')
        plt.show(block=True)

    def _on_step(self) -> bool:
        self.step += 1
        if self.step % self.check_freq == 0:
            episode_initial_angle = self.training_env.envs[0].initial_angle
            episode_final_angle   = self.training_env.envs[0].final_angle
            delays                = self.training_env.envs[0].delays
            number_of_episode = np.arange(1.,len(episode_rewards)+1)
            
            if self._polt1 is None: # make the plot
                plt.ion()
                fig = plt.figure(figsize=(12,4))
                ax1 = fig.add_subplot(131)
                ax1.set_xlabel('N° Episode')
                ax1.set_ylabel('Initial Angle')
                ax1.set_title('Episodes Initial Angle')
                line1, = ax1.plot(number_of_episode,episode_initial_angle)
                self._polt1 = (line1, ax1, fig)
                ax2 = fig.add_subplot(132)
                ax2.set_xlabel('N° Episode')
                ax2.set_ylabel('Final Angle')
                ax2.set_title('Episodes Final Angle')
                line2, = ax2.plot(number_of_episode,episode_final_angle)
                self._polt2 = (line2, ax2, fig)
                ax3 = fig.add_subplot(133)
                ax3.set_xlabel('N° Episode')
                ax3.set_ylabel('Delay')
                ax3.set_title('Delay between reset and 1st state measurement')
                line3, = ax3.plot(number_of_episode,delays)
                self._polt3 = (line3, ax3, fig)
                plt.savefig('logs/debugging.svg', format='svg')
                plt.show()
            else: # update and rescale the plot
                self._polt1[0].set_data(number_of_episode,episode_initial_angle)
                self._polt1[-2].relim()
                self._polt1[-2].autoscale_view(True,True,True)
                self._polt1[-1].canvas.draw()
                self._polt2[0].set_data(number_of_episode,episode_final_angle)
                self._polt2[-2].relim()
                self._polt2[-2].autoscale_view(True,True,True)
                self._polt2[-1].canvas.draw()
                self._polt3[0].set_data(number_of_episode,delays)
                self._polt3[-2].relim()
                self._polt3[-2].autoscale_view(True,True,True)
                self._polt3[-1].canvas.draw()
                plt.savefig('logs/debugging.svg', format='svg')