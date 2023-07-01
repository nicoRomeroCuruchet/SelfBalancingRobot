import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import numpy as np

# Create a single QApplication instance
app = QApplication([])

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
    steps_antes_de_done = env.steps_antes_de_done
    
    number_of_episode = np.arange(1.,len(episode_times)+1)
    
    plt.ion()
    fig = plt.figure(figsize=(15,9))
    ax1 = fig.add_subplot(331)
    ax1.set_xlabel('N° Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episodes Total Reward')
    ax1.plot(number_of_episode,episode_rewards)
    ax2 = fig.add_subplot(332)
    ax2.set_xlabel('N° Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episodes Length')
    ax2.plot(number_of_episode,episode_lengths)
    ax3 = fig.add_subplot(333)
    ax3.set_xlabel('N° Episode')
    ax3.set_ylabel('Episode Time')
    ax3.set_title('Episodes Times')
    ax3.plot(number_of_episode,episode_times)
    ax4 = fig.add_subplot(334)
    ax4.set_xlabel('N° Episode')
    ax4.set_ylabel('Initial Angle')
    ax4.set_title('Episodes Initial Angle')
    ax4.plot(episode_initial_angle)#number_of_episode,episode_initial_angle)
    ax4.plot(number_of_episode, env.threshold_angle*np.ones(len(number_of_episode)))
    ax4.plot(number_of_episode, -env.threshold_angle*np.ones(len(number_of_episode)))
    ax5 = fig.add_subplot(335)
    ax5.set_xlabel('N° Episode')
    ax5.set_ylabel('Final Angle')
    ax5.set_title('Episodes Final Angle')
    ax5.plot(episode_final_angle)#number_of_episode,episode_final_angle)
    ax5.plot(number_of_episode, env.threshold_angle*np.ones(len(number_of_episode)))
    ax5.plot(number_of_episode, -env.threshold_angle*np.ones(len(number_of_episode)))
    ax6 = fig.add_subplot(336)
    ax6.set_xlabel('N° Episode')
    ax6.set_ylabel('Delay')
    ax6.set_title('Delay between reset and 1st state measurement')
    ax6.plot(delays)
    ax7 = fig.add_subplot(337)
    ax7.set_xlabel('N° Episode')
    ax7.set_ylabel('N° of steps before 1st measure after reset')
    ax7.set_title('Steps before 1st measure')
    ax7.plot(steps_before_1st_measure)
    ax8 = fig.add_subplot(338)
    ax8.set_xlabel('N° Episode')
    ax8.set_ylabel('Steps antes de done')
    ax8.set_title('Steps antes de done')
    ax8.plot(steps_antes_de_done)
    
    plt.show(block=True)



def plot_debugging(env):
    fin = 50
    step_time = env.step_time#[0:fin]
    step_angle = env.step_angle#[0:fin]
    step_action = env.step_action#[0:fin]
    step_done = env.step_done#[0:fin]
    measures = env.measures
    measure_times = env.measure_times

    time0 = 0
    y = []
    x = range(1,len(step_time) + 1)
    for k in range(0,len(step_time)):
        y += [step_time[k] - time0]
        time0 = step_time[k]
    
    
    data = [[(x,y)], 
            [(measure_times, measures), (step_time, step_angle)]]
    
    data3 = []
    for k in range(0, len(measure_times)):
        x = [measure_times[k], measure_times[k]]
        y = [1,-1]
        data3.append((x, y))
    data3.append((step_time, step_action))
    data.append(data3)
    data.append([(step_time, step_done)])
    titles = ['Step time', 'Angle', 'Action', 'Done']
    ylabels = ['time'    , 'Angle', 'Action', 'Done']
    xlabels = ['N° Step' , 'Time',  'Time'  , 'Time']
    plot_pg(data, titles= titles, ylabels=ylabels, xlabels=xlabels)
    # plot_pg(x, y, title = 'Step time', ylabel = 'time', xlabel = 'N° Step')

    '''
    plt.ion()
    fig = plt.figure(figsize=(9,9))
    ax1 = fig.add_subplot(221)
    ax1.set_xlabel('N° Step')
    ax1.set_ylabel('time')
    ax1.set_title('Step time')
    time0 = 0
    aux = []
    for k in range(0,len(step_time)):
        aux += [step_time[k] - time0]
        time0 = step_time[k]
    ax1.plot(aux)
    ax2 = fig.add_subplot(222)
    # ax2.set_xlabel('N° Step')
    ax2.set_xlabel('Times')
    ax2.set_ylabel('Angle')
    ax2.set_title('Angle')
    ax2.plot(measure_times, measures,'--')
    ax2.plot(step_time, step_angle,'o')
    # ax2.plot(step_time, step_action)
    ax3 = fig.add_subplot(223)
    ax3.set_xlabel('Times')
    ax3.set_ylabel('Action')
    ax3.set_title('Action')
    # ax3.plot(step_action)
    for k in range(0, len(measure_times)):
        x = [measure_times[k], measure_times[k]]
        y = [1,-1]
        ax3.plot(x, y,'r')
    ax3.plot(step_time, step_action, '.--')
    # ax3 = fig.add_subplot(223)
    # ax3.set_xlabel('Time')
    # ax3.set_ylabel('Angle measured')
    # ax3.set_title('Angle measured')
    # ax3.plot(measure_times, measures)
    ax4 = fig.add_subplot(224)
    ax4.set_xlabel('Times')
    ax4.set_ylabel('Done')
    ax4.set_title('Done')
    ax4.plot(step_time, step_done)

    plt.show(block=True)
    '''


# def plot_pg(layout, x1, y1, x2 = [], y2 = [], title = None, ylabel = None, xlabel = None, legend1 = [], legend2 = []):
#     # Create plot window object
#     plt = layout.addPlot(title=title)
#     # Showing x and y grids
#     plt.showGrid(x = True, y = True)
#     # Adding legend
#     plt.addLegend()
#     # Set properties of the label for y axis
#     plt.setLabel('left', ylabel)#, units ='y')
#     # Set properties of the label for x axis
#     plt.setLabel('bottom', xlabel)#, units ='s')
#     # Plotting line in green color with dot symbol as x, not a mandatory field
#     line1 = plt.plot(x1, y1, pen ='g', symbol ='x', symbolPen ='g', symbolBrush = 0.2, name =legend1)
#     if x2:
#         # Plotting line2 with blue color with dot symbol as o
#         line2 = plt.plot(x2, y2, pen ='b', symbol ='o', symbolPen ='b', symbolBrush = 0.2, name =legend2)

def plot_pg(data, titles=None, ylabels=None, xlabels=None, legends=None):
    """
    Plot multiple subplots using PyQtGraph.

    Args:
        data (list): List of data for subplots. Each element represents a subplot and can contain multiple lines.
                     Each line is specified as a tuple of (x, y) data.
        titles (list, optional): List of subplot titles. Defaults to None.
        ylabels (list, optional): List of y-axis labels. Defaults to None.
        xlabels (list, optional): List of x-axis labels. Defaults to None.
        legends (list, optional): List of legends for each line in the subplots. Defaults to None.
    """

    num_plots = len(data)

    # Create the graphics layout widget and set the number of rows and columns
    layout = pg.GraphicsLayoutWidget()
    layout.resize(800, 600)
    layout.setWindowTitle('Subplots')

    # Create subplots based on the number of plots
    plots = []
    for i in range(num_plots):
        plot = layout.addPlot(row=i // 2, col=i % 2, title=titles[i] if titles else None)
        plot.showGrid(x=True, y=True)
        plot.setLabel('left', ylabels[i] if ylabels else None)
        plot.setLabel('bottom', xlabels[i] if xlabels else None)

        # Plot the data and add legend
        for j, d in enumerate(data[i]):
            legend = legends[j] if legends else 'None'
            plot.plot(d[0], d[1], pen=(j, len(data[i])), symbol='x', symbolPen=(j, len(data[i])), symbolBrush=0.2, name=legend)

        plots.append(plot)
    
    layout.show()
    # Start the Qt event loop
    app.exec_()
