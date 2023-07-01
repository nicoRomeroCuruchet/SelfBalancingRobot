import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

# Create a single QApplication instance
app = QApplication([])

def plot_pg(layout, x, y, title=None, ylabel=None, xlabel=None, legend=None):
    # Create plot window object
    plt = layout.addPlot(title=title)
    # Showing x and y grids
    plt.showGrid(x=True, y=True)
    # Adding legend
    plt.addLegend()
    # Set properties of the label for y axis
    plt.setLabel('left', ylabel)
    # Set properties of the label for x axis
    plt.setLabel('bottom', xlabel)
    # Plotting line in green color with dot symbol as x
    line = plt.plot(x, y, pen='g', symbol='x', symbolPen='g', symbolBrush=0.2, name=legend)

# Example data for two subplots
x1 = [1, 2, 3, 4, 5]
y1 = [0.1, 0.5, 0.3, 0.6, 0.2]

x2 = [1, 2, 3, 4, 5]
y2 = [0.5, 0.2, 0.4, 0.8, 0.6]

# Create the graphics layout widget
layout = pg.GraphicsLayoutWidget()
layout.resize(800, 600)
layout.setWindowTitle('Subplots')

# Plot first subplot
plot_pg(layout, x1, y1, title='Step time 1', ylabel='time 1', xlabel='N° Step 1', legend='Line 1')

# Plot second subplot
plot_pg(layout, x2, y2, title='Step time 2', ylabel='time 2', xlabel='N° Step 2', legend='Line 2')

layout.show()

# Start the Qt event loop after adding all the subplots
app.exec_()

