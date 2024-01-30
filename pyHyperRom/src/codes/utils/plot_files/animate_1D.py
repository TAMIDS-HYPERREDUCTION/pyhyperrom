import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


class AnimatedPlot:
    def __init__(self, x, y_data, x_lim=None):
        """
        Initialize the AnimatedPlot object.

        :param x: array-like, The x data for the plot.
        :param y_data: array-like, The y data for the plot, should be 2D where each row is a frame.
        :param x_lim: tuple, The x limits for the plot.
        """
        self.x = x
        self.y_data = y_data
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(x, y_data[0])
        
        # Set the x and y limits
        if x_lim:
            self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_data.min(), y_data.max())

    def update(self, frame):
        """
        Update the plot for a specific frame.

        :param frame: int, The frame number.
        """
        self.line.set_ydata(self.y_data[frame])
        return self.line,

    def animate(self, interval=200):
        """
        Create the animation object.
        
        :param interval: int, Delay between frames in milliseconds.
        """
        self.ani = FuncAnimation(self.fig, self.update, frames=range(len(self.y_data)), blit=True, interval=interval)

    def display_animation(self, interval=200):
        """
        Display the animation as a HTML video.
        
        :param interval: int, Delay between frames in milliseconds.
        """
        self.animate(interval)
        return HTML(self.ani.to_jshtml())
    
    def save_animation(self, filename, writer='ffmpeg', interval=200):
        """
        Save the animation as a video file.

        :param filename: str, The name of the file to save the animation as.
        :param writer: str, The writer to use to save the animation.
        :param interval: int, Delay between frames in milliseconds. Lower the number FASTER is the simulation.
        """
        self.animate(interval)
        self.ani.save(filename, writer=writer)



## Usage

#animated_plot = AnimatedPlot(x, reconst_FOM.T) #y_data
#animated_plot.display_animation(interval=20)  # Faster animation
#animated_plot.save_animation('animation_fast.mp4', interval=20)

#```