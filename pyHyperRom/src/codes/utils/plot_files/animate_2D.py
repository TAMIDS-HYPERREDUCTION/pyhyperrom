from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class AnimatedContourPlot:
    def __init__(self, x, y, z_data):
        self.x = x
        self.y = y
        self.z_data = z_data

        # Calculate global min and max for color scaling
        self.z_min = np.min(np.min(z_data))
        self.z_max = np.max(np.max(z_data))

        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots()
        self.sc = self.contour_(self.z_data[:, 0])

        # Add a color bar for reference
        self.colorbar = plt.colorbar(self.sc, ax=self.ax)
        self.colorbar.set_label('Temperature')

    def contour_(self, Z_flat):
        # Generate the contour plot
        X, Y = np.meshgrid(self.x, self.y)
        Z = Z_flat.reshape(len(self.y), len(self.x))
        cp = self.ax.contourf(X, Y, Z, cmap=cm.coolwarm, levels=np.linspace(self.z_min, self.z_max, num=64*2))
        self.ax.set_aspect('equal', adjustable='box')
        return cp

    def update(self, frame):
        # Update the contour plot for a specific frame
        for c in self.sc.collections:
            c.remove()  # Remove old contours
        self.sc = self.contour_(self.z_data[:, frame])
        return self.sc

    def animate(self, interval=200):
        # Create the animation object
        self.ani = FuncAnimation(self.fig, self.update, frames=range(self.z_data.shape[1]), interval=interval)

    def display_animation(self, interval=200):
        # Display the animation as a HTML video
        self.animate(interval)
        return HTML(self.ani.to_jshtml())

    def save_animation(self, filename, writer='ffmpeg', interval=200):
        # Save the animation as a video file
        self.animate(interval)
        self.ani.save(filename, writer=writer)

# # Test data for the animation
# x = np.linspace(0, 10, 100)
# y = np.linspace(0, 10, 100)
# z_data = np.random.rand(100, 50)  # Random data for testing


