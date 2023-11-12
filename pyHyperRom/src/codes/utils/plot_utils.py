from ..basic import *
from .plot_utils_style import *
from scipy import stats

class ThreeDPlot:

    def __init__(self, x, y, z, Z_flat, clr='b', sz=1.0, max_ticker=10):
        self.x = x
        self.y = y
        self.z = z
        self.Z_flat = Z_flat
        self.clr = clr
        self.sz = sz
        self.max_ticker = max_ticker

    def plot3D(self, hmap=True, save_file=False):
        """Generates a 3D scatter plot with optional heatmap and file saving."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Reshape Z to a 3D array for plotting
        Z3d = np.reshape(self.Z_flat, (len(self.z), len(self.y), len(self.x))).T
        xx, yy, zz = np.meshgrid(self.x, self.y, self.z)

        if hmap:
            sc = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=Z3d.flatten(), cmap=cm.coolwarm, s=self.sz)
            plt.colorbar(sc, shrink=0.5)
        else:
            ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=self.clr, s=self.sz)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.view_init(elev=10, azim=-60)

        if save_file:
            from pyevtk.hl import gridToVTK
            gridToVTK("/structured"+f"{i}"+"100",  x,  y,  z, pointData = {"temp" : Z3d})
        
        plt.show()

    def element3D(self, grid_size, xi, FOS, fsize=(8, 8)):

        """Generates a 3D grid with specific elements highlighted based on a mask."""
        fig = plt.figure(figsize=fsize)
        ax = fig.add_subplot(111, projection='3d')
        mask_xi = xi.astype(bool)

        # Draw the 3D grid and color the cells
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):  # Using depth as the third dimension
                    alpha_value = 0.1  # Default transparency
                    clr = (1, 0.5, 0, alpha_value)  # RGB color for orange
                    ax.bar3d(x, y, z, 1, 1, 1, shade=True, color=clr)

        # Color specific cells based on the mask
        for i in range(len(mask_xi)):
            if mask_xi[i]:
                x, y, z = FOS.e_n_2ij(i, el=True)  # Convert flat index to 3D index
                clr = (0.08627450980392157, 0.611764705882353, 0.6039215686274509, 1.0)  # Specific color
                ax.bar3d(x, y, z, 1, 1, 1, shade=True, color=clr)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=15, azim=30)

        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        plt.show()


class TwoDPlot:

    def __init__(self, x, y, Z_flat, clr='b', sz=1.0, max_ticker=5,fontsize=15,elev=15, azim=30):
        rcParams['font.size'] = fontsize
        self.x = x
        self.y = y
        self.Z_flat = Z_flat
        self.clr = clr
        self.sz = sz
        self.max_ticker = max_ticker
        self.elev=elev
        self.azim=azim

    def scatter_(self):
        """Generates a 2D scatter plot based on the provided x, y data points and color mapping."""
        X, Y = np.meshgrid(self.x, self.y)
        fig, ax = plt.subplots()
        sc = ax.scatter(X.flatten(), Y.flatten(), c=self.clr, s=self.sz)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.locator_params(nbins=self.max_ticker)
        plt.autoscale(enable=True, tight=True)
        plt.show()

    def contour_(self):
        """Generates a contour plot based on the provided x, y, and Z_flat data."""
        X, Y = np.meshgrid(self.x, self.y)
        Z = self.Z_flat.reshape(len(self.y), len(self.x))
        fig, ax = plt.subplots()
        cp = ax.contourf(X, Y, Z, cmap=cm.coolwarm)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.locator_params(nbins=self.max_ticker)
        plt.colorbar(cp)
        plt.show()

    def surface_(self, barscale=0.5):
        """Generates a 3D surface plot utilizing heatmap coloring based on the provided dataset."""
        X, Y = np.meshgrid(self.x, self.y)
        Z = self.Z_flat.reshape(len(self.y), len(self.x))

        sx = 4
        sy = sx * (np.max(self.y)-np.min(self.y))/ (np.max(self.x)-np.min(self.x))
        
        fig, ax = plt.subplots(figsize=(2*sx, 2*sy), subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.scatter(X, Y, Z,s=0.1)

        # Setting the number of ticks for the x-axis, y-axis, and z-axis
        ax.zaxis.set_major_locator(LinearLocator(self.max_ticker))
        ax.xaxis.set_major_locator(LinearLocator(self.max_ticker))
        ax.yaxis.set_major_locator(LinearLocator(self.max_ticker))
        
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.set_box_aspect(aspect=(1, sy/sx, 1))

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$T$')
        ax.grid(False)

        plt.colorbar(surf,shrink=barscale)
        plt.autoscale(enable=True, tight=True)
        plt.show()

    def element_(self, grid_size, xi, FOS, highlight_color='#169C9A', clr='orange'):
        """
        Generates a grid plot and highlights specific elements based on a mask.
        """
        fig, ax = plt.subplots()
        mask_xi = xi.astype(bool)

        # Draw the grid lines and initialize all cells with the default color
        for y in range(grid_size[1] + 1):
            ax.axhline(y, lw=0.5, color='white')
        for x in range(grid_size[0] + 1):
            ax.axvline(x, lw=0.5, color='white')

        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                rect = plt.Rectangle((x, grid_size[1] - y - 1), 1, 1, edgecolor='none', facecolor=clr)
                ax.add_patch(rect)

        # Highlight specific elements
        for i in range(len(mask_xi)):
            if mask_xi[i]:
                x,y = FOS.e_n_2ij(i, el=True)
                rect = plt.Rectangle((x,y), 1, 1, linewidth=1, edgecolor='none', facecolor=highlight_color)
                ax.add_patch(rect)

        # Set the aspect ratio and remove ticks and labels
        ax.set_aspect('equal', adjustable='box')
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        plt.xlim(0, grid_size[0])
        plt.ylim(0, grid_size[1])
        ax.grid(False)

        # plt.autoscale(tight=True)
        fig.tight_layout()
        plt.show()


class OneDPlot:

    """
    Class: OneDimensionalPlotter
    Overview: This class provides methods for creating 1D plots with the ability to plot lines, scatter plots,
              and to highlight specific elements of a line plot.

    Attributes:
    - x: 1D array for the x coordinates of the data points.
    - Z_flat: Flattened 1D array for the values at each data point for the color mapping.
    - ax: Matplotlib Axes object where the plot will be drawn. If None, a new figure and axes will be created.
    - clr: Color for the line or scatter points.
    - sz: Size of the scatter points.
    """

    def __init__(self, x, Z_flat, ax=None, clr='orange', sz=1.0, max_ticker=5, fontsize = 15, fsize=(5,4)):
        rcParams['font.size'] = fontsize
        self.x = x
        self.Z_flat = Z_flat
        self.ax = ax
        self.clr = clr
        self.sz = sz
        self.max_ticker = max_ticker

        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=fsize)
            self.ax.minorticks_on()


    def scatter_(self):
        """Generates a scatter plot on the provided axes."""
        self.ax.scatter(self.x, self.Z_flat, c=self.clr, s=self.sz)
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$T$')
        self.ax.locator_params(nbins=self.max_ticker)
        plt.autoscale(tight=True)


    def line_(self):
        """Generates a line plot on the provided axes."""
        cmap = plt.get_cmap("tab20")
        line_color = cmap(np.random.choice(range(20)))
        
        # cmap = plt.get_cmap("PuOr")
        # line_color = cmap(np.random.rand())

        self.ax.plot(self.x, self.Z_flat, color=line_color)
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$T$')
        self.ax.locator_params(nbins=self.max_ticker)
        plt.autoscale(tight=True)


    def element_(self, highlight_color='#169C9A', clr='orange', linewidth=4, fsize=(10, 0.5)):
        """
        Method: element
        Purpose: Generates a 1D line plot and highlights specific elements based on a mask.
        Parameters:
        - mask_xi: A boolean array indicating which elements to highlight.
        - highlight_color: The color used to highlight specific elements.
        - clr: The default color of the line plot.
        - linewidth: The width of the line.
        - figsize: The size of the figure.
        """
        mask_xi = self.Z_flat.astype(bool)
        self.fig, self.ax = plt.subplots(figsize=fsize)
        x_values = np.arange(0, len(self.x), 1)
        y_values = np.zeros_like(x_values)  # All y-values are 0 to represent a line on the x-axis

        # Draw the basic line plot
        self.ax.plot(x_values, y_values, color=clr, linewidth=linewidth, markevery=(0, 1))

        # Hide axis ticks and labels
        self.ax.set_yticklabels([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_xticks([])

        # Highlight specific elements
        for i in range(len(x_values) - 1):
            if mask_xi[i]:  # Change the color of specific line elements
                self.ax.plot(x_values[i:i+2], y_values[i:i+2], color=highlight_color, linewidth=linewidth)

        # Hide the spines
        for spine in ['top', 'right', 'left', 'bottom']:
            self.ax.spines[spine].set_visible(False)

        plt.autoscale(tight=True)
        plt.show()


def data_stats(data, show_histogram=True, show_boxplot=True, show_mean_std=True, ylabel = "$y$", xlabel = "$x$"):
    """
    Plot a combined boxplot and histogram of the data. Each element can be toggled on or off.
    
    :param data: array-like, The data to plot.
    :param show_histogram: bool, If True, show the histogram.
    :param show_boxplot: bool, If True, show the boxplot.
    :param show_mean_std: bool, If True, show the mean and standard deviation on the histogram.
    """

    # Calculate descriptive statistics
    descriptive_stats = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        # 'Mode': stats.mode(data).mode[0],
        # 'Mode Count': stats.mode(data).count[0],
        'Standard Deviation': np.std(data),
        'Variance': np.var(data)
    }

    # Determine the layout based on what is to be shown
    if show_histogram and show_boxplot:
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)}, figsize=(8, 6))
    elif show_histogram:
        fig, ax_hist = plt.subplots(figsize=(8, 6))
    elif show_boxplot:
        fig, ax_box = plt.subplots(figsize=(16, 2))
    else:
        raise ValueError("At least one of 'show_histogram' or 'show_boxplot' must be True.")
    
    # If the boxplot is to be shown
    if show_boxplot:
        ax_box.set_facecolor('#f5f5f5')
        # Create Box Plot with patch_artist=True
        boxprops = dict(linewidth=2.0, color='orange', facecolor='orange')
        meanprops = dict(marker='^', markeredgecolor='white', markersize = 10, markerfacecolor='white') if show_mean_std else None
        
        box = ax_box.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                             boxprops=boxprops,
                             whiskerprops=dict(linewidth=2.0, color='orange'),
                             capprops=dict(linewidth=2.0, color='orange'),
                             medianprops=dict(linewidth=2.0, color='white'),
                             meanprops=meanprops,
                             showmeans=show_mean_std,
                             flierprops=dict(marker='o', markerfacecolor='#ffb300', markeredgecolor='#ffb300', markersize=5, linestyle='none'))

        # Remove unwanted spines and ticks from the boxplot
        ax_box.set_yticklabels([])
        ax_box.set_yticks([])
        ax_box.spines['top'].set_visible(False)
        ax_box.spines['right'].set_visible(False)
        ax_box.spines['left'].set_visible(False)
        ax_box.set_xlabel(xlabel)

    # If the histogram is to be shown
    if show_histogram:
        ax_hist.set_facecolor('#f5f5f5')
        # Creating the histogram
        n, bins, patches = ax_hist.hist(data, bins=30, linewidth=2, edgecolor='orange', color='orange', alpha=0.5)

        if show_mean_std:
            # Calculate descriptive statistics
            mean_val = np.mean(data)
            std_dev_val = np.std(data)
            # Annotate the mean and standard deviation on the histogram
            ax_hist.axvline(mean_val, color='black', linestyle='dashed', linewidth=2)
            ax_hist.axvline(mean_val - std_dev_val, color='#169C9A', linestyle='dashed', linewidth=2)
            ax_hist.axvline(mean_val + std_dev_val, color='#169C9A', linestyle='dashed', linewidth=2)
            # Add vertical text for mean and standard deviation
            # ax_hist.text(mean_val, np.interp(mean_val, bins[:-1], n), f'$\mu={mean_val:.2f}$', rotation=90, ha='right', va='center', color='black')
            ax_hist.text(mean_val, ax_hist.get_ylim()[1] / 2, f'$\mu={mean_val:.2f}$', rotation=90, ha='right', va='center', color='black')
            # ax_hist.text(mean_val - std_dev_val, np.interp(mean_val, bins[:-1], n), f'$\sigma={std_dev_val:.2f}$', rotation=90, ha='right', va='center', color='#169C9A', fontsize=10)
            # ax_hist.text(mean_val + std_dev_val, np.interp(mean_val, bins[:-1], n), f'$\sigma={std_dev_val:.2f}$', rotation=90, ha='right', va='center', color='#169C9A', fontsize=10)

        # Hide the spines and ticks of the histogram
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.set_xlabel(xlabel)
        ax_hist.set_ylabel(ylabel)

    # Show the final plot
    plt.tight_layout()
    plt.show()

    return descriptive_stats

