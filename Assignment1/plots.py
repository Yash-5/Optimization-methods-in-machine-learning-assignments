from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_surface(x_range, y_range, func, x_label='', y_label='', z_label='', title='', step_x = 0.01, step_y = 0.01, show=False):
    x = np.arange(x_range[0], x_range[1], step_x)
    y = np.arange(y_range[0], y_range[1], step_y)
    X, Y = np.meshgrid(x, y)
    z = np.array([func(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = z.reshape(X.shape)
    fig = plt.figure()
    fig_plot = fig.add_subplot(111, projection='3d')
    fig_plot.plot_surface(X, Y, Z)
    fig_plot.set_xlabel(x_label)
    fig_plot.set_ylabel(y_label)
    fig_plot.set_zlabel(z_label)
    fig_plot.set_title(title)
    plt.savefig(title + "_surface.png")
    if show:
        plt.show()
        
def plot_contour(x_range, y_range, func, fig, subplot_num, x_label='', y_label='', z_label='', title='', step_x = 0.01, step_y = 0.01, show=False):
    x = np.arange(x_range[0], x_range[1], step_x)
    y = np.arange(y_range[0], y_range[1], step_y)
    X, Y = np.meshgrid(x, y)
    z = np.array([func(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = z.reshape(X.shape)
    fig_plot = fig.add_subplot(subplot_num)
    fig_plot.contour(X, Y, Z)
    fig_plot.set_xlabel(x_label)
    fig_plot.set_ylabel(y_label)
    fig_plot.set_title(title)
    return fig_plot