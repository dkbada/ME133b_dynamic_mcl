import bisect
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math               import inf, pi, sin, cos, sqrt, ceil, dist

from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep

######################################################################
#
#   World Definitions (No Fixes Needed)
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (0, 25)
(ymin, ymax) = (0, 25)

obstacles = prep(MultiPolygon([
    Polygon([[2, 14], [5, 14], [5, 23], [2, 23]]),
    Polygon([[7, 23], [11, 23], [11, 20], [7, 20]]),
    Polygon([[7, 17], [11, 17], [11, 14], [7, 14]]),
    Polygon([[14, 23], [19, 23], [19, 20], [16, 20], [16, 14], [14, 14]]),
    Polygon([[21, 23], [23, 23], [23, 14], [18, 14], [18, 17], [21, 17]]),
    Polygon([[2, 11], [8, 11], [8, 8], [6, 8], [6, 2], [4, 2], [4, 8], [2, 8]]),
    Polygon([[10, 11], [15, 11], [15, 8], [12, 8], [12, 5], [15, 5], [15, 2], [10, 2]]),
    Polygon([[17, 11], [23, 11], [23, 8], [21, 8], [21, 5], [20, 5], [20, 2], [17, 2]])
]))

# Start/goal (x, y)
(xstart, ystart) = (6, 1)

######################################################################
#
#   Visualization Class (No Fixes Needed)
#
class Visualization:
    def __init__(self, title=" Visualization"):
        #interactive mode
        plt.ion() 

        # Create a new axes, enable the grid, and set axis limits.
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self.ax.grid(True)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")

        # Show the obstacles.
        for poly in obstacles.context.geoms:
            self.ax.plot(*poly.exterior.xy, 'k-', linewidth=2)

        #stuff that will get updated constantly
        #populates particles
        self.particles_scatter = self.ax.scatter([], [], s=10, alpha=0.35)

        #estimate pose marker (x_hat, y_hat) and direction arrow.
        self.est_point, = self.ax.plot([], [], marker="o", markersize=6)
        self.est_heading,  = self.ax.plot([], [], linewidth=2)   # heading line for estimate
        # True pose marker (optional usage).
        self.true_point, = self.ax.plot([], [], marker="x", markersize=6)
        self.true_heading, = self.ax.plot([], [], linewidth=2)   # heading line for true
        

    def redraw(self):
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

######################################################################
#
#   Update methods
#

    #particles_xy: (N,2) array of x and y
    #weights: optional (N,) for alpha/size scaling; if provided, we normalize.
    def update_particles(self, particles_xy, weights=None):        
        P = np.asarray(particles_xy, dtype=float)
        self.particles_scatter.set_offsets(P[:, :2])

        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if len(w) == len(P) and np.isfinite(w).all():
                w = w - np.min(w)
                if np.max(w) > 0:
                    w = w / np.max(w)
                sizes = 10 + 60 * w
                self.particles_scatter.set_sizes(sizes)

    def update_estimate(self, x, y, theta):
        self.est_point.set_data([x], [y])
        L = 3.0
        x2 = x + L*np.cos(theta)
        y2 = y + L*np.sin(theta)
        self.est_heading.set_data([x, x2], [y, y2])

    def update_true(self, x, y, theta):
        self.true_point.set_data([x], [y])
        L = 3.0
        x2 = x + L*np.cos(theta)
        y2 = y + L*np.sin(theta)
        self.true_heading.set_data([x, x2], [y, y2])

######################################################################
#
#   demo
#
def _demo():
    viz = Visualization() 

    time.sleep(5)

    #Make some random particles to start near the start state.
    N = 400
    particles = np.column_stack([
        xstart + 0.75*np.random.randn(N),
        ystart + 0.75*np.random.randn(N),
    ])

    #fake "true" motion
    x, y, th = xstart, ystart, 0.0

    for k in range(600):
        #Move in a curve
        th += 0.02
        x += 0.03*np.cos(th)
        y += 0.03*np.sin(th)

        #randomly jitter particles (stand-in for prediction noise)
        particles += 0.02*np.random.randn(N, 2)

        #fake "estimate" as particle mean
        xhat, yhat = particles.mean(axis=0)

        #Update artists
        viz.update_particles(particles)
        viz.update_true(x, y, th)
        viz.update_estimate(xhat, yhat, th)

        viz.redraw()

    # Keep window alive if run as script
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    _demo()
