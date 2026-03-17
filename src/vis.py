import numpy as np
import matplotlib.pyplot as plt
from config import xmin, xmax, ymin, ymax
from maps import obstacles

class Visualization:
    """
    Matplotlib-based live visualizer.

    Draws:
      • obstacle outlines
      • particle cloud (scatter + heading arrows)
      • true robot pose  (green circle + heading line)
      • estimated pose   (red dashed circle + heading line)
      • LiDAR hit points (small red dots)
      • current waypoint target (gold star)
    """

    def __init__(self, title=" Visualization"):
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self.ax.grid(True)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")

        # Draw obstacle boundaries once
        for poly in obstacles.context.geoms:
            self.ax.plot(*poly.exterior.xy, 'k-', linewidth=2)

        # Particle positions (scatter) and headings (quiver arrows)
        self.particles_scatter = self.ax.scatter([], [], s=10, alpha=0.35)
        self.particles_quiver  = self.ax.quiver(
            np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
            angles='xy', scale_units='xy', scale=1, width=0.015,
            headwidth=2, headlength=2, color='steelblue', alpha=0.45, zorder=3
        )

        # Pose markers and heading lines
        self.est_point,    = self.ax.plot([], [], marker="o", markersize=6)
        self.est_heading,  = self.ax.plot([], [], linewidth=2, color='red')
        self.true_point,   = self.ax.plot([], [], marker="x", markersize=6)
        self.true_heading, = self.ax.plot([], [], linewidth=2, color='green')

        # Body circles (radius = 0.5 m, representative robot footprint)
        self.true_body = plt.Circle((0, 0), 0.5, fill=False, color='green')
        self.ax.add_patch(self.true_body)
        self.est_body  = plt.Circle((0, 0), 0.5, fill=False, linestyle="--", color='red')
        self.ax.add_patch(self.est_body)

        # LiDAR hit-point overlay
        self.lidar_scatter = self.ax.scatter([], [], s=15, c='red')

    def redraw(self):
        """Flush pending draw calls and pause briefly to allow GUI events."""
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    # Update helper functions

    def update_particles(self, particles, weights=None):
        """
        Refresh the particle cloud.

        Particles with higher weights are drawn larger so the user can
        see which hypotheses the filter currently favours.
        """
        P  = np.asarray(particles, dtype=float)
        xy = P[:, :2]
        th = P[:,  2]
        self.particles_scatter.set_offsets(xy)

        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if len(w) == len(P) and np.isfinite(w).all():
                # Normalise weights to [0, 1] for display purposes only
                w = w - np.min(w)
                if np.max(w) > 0:
                    w = w / np.max(w)
                sizes = 10 + 60 * w   # map to marker sizes [10, 70]
                self.particles_scatter.set_sizes(sizes)

        # Quiver must be recreated each frame (no in-place update API)
        self.particles_quiver.remove()
        self.particles_quiver = self.ax.quiver(
            xy[:, 0], xy[:, 1],
            0.4 * np.cos(th), 0.4 * np.sin(th),
            angles='xy', scale_units='xy', scale=1, width=0.015,
            headwidth=2, headlength=2, color='steelblue', alpha=0.45, zorder=3
        )

    def update_estimate(self, x, y, theta):
        """Move the red estimated-pose marker to (x, y, theta)."""
        self.est_point.set_data([x], [y])
        self.est_body.center = (x, y)
        L = 3.0   # heading line length in world units
        self.est_heading.set_data([x, x + L*np.cos(theta)],
                                  [y, y + L*np.sin(theta)])

    def update_true(self, x, y, theta):
        """Move the green true-pose marker to (x, y, theta)."""
        self.true_point.set_data([x], [y])
        self.true_body.center = (x, y)
        L = 3.0
        self.true_heading.set_data([x, x + L*np.cos(theta)],
                                   [y, y + L*np.sin(theta)])

    def update_lidar_hits(self, hit_points):
        """Overlay the (x, y) world coordinates where LiDAR rays terminated."""
        if len(hit_points) == 0:
            self.lidar_scatter.set_offsets(np.empty((0, 2)))
            return
        self.lidar_scatter.set_offsets(np.array(hit_points))

