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
# Target location
(xtarget_start, ytarget_start) = (20, 20)

def inFreespace(x, y):
    # Boundary check
    if (x <= xmin or x >= xmax or
        y <= ymin or y >= ymax):
        return False
    point = Point(x, y)
    # disjoint = NOT intersecting any obstacle
    return obstacles.disjoint(point)

def step_target(x, y, vx, vy, dt=1.0):
    xn = x + vx * dt
    yn = y + vy * dt
    if inFreespace(xn, yn):
        return xn, yn, vx, vy
    # try flipping x only
    if inFreespace(x - vx * dt, y + vy * dt):
        vx = -vx
    # try flipping y only
    elif inFreespace(x + vx * dt, y - vy * dt):
        vy = -vy
    else:
        vx = -vx
        vy = -vy
    xn = x + vx * dt
    yn = y + vy * dt
    if not inFreespace(xn, yn):
        xn, yn = x, y
    return xn, yn, vx, vy

# Makes the target an actualy polygon that reflects lidar
def make_target_poly(xt, yt, r=0.5):
    return Polygon([
        (xt - r, yt - r), (xt + r, yt - r),
        (xt + r, yt + r), (xt - r, yt + r)
    ])

######################################################################
#
#   Visualization Class
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
        self.est_heading,  = self.ax.plot([], [], linewidth=2, color='red')   # heading line for estimate
        # True pose marker (optional usage).
        self.true_point, = self.ax.plot([], [], marker="x", markersize=6)
        self.true_heading, = self.ax.plot([], [], linewidth=2)   # heading line for true
        self.target_point, = self.ax.plot([], [], marker=".", markersize=12)

        self.est_heading.set_color('red')
        self.true_heading.set_color('green')
        self.true_body = plt.Circle((0,0), 0.5, fill=False, color='green')
        self.ax.add_patch(self.true_body)
        self.est_body = plt.Circle((0,0), 0.5, fill=False, linestyle="--", color='red')
        self.ax.add_patch(self.est_body)
        self.lidar_scatter = self.ax.scatter([], [], s=15, c='red')
        

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
        self.est_body.center = (x, y)
        L = 3.0
        x2 = x + L*np.cos(theta)
        y2 = y + L*np.sin(theta)
        self.est_heading.set_data([x, x2], [y, y2])

    def update_true(self, x, y, theta):
        self.true_point.set_data([x], [y])
        self.true_body.center = (x, y)
        L = 3.0
        x2 = x + L*np.cos(theta)
        y2 = y + L*np.sin(theta)
        self.true_heading.set_data([x, x2], [y, y2])
    
    def update_target(self, x, y):
        self.target_point.set_data([x], [y])
    
    def update_lidar_hits(self, hit_points):
        if len(hit_points) == 0:
            self.lidar_scatter.set_offsets(np.empty((0, 2)))
            return

        pts = np.array(hit_points)
        self.lidar_scatter.set_offsets(pts)

######################################################################
#
#   LIDAR implementation
#
def lidar_scan(x, y, theta, world_geom, num_rays=60, fov=np.pi, max_range=10.0):
    hits = []
    start_angle = theta - fov/2

    for i in range(num_rays):
        angle = start_angle + i * fov / (num_rays - 1)

        x_end = x + max_range*np.cos(angle)
        y_end = y + max_range*np.sin(angle)

        ray = LineString([(x, y), (x_end, y_end)])
        intersection = ray.intersection(world_geom)

        if intersection.is_empty:
            continue

        # Collect ALL candidate points
        candidate_points = []

        if intersection.geom_type == "Point":
            candidate_points = [(intersection.x, intersection.y)]

        elif intersection.geom_type == "MultiPoint":
            candidate_points = [(p.x, p.y) for p in intersection.geoms]

        elif intersection.geom_type == "GeometryCollection":
            for geom in intersection.geoms:
                if geom.geom_type == "Point":
                    candidate_points.append((geom.x, geom.y))

        elif intersection.geom_type == "LineString":
            # If overlapping a wall edge, take closest endpoint
            coords = list(intersection.coords)
            candidate_points = [coords[0], coords[-1]]

        if not candidate_points:
            continue

        # Choose closest hit
        dists = [dist((x, y), p) for p in candidate_points]
        hits.append(candidate_points[np.argmin(dists)])

    return hits

# This returns an array of the distances for each ray
def lidar_distances(x, y, theta, world_geom, num_rays=80, fov=np.pi, max_range=10):
    """Returns array of distances (max_range if no hit) for each ray."""
    start_angle = theta - fov / 2
    dists = []
    for i in range(num_rays):
        angle = start_angle + i * fov / (num_rays - 1)
        x_end = x + max_range * np.cos(angle)
        y_end = y + max_range * np.sin(angle)
        ray = LineString([(x, y), (x_end, y_end)])
        intersection = ray.intersection(world_geom)
        if intersection.is_empty:
            dists.append(max_range)
        else:
            dists.append(dist((x, y), (intersection.centroid.x, intersection.centroid.y)))
    return np.array(dists)

# Added the gaussian to the weight for each particle
def compute_weights(particles, true_dists, world_geom, sigma=0.5, num_rays=80, fov=np.pi, max_range=10):
    N = len(particles)
    weights = np.ones(N)
    for i, (px, py, pth) in enumerate(particles):
        if not inFreespace(px, py):
            weights[i] = 1e-300
            continue
        p_dists = lidar_distances(px, py, pth, world_geom, num_rays=num_rays, fov=fov, max_range=max_range)
        diff = true_dists - p_dists
        weights[i] = np.exp(-0.5 * np.sum(diff**2) / sigma**2)
    weights /= weights.sum()
    return weights
######################################################################
#
#   demo
#
def _demo():
    viz = Visualization() 

    move = {'fwd': 0.0, 'dth': 0.0}
    def on_key(event):
        if event.key == 'w':     move['fwd'] =  0.3
        elif event.key == 's':   move['fwd'] = -0.3
        elif event.key == 'a':   move['dth'] =  0.05
        elif event.key == 'd':   move['dth'] = -0.05
    viz.fig.canvas.mpl_connect('key_press_event', on_key)

    #time.sleep(5)

    #Make some random particles to start near the start state.
    N = 50
    particles = np.column_stack([
        xstart + 0.75*np.random.randn(N),
        ystart + 0.75*np.random.randn(N),
        np.random.uniform(-np.pi, np.pi, N)
    ])

    weights = np.ones(N) / N
    #fake "true" motion
    x, y, th = xstart, ystart, 0.0

    # initialize the moving target location
    xt = xtarget_start
    yt = ytarget_start
    if not inFreespace(xt, yt):
        raise ValueError("Target start location is inside obstacle!")
    vxt = 0.005
    vyt = 0.005

    while True:
        # this is for keyboard movement
        th += move['dth']
        nx = x + move['fwd'] * np.cos(th)
        ny = y + move['fwd'] * np.sin(th)
        if inFreespace(nx, ny):
            x, y = nx, ny
        move_fwd = move['fwd']
        move_dth = move['dth']
        move['fwd'] = 0.0
        move['dth'] = 0.0

        particles[:, 0] += move_fwd * np.cos(particles[:, 2]) + 0.05*np.random.randn(N)
        particles[:, 1] += move_fwd * np.sin(particles[:, 2]) + 0.05*np.random.randn(N)
        particles[:, 2] += move_dth + 0.01*np.random.randn(N)
        for i in range(N):
            if not inFreespace(particles[i, 0], particles[i, 1]):
                particles[i, 0] = x + 0.5*np.random.randn()
                particles[i, 1] = y + 0.5*np.random.randn()
        
        target_poly = make_target_poly(xt, yt)
        world_geom = obstacles.context.union(target_poly)

        hits = lidar_scan(x, y, th, world_geom, num_rays=80, fov=np.pi)
        true_dists = lidar_distances(x, y, th, world_geom)
        
        #add gaussian noise to simulate real sensor measurements
        z_real = true_dists + np.random.normal(0, 0.1, size=true_dists.shape)
        z_real = np.clip(z_real, 0, 10)

        weights = compute_weights(particles, z_real, world_geom)

        # Weighted mean for estimate
        xhat = np.average(particles[:, 0], weights=weights)
        yhat = np.average(particles[:, 1], weights=weights)
        that = np.average(particles[:, 2], weights=weights)

        # Resample
        indices = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[indices]

        xt, yt, vxt, vyt = step_target(xt, yt, vxt, vyt, dt=1.0)

        #Update artists
        viz.update_particles(particles[:, :2], weights)
        viz.update_true(x, y, th)
        viz.update_estimate(xhat, yhat, that)
        viz.update_target(xt, yt)
        viz.update_lidar_hits(hits)


        viz.redraw()

    # Keep window alive if run as script
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    _demo()
