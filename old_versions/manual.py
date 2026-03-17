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


# CHOOSE MAP!!!! (uncomment the one you want to choose)

#varied obstacles
# obstacles = prep(MultiPolygon([
#     Polygon([[2, 14], [5, 14], [5, 23], [2, 23]]),
#     Polygon([[7, 23], [11, 23], [11, 20], [7, 20]]),
#     Polygon([[7, 17], [11, 17], [11, 14], [7, 14]]),
#     Polygon([[14, 23], [19, 23], [19, 20], [16, 20], [16, 14], [14, 14]]),
#     Polygon([[21, 23], [23, 23], [23, 14], [18, 14], [18, 17], [21, 17]]),
#     Polygon([[2, 11], [8, 11], [8, 8], [6, 8], [6, 2], [4, 2], [4, 8], [2, 8]]),
#     Polygon([[10, 11], [15, 11], [15, 8], [12, 8], [12, 5], [15, 5], [15, 2], [10, 2]]),
#     Polygon([[17, 11], [23, 11], [23, 8], [21, 8], [21, 5], [20, 5], [20, 2], [17, 2]])
# ]))

#long hallways
obstacles = prep(MultiPolygon([
    Polygon([[2, 23], [23, 23], [23, 20], [2, 20]]),
    Polygon([[2, 17], [23, 17], [23, 14], [2, 14]]),
    Polygon([[2, 11], [23, 11], [23, 8], [2, 8]]),
    Polygon([[2, 5], [23, 5], [23, 2], [2, 2]])
]))

#circle
# center_coords = (12.5, 12.5)
# radius = 10.0
# center_point = Point(center_coords)
# obstacles = prep(MultiPolygon([center_point.buffer(radius)]))

#hexagon
obstacles = prep(MultiPolygon([
     Polygon([[7, 4], [18, 4], [22, 12], [18, 21], [7, 21], [3, 12]])
]))


# Start
(xstart, ystart) = (1, 1)

def inFreespace(x, y):
    if (x <= xmin or x >= xmax or
        y <= ymin or y >= ymax):
        return False
    point = Point(x, y)
    return obstacles.disjoint(point)

# def step_target(x, y, vx, vy, dt=1.0):
#     xn = x + vx * dt
#     yn = y + vy * dt
#     if inFreespace(xn, yn):
#         return xn, yn, vx, vy
#     if inFreespace(x - vx * dt, y + vy * dt):
#         vx = -vx
#     elif inFreespace(x + vx * dt, y - vy * dt):
#         vy = -vy
#     else:
#         vx = -vx
#         vy = -vy
#     xn = x + vx * dt
#     yn = y + vy * dt
#     if not inFreespace(xn, yn):
#         xn, yn = x, y
#     return xn, yn, vx, vy

# def make_target_poly(xt, yt, r=0.5):
#     return Polygon([
#         (xt - r, yt - r), (xt + r, yt - r),
#         (xt + r, yt + r), (xt - r, yt + r)
#     ])

######################################################################
#
#   Visualization Class
#
class Visualization:
    def __init__(self, title=" Visualization"):
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self.ax.grid(True)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")

        for poly in obstacles.context.geoms:
            self.ax.plot(*poly.exterior.xy, 'k-', linewidth=2)

        self.particles_scatter = self.ax.scatter([], [], s=10, alpha=0.35)
        self.particles_quiver = self.ax.quiver(
            np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
            angles='xy', scale_units='xy', scale=1, width=0.015,
            headwidth=2, headlength=2, color='steelblue', alpha=0.45, zorder=3
        )

        self.est_point, = self.ax.plot([], [], marker="o", markersize=6)
        self.est_heading,  = self.ax.plot([], [], linewidth=2, color='red')
        self.true_point, = self.ax.plot([], [], marker="x", markersize=6)
        self.true_heading, = self.ax.plot([], [], linewidth=2)
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

    def update_particles(self, particles, weights=None):
        P = np.asarray(particles, dtype=float)
        xy = P[:, :2]
        th = P[:, 2]
        self.particles_scatter.set_offsets(xy)

        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if len(w) == len(P) and np.isfinite(w).all():
                w = w - np.min(w)
                if np.max(w) > 0:
                    w = w / np.max(w)
                sizes = 10 + 60 * w
                self.particles_scatter.set_sizes(sizes)

        self.particles_quiver.remove()
        self.particles_quiver = self.ax.quiver(
            xy[:, 0], xy[:, 1],
            0.4 * np.cos(th), 0.4 * np.sin(th),
            angles='xy', scale_units='xy', scale=1, width=0.015,
            headwidth=2, headlength=2, color='steelblue', alpha=0.45, zorder=3
        )

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

    # def update_target(self, x, y):
    #     self.target_point.set_data([x], [y])

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

NUM_RAYS  = 60
FOV       = 2 * pi
MAX_RANGE = 36.0
SIGMA     = 0.4

def build_segment_list(world_geom):
    segs = []
    for poly in world_geom.geoms:
        coords = list(poly.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            segs.append([a[0], a[1], b[0], b[1]])
    segs.append([xmin, ymin, xmax, ymin])
    segs.append([xmax, ymin, xmax, ymax])
    segs.append([xmax, ymax, xmin, ymax])
    segs.append([xmin, ymax, xmin, ymin])
    return np.array(segs)

def lidar_scan(x, y, theta, segs):
    hits = []
    for i in range(NUM_RAYS):
        angle = theta - FOV/2 + i * FOV / (NUM_RAYS - 1)
        dx = np.cos(angle)
        dy = np.sin(angle)
        best_t = MAX_RANGE
        for seg in segs:
            x1, y1, x2, y2 = seg
            sx, sy = x2 - x1, y2 - y1
            denom = dx * sy - dy * sx
            if abs(denom) < 1e-10:
                continue
            ao_x, ao_y = x1 - x, y1 - y
            t = (ao_x * sy - ao_y * sx) / denom
            u = (ao_x * dy - ao_y * dx) / denom
            if 0 <= t <= MAX_RANGE and 0 <= u <= 1:
                best_t = min(best_t, t)
        if best_t < MAX_RANGE:
            hits.append((x + best_t * dx, y + best_t * dy))
    return hits

def lidar_distances(x, y, theta, segs):
    dists = []
    for i in range(NUM_RAYS):
        angle = theta - FOV/2 + i * FOV / (NUM_RAYS - 1)
        dx = np.cos(angle)
        dy = np.sin(angle)
        best_t = MAX_RANGE
        for seg in segs:
            x1, y1, x2, y2 = seg
            sx, sy = x2 - x1, y2 - y1
            denom = dx * sy - dy * sx
            if abs(denom) < 1e-10:
                continue
            ao_x, ao_y = x1 - x, y1 - y
            t = (ao_x * sy - ao_y * sx) / denom
            u = (ao_x * dy - ao_y * dx) / denom
            if 0 <= t <= MAX_RANGE and 0 <= u <= 1:
                best_t = min(best_t, t)
        dists.append(best_t)
    return np.array(dists)

def all_particle_lidar_distances(particles, segs):
    xs = particles[:, 0]
    ys = particles[:, 1]
    thetas = particles[:, 2]

    angles = thetas[:, None] - FOV/2 + np.arange(NUM_RAYS) * FOV / (NUM_RAYS - 1)
    dx = np.cos(angles)[:, :, None]
    dy = np.sin(angles)[:, :, None]
    ox = xs[:, None, None]
    oy = ys[:, None, None]

    x1 = segs[None, None, :, 0];  y1 = segs[None, None, :, 1]
    x2 = segs[None, None, :, 2];  y2 = segs[None, None, :, 3]
    sx = x2 - x1;  sy = y2 - y1

    denom = dx * sy - dy * sx
    ao_x = x1 - ox
    ao_y = y1 - oy

    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(np.abs(denom) > 1e-10, (ao_x * sy - ao_y * sx) / denom, np.inf)
        u = np.where(np.abs(denom) > 1e-10, (ao_x * dy - ao_y * dx) / denom, np.inf)

    valid = (t >= 0) & (t <= MAX_RANGE) & (u >= 0) & (u <= 1)
    t[~valid] = np.inf
    dists = np.min(t, axis=2)
    dists[np.isinf(dists)] = MAX_RANGE
    return dists

def compute_weights(particles, z_real, segs):
    in_free = np.array([inFreespace(p[0], p[1]) for p in particles])
    all_dists = all_particle_lidar_distances(particles, segs)
    diff = z_real[None, :] - all_dists
    log_weights = -0.5 * np.sum(diff**2, axis=1) / SIGMA**2
    log_weights[~in_free] = -1e9
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total = weights.sum()
    if total <= 1e-300 or not np.isfinite(total):
        weights = np.ones(len(particles))/len(particles)
    else:
        weights /= total
    return weights

######################################################################
#
#   demo
#
def random_free_particle():
    while True:
        px = np.random.uniform(xmin + 0.5, xmax - 0.5)
        py = np.random.uniform(ymin + 0.5, ymax - 0.5)
        if inFreespace(px, py):
            return [px, py, np.random.uniform(-pi, pi)]

def resample(particles, weights, N, inject_frac=0.05):
    n_keep = N - int(N * inject_frac)
    cumsum = np.cumsum(weights)
    step = 1.0 / n_keep
    start = np.random.uniform(0, step)
    pos = start + step * np.arange(n_keep)
    indices = np.clip(np.searchsorted(cumsum, pos), 0, N - 1)
    kept = particles[indices].copy()
    injected = np.array([random_free_particle() for _ in range(N - n_keep)])
    return np.vstack([kept, injected])

def motion_update(particles, fwd, dth, alpha=(0.10, 0.04)):
    N = len(particles)
    fwd_noise = alpha[0] * abs(fwd)
    th_noise = alpha[1] * abs(dth) + 0.008 * abs(fwd)
    particles[:, 2] += dth + th_noise * np.random.randn(N)
    noisy_fwd = fwd + fwd_noise * np.random.randn(N)
    particles[:, 0] += noisy_fwd * np.cos(particles[:, 2])
    particles[:, 1] += noisy_fwd * np.sin(particles[:, 2])
    return particles

def _demo():
    viz = Visualization()

    move = {'fwd': 0.0, 'dth': 0.0}
    def on_key(event):
        if event.key == 'w':     move['fwd'] =  0.3
        elif event.key == 's':   move['fwd'] = -0.3
        elif event.key == 'a':   move['dth'] =  0.05
        elif event.key == 'd':   move['dth'] = -0.05
    viz.fig.canvas.mpl_connect('key_press_event', on_key)

    N = 500
    particles = np.array([random_free_particle() for _ in range(N)])
    weights = np.ones(N) / N

    x, y, th = float(xstart), float(ystart), 0.0
    world_geom = obstacles.context
    segs = build_segment_list(world_geom)

    while True:
        th += move['dth']
        nx = x + move['fwd'] * np.cos(th)
        ny = y + move['fwd'] * np.sin(th)
        if inFreespace(nx, ny):
            x, y = nx, ny
        move_fwd = move['fwd']
        move_dth = move['dth']
        move['fwd'] = 0.0
        move['dth'] = 0.0

        particles = motion_update(particles, move_fwd, move_dth)

        for i in range(N):
            if not inFreespace(particles[i, 0], particles[i, 1]):
                for _ in range(5):
                    cx = particles[i, 0] + 0.4 * np.random.randn()
                    cy = particles[i, 1] + 0.4 * np.random.randn()
                    if inFreespace(cx, cy):
                        particles[i, 0] = cx
                        particles[i, 1] = cy
                        break
                else:
                    particles[i] = random_free_particle()

        hits = lidar_scan(x, y, th, segs)
        true_dists = lidar_distances(x, y, th, segs)

        z_real = true_dists + np.random.randn(NUM_RAYS) * SIGMA
        z_real = np.clip(z_real, 0.0, MAX_RANGE)

        weights = compute_weights(particles, z_real, segs)

        N_eff = 1.0 / max(np.sum(weights**2), 1e-300)
        xhat = np.average(particles[:, 0], weights=weights)
        yhat = np.average(particles[:, 1], weights=weights)
        that = np.arctan2(
            np.average(np.sin(particles[:, 2]), weights=weights),
            np.average(np.cos(particles[:, 2]), weights=weights))
        
        if N_eff < N * 0.5:
            particles = resample(particles, weights, N, inject_frac=0.05)

            particles[:, 0] += 0.03 * np.random.randn(N)
            particles[:, 1] += 0.03 * np.random.randn(N)
            particles[:, 2] += 0.01 * np.random.randn(N)
            weights = np.ones(N) / N

        viz.update_particles(particles, weights)
        viz.update_true(x, y, th)
        viz.update_estimate(xhat, yhat, that)
        viz.update_lidar_hits(hits)
        viz.redraw()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    _demo()