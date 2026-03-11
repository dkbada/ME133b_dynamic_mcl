import matplotlib.pyplot as plt
import numpy as np
import heapq
import time

from math             import dist
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.prepared import prep

######################################################################
#
#   World Definitions
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

(xstart, ystart)               = (6,  1)
(xgoal,  ygoal)                = (20, 20)   # ← set your desired goal here
(xtarget_start, ytarget_start) = (20, 20)

# --- Motion noise ---
ALPHA_FWD   = 0.05
ALPHA_ROT   = 0.01

# --- Sensor noise ---
LIDAR_SIGMA = 0.1

# --- Filter parameters ---
N            = 300
NUM_RAYS     = 80
FOV          = np.pi
MAX_RANGE    = 10.0
SENSOR_SIGMA = 0.3

# --- Resampling / recovery ---
ESS_THRESH       = N / 2
INJECT_FRAC      = 0.05
INJECT_THRESHOLD = 3.0 / N

# --- Motion gate ---
MOTION_THRESHOLD = 0.02
ANGLE_THRESHOLD  = 0.01

# --- Path follower ---
WAYPOINT_RADIUS  = 0.4   # distance at which we consider a waypoint reached
ROBOT_SPEED      = 1.0  # forward step per loop iteration (world units)
ROBOT_ROT_SPEED  = 0.3  # max heading correction per step (radians)


######################################################################
#
#   World helpers
#
def inFreespace(x, y):
    if x <= xmin or x >= xmax or y <= ymin or y >= ymax:
        return False
    return obstacles.disjoint(Point(x, y))

def step_target(x, y, vx, vy, dt=1.0):
    xn, yn = x + vx*dt, y + vy*dt
    if inFreespace(xn, yn):
        return xn, yn, vx, vy
    if   inFreespace(x - vx*dt, y + vy*dt): vx = -vx
    elif inFreespace(x + vx*dt, y - vy*dt): vy = -vy
    else:                                    vx, vy = -vx, -vy
    xn, yn = x + vx*dt, y + vy*dt
    if not inFreespace(xn, yn):
        xn, yn = x, y
    return xn, yn, vx, vy

def make_target_poly(xt, yt, r=0.5):
    return Polygon([(xt-r, yt-r), (xt+r, yt-r), (xt+r, yt+r), (xt-r, yt+r)])

def random_freespace_particle():
    while True:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        if inFreespace(x, y):
            return np.array([x, y, np.random.uniform(-np.pi, np.pi)])


######################################################################
#
#   A* path planner
#
#   We rasterise the world onto a fine grid, inflate obstacles by the
#   robot radius so the centre-line path is guaranteed collision-free,
#   then run A* and convert grid waypoints back to world coordinates.
#   The resulting path is then smoothed with a simple string-pull so
#   the robot doesn't hug every grid corner.
#

GRID_RES    = 0.5    # world units per grid cell
ROBOT_RADIUS = 0.6   # inflation radius (> robot body radius of 0.5)

def _world_to_grid(x, y, res=GRID_RES):
    return int((x - xmin) / res), int((y - ymin) / res)

def _grid_to_world(gx, gy, res=GRID_RES):
    return xmin + (gx + 0.5) * res, ymin + (gy + 0.5) * res

def build_grid(res=GRID_RES, inflation=ROBOT_RADIUS):
    """
    Build a 2-D occupancy grid.
    Cells whose centres are within `inflation` of any obstacle are marked
    occupied (True).  All others are free (False).
    """
    cols = int((xmax - xmin) / res)
    rows = int((ymax - ymin) / res)
    grid = np.zeros((cols, rows), dtype=bool)

    # Number of cells to check around each obstacle point
    pad = int(np.ceil(inflation / res)) + 1

    for gx in range(cols):
        for gy in range(rows):
            wx, wy = _grid_to_world(gx, gy, res)
            pt = Point(wx, wy)
            # Mark as occupied if closer than inflation to any obstacle
            if not obstacles.disjoint(pt.buffer(inflation)):
                grid[gx, gy] = True

    return grid, cols, rows

def astar(grid, cols, rows, start_g, goal_g):
    """
    A* on the occupancy grid.
    Returns list of grid cells from start to goal, or [] if no path.
    Uses 8-connected neighbours with diagonal cost sqrt(2).
    """
    sx, sy = start_g
    gx, gy = goal_g

    # Snap any endpoint that landed in an occupied cell to the nearest
    # free neighbour.  This handles cases where the exact world coordinate
    # maps to an inflated cell (e.g. a goal right next to a wall).
    def _snap(cx, cy):
        if not grid[cx, cy]:
            return cx, cy
        for radius in range(1, max(cols, rows)):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue   # only check the shell at this radius
                    nx2, ny2 = cx+dx, cy+dy
                    if 0 <= nx2 < cols and 0 <= ny2 < rows and not grid[nx2, ny2]:
                        return nx2, ny2
        raise ValueError(f"No free cell found near ({cx},{cy})")
    sx, sy = _snap(sx, sy)
    gx, gy = _snap(gx, gy)

    # Priority queue entries: (f, g_cost, (x, y))
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, (sx, sy)))

    came_from = {}
    g_cost    = {(sx, sy): 0.0}

    neighbours_8 = [( 1, 0,1.0),(-1, 0,1.0),( 0, 1,1.0),( 0,-1,1.0),
                    ( 1, 1,1.414),( 1,-1,1.414),(-1, 1,1.414),(-1,-1,1.414)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == (gx, gy):
            # Reconstruct
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((sx, sy))
            path.reverse()
            return path

        if g > g_cost.get(current, float('inf')):
            continue   # stale entry

        cx, cy = current
        for dx, dy, step_cost in neighbours_8:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if grid[nx, ny]:
                continue
            new_g = g_cost[current] + step_cost
            nb    = (nx, ny)
            if new_g < g_cost.get(nb, float('inf')):
                g_cost[nb]   = new_g
                came_from[nb] = current
                h = abs(nx - gx) + abs(ny - gy)   # Manhattan heuristic
                heapq.heappush(open_heap, (new_g + h, new_g, nb))

    return []   # no path found

def string_pull(path_world, world_geom_inflated):
    """
    Greedy visibility string-pull (also called 'funnel lite').
    Starting from the first waypoint, skip ahead as far as possible
    while there is a clear line-of-sight in the inflated world.
    Returns a shorter list of waypoints.
    """
    if len(path_world) <= 2:
        return path_world

    pruned = [path_world[0]]
    i = 0
    while i < len(path_world) - 1:
        # Find the furthest waypoint visible from path_world[i]
        j = len(path_world) - 1
        while j > i + 1:
            seg = LineString([path_world[i], path_world[j]])
            if seg.disjoint(world_geom_inflated):
                break
            j -= 1
        pruned.append(path_world[j])
        i = j
    return pruned

def plan_path(start_xy, goal_xy):
    """
    Full pipeline: build grid → A* → convert to world → string-pull.
    Returns list of (x, y) waypoints including start and goal.
    """
    grid, cols, rows = build_grid()

    sg = _world_to_grid(*start_xy)
    gg = _world_to_grid(*goal_xy)

    grid_path = astar(grid, cols, rows, sg, gg)
    if not grid_path:
        raise RuntimeError(f"A* found no path from {start_xy} to {goal_xy}")

    world_path = [_grid_to_world(gx, gy) for gx, gy in grid_path]

    # Build a slightly inflated obstacle geometry for the string-pull
    # visibility check (uses the raw MultiPolygon, not the prep'd version)
    inflated_geom = obstacles.context.buffer(ROBOT_RADIUS)
    world_path    = string_pull(world_path, inflated_geom)

    # Always use the exact start/goal coordinates (not snapped grid centres)
    world_path[0]  = start_xy
    world_path[-1] = goal_xy

    return world_path


######################################################################
#
#   Pure-pursuit path follower
#
#   Given the robot's current pose and the remaining waypoint list,
#   returns (fwd, dth) control inputs for one step.
#
#   Pure pursuit: always steer toward the next waypoint. When within
#   WAYPOINT_RADIUS of the current waypoint, advance to the next one.
#

def follow_path(x, y, th, waypoints):
    """
    Returns (fwd, dth, waypoints) — updated waypoint list with reached
    waypoints popped off the front.
    """
    if not waypoints:
        return 0.0, 0.0, waypoints

    # Pop any waypoints we've already reached
    while waypoints:
        wx, wy = waypoints[0]
        if dist((x, y), (wx, wy)) < WAYPOINT_RADIUS:
            waypoints = waypoints[1:]
        else:
            break

    if not waypoints:
        return 0.0, 0.0, waypoints   # goal reached

    wx, wy  = waypoints[0]
    desired = np.arctan2(wy - y, wx - x)

    # Heading error wrapped to [-π, π]
    dth = desired - th
    dth = (dth + np.pi) % (2 * np.pi) - np.pi

    # Clamp rotation to max turn rate
    dth = np.clip(dth, -ROBOT_ROT_SPEED, ROBOT_ROT_SPEED)

    # Move forward only when roughly facing the waypoint;
    # slow down proportionally when misaligned so the robot turns first.
    alignment = np.cos(dth / ROBOT_ROT_SPEED * (np.pi / 2))
    fwd       = ROBOT_SPEED * max(0.0, alignment)

    return fwd, dth, waypoints


######################################################################
#
#   Visualization
#
class Visualization:
    def __init__(self, title="MCL + A*"):
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
        self.est_point,        = self.ax.plot([], [], 'ro', markersize=6)
        self.est_heading,      = self.ax.plot([], [], 'r-', linewidth=2)
        self.true_point,       = self.ax.plot([], [], 'gx', markersize=8)
        self.true_heading,     = self.ax.plot([], [], 'g-', linewidth=2)
        self.target_point,     = self.ax.plot([], [], 'b.', markersize=12)
        self.lidar_scatter     = self.ax.scatter([], [], s=15, c='red')
        # Planned path line
        self.path_line,        = self.ax.plot([], [], 'c--', linewidth=1,
                                              alpha=0.6, label='planned path')
        # Goal marker
        self.goal_point,       = self.ax.plot([], [], 'y*', markersize=14,
                                              label='goal')

        self.true_body = plt.Circle((0,0), 0.5, fill=False, color='green')
        self.est_body  = plt.Circle((0,0), 0.5, fill=False,
                                    linestyle='--', color='red')
        self.ax.add_patch(self.true_body)
        self.ax.add_patch(self.est_body)
        self.ax.legend(loc='upper left', fontsize=7)

    def redraw(self):
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def update_path(self, waypoints):
        if waypoints:
            xs, ys = zip(*waypoints)
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

    def update_goal(self, x, y):
        self.goal_point.set_data([x], [y])

    def update_particles(self, particles, weights=None):
        P = np.asarray(particles, dtype=float)
        self.particles_scatter.set_offsets(P[:, :2])
        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            if len(w) == len(P) and np.isfinite(w).all():
                w = w - w.min()
                if w.max() > 0: w /= w.max()
                self.particles_scatter.set_sizes(10 + 60*w)

    def update_estimate(self, x, y, th):
        self.est_point.set_data([x], [y])
        self.est_body.center = (x, y)
        self.est_heading.set_data([x, x + 3*np.cos(th)],
                                  [y, y + 3*np.sin(th)])

    def update_true(self, x, y, th):
        self.true_point.set_data([x], [y])
        self.true_body.center = (x, y)
        self.true_heading.set_data([x, x + 3*np.cos(th)],
                                   [y, y + 3*np.sin(th)])

    def update_target(self, x, y):
        self.target_point.set_data([x], [y])

    def update_lidar_hits(self, hits):
        if not hits:
            self.lidar_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.lidar_scatter.set_offsets(np.array(hits))


######################################################################
#
#   LiDAR
#
def _cast_ray(x, y, angle, world_geom, max_range):
    x_end = x + max_range * np.cos(angle)
    y_end = y + max_range * np.sin(angle)
    ray   = LineString([(x, y), (x_end, y_end)])
    inter = ray.intersection(world_geom)
    if inter.is_empty:
        return None, max_range
    candidates = []
    gt = inter.geom_type
    if   gt == "Point":             candidates = [(inter.x, inter.y)]
    elif gt == "MultiPoint":        candidates = [(p.x, p.y) for p in inter.geoms]
    elif gt == "LineString":        candidates = list(inter.coords)[:2]
    elif gt == "GeometryCollection":
        candidates = [(g.x, g.y) for g in inter.geoms if g.geom_type == "Point"]
    if not candidates:
        return None, max_range
    closest = min(candidates, key=lambda p: dist((x, y), p))
    return closest, dist((x, y), closest)

def lidar_scan(x, y, theta, world_geom,
               num_rays=NUM_RAYS, fov=FOV, max_range=MAX_RANGE):
    hits        = []
    start_angle = theta - fov / 2
    for i in range(num_rays):
        angle = start_angle + i * fov / (num_rays - 1)
        pt, _ = _cast_ray(x, y, angle, world_geom, max_range)
        if pt is not None:
            hits.append(pt)
    return hits

def lidar_distances(x, y, theta, world_geom,
                    num_rays=NUM_RAYS, fov=FOV, max_range=MAX_RANGE):
    start_angle = theta - fov / 2
    ranges      = np.empty(num_rays)
    for i in range(num_rays):
        angle     = start_angle + i * fov / (num_rays - 1)
        _, r      = _cast_ray(x, y, angle, world_geom, max_range)
        ranges[i] = r
    return ranges


######################################################################
#
#   MCL — sample / weight / resample
#
def sample_motion_model(particles, fwd, dth,
                        alpha_fwd=ALPHA_FWD, alpha_rot=ALPHA_ROT):
    N   = len(particles)
    out = particles.copy()
    fwd_noise = np.random.normal(fwd, alpha_fwd * (abs(fwd) + 1e-6), N)
    dth_noise = np.random.normal(dth, alpha_rot * (abs(dth) + 1e-6)
                                      + alpha_fwd * abs(fwd), N)
    out[:, 0] += fwd_noise * np.cos(out[:, 2])
    out[:, 1] += fwd_noise * np.sin(out[:, 2])
    out[:, 2] += dth_noise
    out[:, 2]  = (out[:, 2] + np.pi) % (2*np.pi) - np.pi
    return out

def compute_weights(particles, z_measured, world_geom,
                    sigma=SENSOR_SIGMA, num_rays=NUM_RAYS,
                    fov=FOV, max_range=MAX_RANGE):
    N       = len(particles)
    weights = np.empty(N)
    for i, (px, py, pth) in enumerate(particles):
        if not inFreespace(px, py):
            weights[i] = 1e-12
            continue
        p_dists    = lidar_distances(px, py, pth, world_geom,
                                     num_rays=num_rays, fov=fov,
                                     max_range=max_range)
        diff       = z_measured - p_dists
        log_w      = -0.5 * np.sum(diff**2) / sigma**2
        weights[i] = np.exp(np.clip(log_w, -700, 0))
    weights += 1e-12
    weights /= weights.sum()
    return weights

def systematic_resample(particles, weights):
    N   = len(particles)
    r   = np.random.uniform(0, 1.0/N)
    c   = weights[0]
    i   = 0
    out = np.empty_like(particles)
    for m in range(N):
        U = r + m / N
        while U > c:
            i += 1
            c += weights[i]
        out[m] = particles[i]
    return out


######################################################################
#
#   Main loop
#
def _demo():
    viz = Visualization()

    # ── Plan path ──────────────────────────────────────────────────────
    print("Planning path with A*...")
    waypoints = plan_path((xstart, ystart), (xgoal, ygoal))
    print(f"  Path found: {len(waypoints)} waypoints after string-pull.")
    viz.update_path(waypoints)
    viz.update_goal(xgoal, ygoal)

    # ── Initialise particles ───────────────────────────────────────────
    particles = np.column_stack([
        xstart + 0.75*np.random.randn(N),
        ystart + 0.75*np.random.randn(N),
        np.random.uniform(-np.pi, np.pi, N)
    ])
    weights = np.ones(N) / N

    x, y, th = float(xstart), float(ystart), 0.0

    xt, yt = float(xtarget_start), float(ytarget_start)
    if not inFreespace(xt, yt):
        raise ValueError("Target start is inside an obstacle.")
    vxt, vyt = 0.005, 0.005

    N_INJECT = max(1, int(N * INJECT_FRAC))
    N_KEEP   = N - N_INJECT

    goal_reached = False

    while True:

        # ── Path follower — compute control inputs ─────────────────────
        # The follower steers the TRUE robot toward the next waypoint.
        # MCL localises using only noisy odometry + lidar, so the
        # estimated pose may lag or differ from the true pose — this is
        # exactly what we want to observe.
        if not goal_reached:
            fwd, dth, waypoints = follow_path(x, y, th, waypoints)
            if not waypoints:
                goal_reached = True
                fwd, dth = 0.0, 0.0
                print("Goal reached!")
        else:
            fwd, dth = 0.0, 0.0

        # ── True robot motion ──────────────────────────────────────────
        th += dth
        nx  = x + fwd * np.cos(th)
        ny  = y + fwd * np.sin(th)
        if inFreespace(nx, ny):
            x, y = nx, ny

        # ── STEP 1: SAMPLE ─────────────────────────────────────────────
        particles = sample_motion_model(particles, fwd, dth)

        for i in range(N):
            if not inFreespace(particles[i, 0], particles[i, 1]):
                particles[i, 0] = x + 0.5*np.random.randn()
                particles[i, 1] = y + 0.5*np.random.randn()

        # ── Build world geometry ───────────────────────────────────────
        target_poly = make_target_poly(xt, yt)
        world_geom  = obstacles.context.union(target_poly)

        # ── STEP 2: WEIGHT ─────────────────────────────────────────────
        moved = (abs(fwd) > MOTION_THRESHOLD or abs(dth) > ANGLE_THRESHOLD)

        if moved:
            true_dists = lidar_distances(x, y, th, world_geom)
            z_measured = true_dists + np.random.normal(0, LIDAR_SIGMA,
                                                        size=true_dists.shape)
            z_measured = np.clip(z_measured, 0, MAX_RANGE)
            weights    = compute_weights(particles, z_measured, world_geom)

        # ── STEP 3: RESAMPLE ───────────────────────────────────────────
        Neff = 1.0 / np.sum(weights**2)

        if Neff < ESS_THRESH:
            if weights.max() < INJECT_THRESHOLD:
                resampled = systematic_resample(particles, weights)[:N_KEEP]
                injected  = np.array([random_freespace_particle()
                                      for _ in range(N_INJECT)])
                particles = np.vstack([resampled, injected])
            else:
                particles = systematic_resample(particles, weights)
            weights = np.ones(N) / N

        # ── Pose estimate ──────────────────────────────────────────────
        xhat  = float(np.sum(particles[:, 0] * weights))
        yhat  = float(np.sum(particles[:, 1] * weights))
        thhat = float(np.arctan2(
            np.sum(np.sin(particles[:, 2]) * weights),
            np.sum(np.cos(particles[:, 2]) * weights)
        ))

        # ── Advance target ─────────────────────────────────────────────
        xt, yt, vxt, vyt = step_target(xt, yt, vxt, vyt, dt=1.0)

        # ── Draw ───────────────────────────────────────────────────────
        hits  = lidar_scan(x, y, th, world_geom)
        error = np.sqrt((x - xhat)**2 + (y - yhat)**2)

        viz.update_path(waypoints)   # shrinks as waypoints are consumed
        viz.update_particles(particles[:, :2], weights)
        viz.update_true(x, y, th)
        viz.update_estimate(xhat, yhat, thhat)
        viz.update_target(xt, yt)
        viz.update_lidar_hits(hits)

        status = "GOAL REACHED" if goal_reached else f"wp left: {len(waypoints)}"
        viz.ax.set_title(
            f"Error: {error:.2f}  |  ESS: {Neff:.0f}/{N}  |  {status}"
        )
        viz.redraw()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    _demo()