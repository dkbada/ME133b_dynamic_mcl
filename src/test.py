import matplotlib.pyplot as plt
import numpy as np
import heapq

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

(xstart, ystart) = (6, 1)

######################################################################
#
#   Configuration — edit these
#
# --- Moving obstacles ---
# Set to 0 to disable entirely.  Each obstacle is a small square of
# half-width OBS_RADIUS that bounces around freespace.
NUM_MOVING_OBS = 0      # number of moving obstacles  ← change me
OBS_RADIUS     = 0.4    # half-width of each moving obstacle (world units)
OBS_SPEED      = 0.08   # world units per step

# --- Sweep waypoints ---
# The robot visits these in order, looping back to the first one forever.
# They are chosen to give good map coverage without crossing any wall.
# You can add/remove/reorder them freely; the planner will connect them.
# Interior-first sweep: starts in the middle of the map, weaves through
# every interior corridor slot and upper room before briefly touching the
# outer perimeter.  This matters for MCL: interior geometry (obstacle
# faces, narrow gaps) is more distinctive than open outer edges, so the
# filter converges faster when the robot sees it first.
SWEEP_WAYPOINTS = [
    # ── lower interior: weave through the F-G and G-H obstacle slots ──
    ( 6,  1),   # start
    ( 9,  3),   # enter F-G gap (bottom)
    ( 9,  9),   # F-G gap (top) — traverses the full slot height
    (16,  9),   # G-H gap (top)
    (16,  3),   # G-H gap (bottom) — back down for full slot coverage
    ( 9,  6),   # F-G mid zigzag — extra coverage pass
    ( 1,  6),   # left of obstacle F — leftmost lower pocket

    # ── middle corridor: full-width sweep at y≈12 ─────────────────────
    ( 3, 12),   # mid-left
    ( 9, 12),   # mid-centre-left
    (12, 12),   # centre junction (most connected point in the map)
    (16, 12),   # mid-centre-right
    (20, 12),   # mid-right

    # ── upper interior: weave through every upper room ─────────────────
    (20, 18),   # D-E gap (right side upper)
    (20, 21),   # top-right open pocket
    (12, 21),   # B-D gap top
    (12, 15),   # C-D gap (centre upper)
    ( 9, 18),   # B-C gap (between obstacles B and C)
    ( 6, 21),   # A-B gap
    ( 6, 16),   # A-C gap
    ( 1, 18),   # upper-left room

    # ── top perimeter: only now briefly visit the outer top edge ───────
    ( 1, 24),   # top-left corner
    (12, 24),   # top-centre
    (20, 24),   # top-right

    # ── return to centre to close the loop ────────────────────────────
    (12, 12),   # back to centre junction
]

# --- Motion noise ---
ALPHA_FWD    = 0.05
ALPHA_ROT    = 0.01

# --- Sensor noise ---
LIDAR_SIGMA  = 0.1

# --- Filter ---
N            = 700   # more particles = visible cloud + better coverage
NUM_RAYS     = 80
FOV          = np.pi  # full 360° scan; much better localization
MAX_RANGE    = 10.0
SENSOR_SIGMA = 3.0   # wider sigma keeps particles alive longer

# --- Resampling / recovery ---
ESS_THRESH       = N / 3   # less aggressive resampling preserves diversity
INJECT_FRAC      = 0.08   # more injection = cloud never fully collapses
INJECT_THRESHOLD = 3.0 / N

# --- Motion gate ---
MOTION_THRESHOLD = 0.02
ANGLE_THRESHOLD  = 0.01

# --- Path follower ---
WAYPOINT_RADIUS = 0.4
ROBOT_SPEED     = 0.5
ROBOT_ROT_SPEED = 0.15

# --- Planner ---
GRID_RES     = 0.5
ROBOT_RADIUS = 0.6


######################################################################
#
#   World helpers
#
def inFreespace(x, y):
    if x <= xmin or x >= xmax or y <= ymin or y >= ymax:
        return False
    return obstacles.disjoint(Point(x, y))

def random_freespace_particle():
    while True:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        if inFreespace(x, y):
            return np.array([x, y, np.random.uniform(-np.pi, np.pi)])


######################################################################
#
#   Moving obstacles
#
#   Each obstacle is a small square polygon that bounces around freespace.
#   On each step it tries to move in its current direction; if the new
#   centre would be in an obstacle or out of bounds it picks a new random
#   heading and retries.  This gives organic-looking wandering.
#

class MovingObstacle:
    def __init__(self, x, y, r=OBS_RADIUS, speed=OBS_SPEED):
        self.x     = x
        self.y     = y
        self.r     = r
        self.speed = speed
        # Random initial heading
        self.vx = speed * np.cos(np.random.uniform(-np.pi, np.pi))
        self.vy = speed * np.sin(np.random.uniform(-np.pi, np.pi))

    def step(self):
        """Advance one timestep, bouncing off obstacles and boundaries."""
        nx = self.x + self.vx
        ny = self.y + self.vy
        # Check all four corners of the square so it never overlaps a wall
        corners_ok = all(
            inFreespace(nx + dx*self.r, ny + dy*self.r)
            for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]
        )
        if corners_ok:
            self.x, self.y = nx, ny
        else:
            # Pick a new random direction and try again next step
            angle    = np.random.uniform(-np.pi, np.pi)
            self.vx  = self.speed * np.cos(angle)
            self.vy  = self.speed * np.sin(angle)

    def polygon(self):
        r = self.r
        return Polygon([
            (self.x-r, self.y-r), (self.x+r, self.y-r),
            (self.x+r, self.y+r), (self.x-r, self.y+r)
        ])


def spawn_moving_obstacles(n):
    """
    Place n moving obstacles at random freespace locations that are also
    far enough from the robot start so they don't trap it on launch.
    """
    obs_list = []
    while len(obs_list) < n:
        x = np.random.uniform(xmin + 1, xmax - 1)
        y = np.random.uniform(ymin + 1, ymax - 1)
        # Must be in freespace and not too close to the robot start
        if inFreespace(x, y) and dist((x, y), (xstart, ystart)) > 3.0:
            obs_list.append(MovingObstacle(x, y))
    return obs_list


######################################################################
#
#   A* path planner
#
def _world_to_grid(x, y):
    return int((x - xmin) / GRID_RES), int((y - ymin) / GRID_RES)

def _grid_to_world(gx, gy):
    return xmin + (gx + 0.5) * GRID_RES, ymin + (gy + 0.5) * GRID_RES

def build_grid():
    cols = int((xmax - xmin) / GRID_RES)
    rows = int((ymax - ymin) / GRID_RES)
    grid = np.zeros((cols, rows), dtype=bool)
    for gx in range(cols):
        for gy in range(rows):
            wx, wy = _grid_to_world(gx, gy)
            pt = Point(wx, wy)
            if not obstacles.disjoint(pt.buffer(ROBOT_RADIUS)):
                grid[gx, gy] = True
    return grid, cols, rows

def astar(grid, cols, rows, start_g, goal_g):
    sx, sy = start_g
    gx, gy = goal_g

    def _snap(cx, cy):
        if not grid[cx, cy]:
            return cx, cy
        for radius in range(1, max(cols, rows)):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx2, ny2 = cx+dx, cy+dy
                    if 0 <= nx2 < cols and 0 <= ny2 < rows and not grid[nx2, ny2]:
                        return nx2, ny2
        raise ValueError(f"No free cell near ({cx},{cy})")

    sx, sy = _snap(sx, sy)
    gx, gy = _snap(gx, gy)

    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, (sx, sy)))
    came_from = {}
    g_cost    = {(sx, sy): 0.0}

    nb8 = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
           (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == (gx, gy):
            path, node = [], current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((sx, sy))
            path.reverse()
            return path
        if g > g_cost.get(current, float('inf')):
            continue
        cx, cy = current
        for dx, dy, sc in nb8:
            nx2, ny2 = cx+dx, cy+dy
            if not (0 <= nx2 < cols and 0 <= ny2 < rows): continue
            if grid[nx2, ny2]: continue
            ng = g_cost[current] + sc
            nb = (nx2, ny2)
            if ng < g_cost.get(nb, float('inf')):
                g_cost[nb]    = ng
                came_from[nb] = current
                h = abs(nx2-gx) + abs(ny2-gy)
                heapq.heappush(open_heap, (ng+h, ng, nb))
    return []

def string_pull(path_world):
    if len(path_world) <= 2:
        return path_world
    inflated = obstacles.context.buffer(ROBOT_RADIUS)
    pruned   = [path_world[0]]
    i = 0
    while i < len(path_world) - 1:
        j = len(path_world) - 1
        while j > i + 1:
            if LineString([path_world[i], path_world[j]]).disjoint(inflated):
                break
            j -= 1
        pruned.append(path_world[j])
        i = j
    return pruned

def plan_segment(start_xy, goal_xy, grid, cols, rows):
    """Plan and string-pull a single A* segment."""
    sg         = _world_to_grid(*start_xy)
    gg         = _world_to_grid(*goal_xy)
    grid_path  = astar(grid, cols, rows, sg, gg)
    if not grid_path:
        raise RuntimeError(f"A* found no path {start_xy} → {goal_xy}")
    world_path        = [_grid_to_world(gx, gy) for gx, gy in grid_path]
    world_path        = string_pull(world_path)
    world_path[0]     = start_xy
    world_path[-1]    = goal_xy
    return world_path

def build_sweep_path(waypoints):
    """
    Connect every consecutive pair in SWEEP_WAYPOINTS with an A* segment,
    then add a final segment back to waypoints[0] to close the loop.
    Duplicate junction points are removed so the robot doesn't stall.
    """
    print("Building sweep path with A*...")
    grid, cols, rows = build_grid()
    full_path = []

    # Close the loop by appending the start again at the end
    loop = list(waypoints) + [waypoints[0]]

    for i in range(len(loop) - 1):
        seg = plan_segment(loop[i], loop[i+1], grid, cols, rows)
        if full_path:
            seg = seg[1:]   # drop duplicate junction point
        full_path.extend(seg)

    print(f"  Sweep path: {len(full_path)} waypoints total (looping).")
    return full_path


######################################################################
#
#   Pure-pursuit path follower
#
def follow_path(x, y, th, waypoints):
    """
    Returns (fwd, dth, remaining_waypoints).
    Pops waypoints as they are reached.  Returns empty list when the
    entire path is consumed (caller should reload for looping).
    """
    if not waypoints:
        return 0.0, 0.0, waypoints

    while waypoints and dist((x, y), waypoints[0]) < WAYPOINT_RADIUS:
        waypoints = waypoints[1:]

    if not waypoints:
        return 0.0, 0.0, waypoints

    wx, wy  = waypoints[0]
    desired = np.arctan2(wy - y, wx - x)
    dth     = (desired - th + np.pi) % (2*np.pi) - np.pi
    dth     = np.clip(dth, -ROBOT_ROT_SPEED, ROBOT_ROT_SPEED)
    align   = np.cos(dth / ROBOT_ROT_SPEED * (np.pi / 2))
    fwd     = ROBOT_SPEED * max(0.0, align)
    return fwd, dth, waypoints


######################################################################
#
#   Visualization
#
class Visualization:
    def __init__(self, title="MCL — Sweep"):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self.ax.grid(True)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")

        for poly in obstacles.context.geoms:
            self.ax.plot(*poly.exterior.xy, 'k-', linewidth=2)

        self.particles_scatter = self.ax.scatter([], [], s=50, alpha=0.55,
                                                  zorder=3)
        self.est_point,        = self.ax.plot([], [], 'ro', markersize=6,
                                               zorder=5)
        self.est_heading,      = self.ax.plot([], [], 'r-', linewidth=2,
                                               zorder=5)
        self.true_point,       = self.ax.plot([], [], 'gx', markersize=8,
                                               zorder=5)
        self.true_heading,     = self.ax.plot([], [], 'g-', linewidth=2,
                                               zorder=5)
        self.lidar_scatter     = self.ax.scatter([], [], s=15, c='red',
                                                  zorder=4)
        self.path_line,        = self.ax.plot([], [], 'c--', linewidth=0.8,
                                               alpha=0.5)

        self.true_body = plt.Circle((0,0), 0.5, fill=False, color='green',
                                     zorder=5)
        self.est_body  = plt.Circle((0,0), 0.5, fill=False, linestyle='--',
                                     color='red', zorder=5)
        self.ax.add_patch(self.true_body)
        self.ax.add_patch(self.est_body)

        # Moving obstacle patches — created dynamically in update_moving_obs
        self._obs_patches = []

    def redraw(self):
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def update_path(self, waypoints):
        if waypoints:
            xs, ys = zip(*waypoints)
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

    def update_particles(self, particles, weights=None):
        P = np.asarray(particles, dtype=float)
        self.particles_scatter.set_offsets(P[:, :2])
        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            if len(w) == len(P) and np.isfinite(w).all():
                w = w - w.min()
                if w.max() > 0: w /= w.max()
                # When weights are uniform (spread out) all particles show equally.
                # When peaked (converged) high-weight particles are larger.
                self.particles_scatter.set_sizes(20 + 80*w)

    def update_estimate(self, x, y, th):
        self.est_point.set_data([x], [y])
        self.est_body.center = (x, y)
        self.est_heading.set_data([x, x+3*np.cos(th)], [y, y+3*np.sin(th)])

    def update_true(self, x, y, th):
        self.true_point.set_data([x], [y])
        self.true_body.center = (x, y)
        self.true_heading.set_data([x, x+3*np.cos(th)], [y, y+3*np.sin(th)])

    def update_lidar_hits(self, hits):
        if not hits:
            self.lidar_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.lidar_scatter.set_offsets(np.array(hits))

    def update_moving_obs(self, obs_list):
        """Redraw all moving obstacle squares. Reuses patches when possible."""
        # Remove old patches
        for p in self._obs_patches:
            p.remove()
        self._obs_patches = []
        for ob in obs_list:
            r  = ob.r
            patch = plt.Polygon(
                [(ob.x-r, ob.y-r), (ob.x+r, ob.y-r),
                 (ob.x+r, ob.y+r), (ob.x-r, ob.y+r)],
                closed=True, facecolor='orange', edgecolor='darkorange',
                alpha=0.7, zorder=4
            )
            self.ax.add_patch(patch)
            self._obs_patches.append(patch)


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
    if   gt == "Point":              candidates = [(inter.x, inter.y)]
    elif gt == "MultiPoint":         candidates = [(p.x, p.y) for p in inter.geoms]
    elif gt == "LineString":         candidates = list(inter.coords)[:2]
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
#   MCL
#
def sample_motion_model(particles, fwd, dth,
                        alpha_fwd=ALPHA_FWD, alpha_rot=ALPHA_ROT):
    N   = len(particles)
    out = particles.copy()
    fwd_n = np.random.normal(fwd, alpha_fwd*(abs(fwd)+1e-6), N)
    dth_n = np.random.normal(dth, alpha_rot*(abs(dth)+1e-6)+alpha_fwd*abs(fwd), N)
    out[:, 0] += fwd_n * np.cos(out[:, 2])
    out[:, 1] += fwd_n * np.sin(out[:, 2])
    out[:, 2] += dth_n
    out[:, 2]  = (out[:, 2] + np.pi) % (2*np.pi) - np.pi
    return out

def compute_weights(particles, z_measured, world_geom,
                    sigma=SENSOR_SIGMA, num_rays=NUM_RAYS,
                    fov=FOV, max_range=MAX_RANGE):
    N           = len(particles)
    log_weights = np.full(N, -1e9)   # log-domain; -1e9 ≈ zero probability

    for i, (px, py, pth) in enumerate(particles):
        if not inFreespace(px, py):
            continue  # stays at -1e9
        p_dists        = lidar_distances(px, py, pth, world_geom,
                                         num_rays=num_rays, fov=fov,
                                         max_range=max_range)
        diff           = z_measured - p_dists
        log_weights[i] = -0.5 * np.sum(diff**2) / sigma**2

    # Numerically stable softmax: subtract max before exp so the best
    # particle always maps to exp(0)=1.  Without this, ALL weights can
    # simultaneously underflow to 0 when the robot is in an open area
    # and every particle has a large squared error.
    log_weights -= np.max(log_weights)
    weights      = np.exp(log_weights)
    weights     += 1e-12   # floor so normalisation never divides by zero
    weights     /= weights.sum()
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

    # ── Build full looping sweep path ──────────────────────────────────
    # plan_segment is called once at startup using only the static
    # obstacles grid.  Moving obstacles are NOT in the planned path —
    # the robot's lidar will detect them at runtime and the MCL filter
    # will see unexpected range returns, which is the interesting part.
    full_sweep = build_sweep_path(SWEEP_WAYPOINTS)
    # We keep a separate display copy (never consumed) and a working
    # deque that gets popped as waypoints are reached.
    remaining  = list(full_sweep)
    lap        = 0

    viz.update_path(full_sweep)

    # ── Spawn moving obstacles ─────────────────────────────────────────
    moving_obs = spawn_moving_obstacles(NUM_MOVING_OBS)
    print(f"  Spawned {len(moving_obs)} moving obstacle(s).")

    # ── Initialise MCL particles ───────────────────────────────────────
    """particles = np.column_stack([
        xstart + 0.75*np.random.randn(N),
        ystart + 0.75*np.random.randn(N),
        np.random.uniform(-np.pi, np.pi, N)
    ])"""
    particles = np.array([random_freespace_particle() for _ in range(N)])
    weights  = np.ones(N) / N

    x, y, th = float(xstart), float(ystart), 0.0
    # Initialise estimate to start pose so follow_path has a valid
    # target on the very first iteration before the loop computes it.
    xhat, yhat, thhat = x, y, th
    N_INJECT = max(1, int(N * INJECT_FRAC))
    N_KEEP   = N - N_INJECT

    while True:

        # ── Reload path when the lap is complete (looping sweep) ───────
        if not remaining:
            lap      += 1
            remaining = list(full_sweep)
            print(f"  Lap {lap} complete — starting lap {lap+1}.")

        # ── Path follower ──────────────────────────────────────────────
        # CHANGE: path follower uses the MCL *estimate* (xhat,yhat,thhat),
        # not ground truth (x,y,th).  This is the correct real-robot design —
        # the controller only knows what the filter tells it.  If the filter
        # diverges the robot will visibly steer wrong, which is the intended
        # failure mode to observe.
        fwd, dth, remaining = follow_path(xhat, yhat, thhat, remaining)

        # ── True robot motion ──────────────────────────────────────────
        th += dth
        nx  = x + fwd * np.cos(th)
        ny  = y + fwd * np.sin(th)
        if inFreespace(nx, ny):
            x, y = nx, ny

        # ── Advance moving obstacles ───────────────────────────────────
        for ob in moving_obs:
            ob.step()

        # ── Build world geometry: static + moving obstacles ────────────
        # The robot's lidar sees moving obstacles as real objects, so they
        # must be unioned into world_geom for both lidar_scan and
        # lidar_distances on the true robot pose.
        # Particles do NOT know about moving obstacles — they only have
        # the static map.  This is realistic: a real robot's map doesn't
        # include dynamic objects.  The mismatch causes the particle
        # weights to drop transiently when a moving obs occludes a ray,
        # which you can observe in the ESS readout.
        moving_geom  = MultiPolygon([ob.polygon() for ob in moving_obs]) \
                       if moving_obs else None
        if moving_geom:
            world_geom_true = obstacles.context.union(moving_geom)
        else:
            world_geom_true = obstacles.context
        world_geom_particles = obstacles.context   # particles use static map

        # ── STEP 1: SAMPLE ─────────────────────────────────────────────
        particles = sample_motion_model(particles, fwd, dth)
        # Snap out-of-obstacle particles: try near the true robot pose first
        # (keeps positional information), fall back to random freespace only
        # if 10 nearby attempts fail (e.g. robot itself is near a wall).
        for i in range(N):
            if not inFreespace(particles[i, 0], particles[i, 1]):
                for _ in range(10):
                    px = x + np.random.randn() * 0.5
                    py = y + np.random.randn() * 0.5
                    if inFreespace(px, py):
                        particles[i, 0] = px
                        particles[i, 1] = py
                        break
                else:
                    particles[i] = random_freespace_particle()

        # ── STEP 2: WEIGHT ─────────────────────────────────────────────
        moved = (abs(fwd) > MOTION_THRESHOLD or abs(dth) > ANGLE_THRESHOLD)
        if moved:
            true_dists = lidar_distances(x, y, th, world_geom_true)
            z_measured = true_dists + np.random.normal(0, LIDAR_SIGMA,
                                                        size=true_dists.shape)
            z_measured = np.clip(z_measured, 0, MAX_RANGE)
            # Particles compare against the static map only — moving
            # obstacles appear as unexplained short-range returns, causing
            # a transient ESS drop that recovers once the obs moves away.
            weights    = compute_weights(particles, z_measured,
                                         world_geom_particles)

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

        # ── Draw ───────────────────────────────────────────────────────
        hits  = lidar_scan(x, y, th, world_geom_true)
        error = np.sqrt((x - xhat)**2 + (y - yhat)**2)

        viz.update_path(remaining)
        viz.update_particles(particles[:, :2], weights)
        viz.update_true(x, y, th)
        viz.update_estimate(xhat, yhat, thhat)
        viz.update_lidar_hits(hits)
        viz.update_moving_obs(moving_obs)
        viz.ax.set_title(
            f"Lap {lap+1}  |  Error: {error:.2f}  |  "
            f"ESS: {Neff:.0f}/{N}  |  wp: {len(remaining)}"
        )
        viz.redraw()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    _demo()