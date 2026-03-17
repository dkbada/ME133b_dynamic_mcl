import numpy as np
from config import NUM_RAYS, FOV, MAX_RANGE, xmin, ymin, xmax, ymax
from maps import obstacles

def build_segment_list(world_geom):
    """
    Convert all obstacle polygons and the four world-boundary walls into a
    flat (M, 4) array of line segments [x1, y1, x2, y2].

    Pre-computing segments once and reusing them is much faster than
    querying Shapely geometries inside the hot LiDAR inner loop.
    """
    segs = []
    for poly in world_geom.geoms:
        coords = list(poly.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            segs.append([a[0], a[1], b[0], b[1]])

    # World boundary walls (in order: bottom, right, top, left)
    segs.append([xmin, ymin, xmax, ymin])
    segs.append([xmax, ymin, xmax, ymax])
    segs.append([xmax, ymax, xmin, ymax])
    segs.append([xmin, ymax, xmin, ymin])
    return np.array(segs)


def _cast_rays(x, y, theta, segs):
    """
    Shared ray-casting core used by both `lidar_scan` and `lidar_distances`.

    Casts NUM_RAYS evenly-spaced rays from pose (x, y, theta) against every
    segment in `segs` and returns the per-ray travel distance to the nearest
    hit, along with the corresponding unit direction vectors.

    Ray vs. segment intersection (parametric form):
      Ray:     P(t) = origin + t * d,          t ∈ [0, MAX_RANGE]
      Segment: Q(u) = seg_start + u * s,       u ∈ [0, 1]
      Solve P(t) = Q(u):
        denom = dx*sy - dy*sx
        t     = (ao_x*sy - ao_y*sx) / denom
        u     = (ao_x*dy - ao_y*dx) / denom
      A valid intersection requires denom ≠ 0, t ∈ [0, MAX_RANGE], u ∈ [0, 1].

    Returns
    -------
    dists : np.ndarray, shape (NUM_RAYS,)
        Distance to the closest obstacle along each ray; MAX_RANGE if no hit.
    ray_dx, ray_dy : np.ndarray, shape (NUM_RAYS,)
        Unit direction vectors for each ray (needed by `lidar_scan` to recover
        hit coordinates from distances).
    """
    dists  = np.empty(NUM_RAYS)
    ray_dx = np.empty(NUM_RAYS)
    ray_dy = np.empty(NUM_RAYS)

    for i in range(NUM_RAYS):
        angle  = theta - FOV/2 + i * FOV / (NUM_RAYS - 1)
        dx, dy = np.cos(angle), np.sin(angle)
        best_t = MAX_RANGE

        for seg in segs:
            x1, y1, x2, y2 = seg
            sx, sy = x2 - x1, y2 - y1
            denom  = dx * sy - dy * sx

            if abs(denom) < 1e-10:    # ray and segment are parallel — skip
                continue

            ao_x, ao_y = x1 - x, y1 - y
            t = (ao_x * sy - ao_y * sx) / denom
            u = (ao_x * dy - ao_y * dx) / denom

            if 0 <= t <= MAX_RANGE and 0 <= u <= 1:
                best_t = min(best_t, t)

        dists[i]  = best_t
        ray_dx[i] = dx
        ray_dy[i] = dy

    return dists, ray_dx, ray_dy


def lidar_scan(x, y, theta, segs):
    """
    Return the (x, y) world coordinates of each LiDAR ray's closest hit.

    Only rays that actually strike something closer than MAX_RANGE are
    included in the output list (used for visualisation only).
    Delegates all ray-casting to `_cast_rays`.
    """
    dists, ray_dx, ray_dy = _cast_rays(x, y, theta, segs)
    return [
        (x + d * dx, y + d * dy)
        for d, dx, dy in zip(dists, ray_dx, ray_dy)
        if d < MAX_RANGE
    ]


def lidar_distances(x, y, theta, segs):
    """
    Return the ray-travel distance to the closest obstacle for each ray
    as a 1-D numpy array of shape (NUM_RAYS,).

    Returns MAX_RANGE for rays that hit nothing within range.
    Delegates all ray-casting to `_cast_rays`.
    """
    dists, _, _ = _cast_rays(x, y, theta, segs)
    return dists


def all_particle_lidar_distances(particles, segs):
    """
    Vectorised LiDAR: compute expected ray distances for ALL particles at
    once using NumPy broadcasting.  Replaces N individual calls to
    `lidar_distances`, giving a large speed-up when N is large.

    Shape summary:
      particles : (N, 3)  — [x, y, theta] per particle
      segs      : (M, 4)  — [x1, y1, x2, y2] per wall segment
      return    : (N, R)  — distance per particle per ray

    Broadcasting dimensions:
      angles  : (N, R)       — absolute ray direction per particle per ray
      dx, dy  : (N, R, 1)    — ray unit vectors, broadcast over segments
      ox, oy  : (N, 1, 1)    — particle origins, broadcast over rays & segs
      t, u    : (N, R, M)    — intersection parameters
    """
    xs     = particles[:, 0]
    ys     = particles[:, 1]
    thetas = particles[:, 2]

    # Angles: (N, R)
    angles = thetas[:, None] - FOV/2 + np.arange(NUM_RAYS) * FOV / (NUM_RAYS - 1)
    # Ray direction vectors: (N, R, 1)
    dx = np.cos(angles)[:, :, None]
    dy = np.sin(angles)[:, :, None]
    # Particle origins: (N, 1, 1)
    ox = xs[:, None, None]
    oy = ys[:, None, None]

    # Segment endpoints: (1, 1, M)
    x1 = segs[None, None, :, 0];  y1 = segs[None, None, :, 1]
    x2 = segs[None, None, :, 2];  y2 = segs[None, None, :, 3]
    sx = x2 - x1;  sy = y2 - y1

    # Denominator (zero ↔ parallel ray/segment)
    denom = dx * sy - dy * sx
    ao_x  = x1 - ox
    ao_y  = y1 - oy

    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(np.abs(denom) > 1e-10, (ao_x * sy - ao_y * sx) / denom, np.inf)
        u = np.where(np.abs(denom) > 1e-10, (ao_x * dy - ao_y * dx) / denom, np.inf)

    # Mask out intersections outside the ray (t<0), beyond max range,
    # or outside the segment (u outside [0,1])
    valid = (t >= 0) & (t <= MAX_RANGE) & (u >= 0) & (u <= 1)
    t[~valid] = np.inf

    # Closest hit per (particle, ray)
    dists = np.min(t, axis=2)          # (N, R)
    dists[np.isinf(dists)] = MAX_RANGE
    return dists

