"""
Monte Carlo Localization (MCL) — Particle Filter Implementation
================================================================
MCL estimates a robot's pose (x, y, theta) by maintaining a set of N
weighted particles, each representing a hypothesis about where the robot is.
Each iteration:
  1. Motion update  — propagate particles forward using noisy odometry
  2. Sensor update  — weight each particle by how well its predicted LiDAR
                      scan matches the real scan
  3. Resample       — draw new particles proportional to their weights
                      (with optional random injection to aid global re-localization)

Dependencies: matplotlib, numpy, shapely, pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import N, NUM_RAYS, SIGMA, MAX_RANGE, xstart, ystart, ACTIVE_MAP, WAYPOINT_RADIUS
from maps import WAYPOINTS, inFreespace
from lidar import build_segment_list, _cast_rays
from particle_filter import random_free_particle, motion_update, compute_weights, resample
from path import move_toward
from vis import Visualization
from maps import obstacles

def _demo():
    """
    Autonomous demo: the robot follows the waypoints for the active map
    while the particle filter localises it in real time.

    Press Escape (or close the window) to stop early.  After the run ends
    (all waypoints visited, or early exit), error metrics are saved to
    'mcl_errors.csv' and a three-panel analysis plot is shown.
    """
    viz = Visualization(title=f"MCL — {ACTIVE_MAP}")

    # Exit flag
    running = {'go': True}
    def on_key(event):
        if event.key == 'escape':
            running['go'] = False
    viz.fig.canvas.mpl_connect('key_press_event', on_key)

    # Initialize particles
    particles = np.array([random_free_particle() for _ in range(N)])
    weights   = np.ones(N) / N   # uniform weights — pose unknown at start

    # True robot pose (starts at the world start position)
    x, y, th = float(xstart), float(ystart), 0.0

    # Pre-compute the segment list used by every LiDAR call
    segs = build_segment_list(obstacles.context)

    # Waypoint state
    wp_idx = 0   # index into WAYPOINTS of the current target

    # Error log
    # Stores per-iteration metrics 
    log = {'pos_error': [], 'ang_error': [], 'N_eff': []}

    # Main loop
    while running['go'] and wp_idx < len(WAYPOINTS):

        # Path following
        target = WAYPOINTS[wp_idx]
        if np.hypot(x - target[0], y - target[1]) < WAYPOINT_RADIUS:
            wp_idx += 1
            if wp_idx >= len(WAYPOINTS):
                break   # path complete
            target = WAYPOINTS[wp_idx]

        # Compute steering command toward current waypoint
        move_fwd, move_dth = move_toward(x, y, th, target)

        # 1. Move the true robot
        th += move_dth
        nx  = x + move_fwd * np.cos(th)
        ny  = y + move_fwd * np.sin(th)
        if inFreespace(nx, ny):   # only commit if the new pose is valid
            x, y = nx, ny

        # 2. Motion update
        # Propagate all particles with the same command + individual noise
        particles = motion_update(particles, move_fwd, move_dth)

        # Collision repair: particles that ended up inside obstacles are
        # jiggled slightly or re-sampled at random if jiggling fails.
        for i in range(N):
            if not inFreespace(particles[i, 0], particles[i, 1]):
                for _ in range(5):    # try 5 small perturbations
                    cx = particles[i, 0] + 0.4 * np.random.randn()
                    cy = particles[i, 1] + 0.4 * np.random.randn()
                    if inFreespace(cx, cy):
                        particles[i, 0] = cx
                        particles[i, 1] = cy
                        break
                else:
                    particles[i] = random_free_particle()   # teleport as last resort

        # 3. Sensor update (correction step)
        # Cast rays once; derive both visualisation hits and distance array.
        true_dists, ray_dx, ray_dy = _cast_rays(x, y, th, segs)
        hits = [
            (x + d * dx, y + d * dy)
            for d, dx, dy in zip(true_dists, ray_dx, ray_dy)
            if d < MAX_RANGE
        ]

        # Simulate noisy measurement:  z = d + η,  η ~ N(0, σ^2)
        z_real = true_dists + np.random.randn(NUM_RAYS) * SIGMA
        z_real = np.clip(z_real, 0.0, MAX_RANGE)

        # Compute importance weights from measurement likelihood
        weights = compute_weights(particles, z_real, segs)

        # 4. Pose estimate (weighted mean)
        # Effective sample size: N_eff = 1 / Σ w_i^2
        # Low N_eff means the distribution is concentrated on few particles
        # -> resample to redistribute.
        N_eff = 1.0 / max(np.sum(weights**2), 1e-300)

        xhat = np.average(particles[:, 0], weights=weights)
        yhat = np.average(particles[:, 1], weights=weights)
        # Circular mean for heading (avoids wrap-around artefacts near ±π)
        that = np.arctan2(
            np.average(np.sin(particles[:, 2]), weights=weights),
            np.average(np.cos(particles[:, 2]), weights=weights)
        )

        # 5. Log localization error
        pos_err = np.hypot(xhat - x, yhat - y)
        ang_err = abs(np.arctan2(np.sin(that - th), np.cos(that - th)))
        log['pos_error'].append(pos_err)
        log['ang_error'].append(ang_err)
        log['N_eff'].append(N_eff)

        # 6. Adaptive resampling
        # Only resample when N_eff drops below 50% of N to avoid
        # unnecessary sample impoverishment during accurate tracking.
        if N_eff < N * 0.5:
            particles = resample(particles, weights, N, inject_frac=0.05)

            # Post-resample jitter: breaks particle degeneracy
            particles[:, 0] += 0.03 * np.random.randn(N)
            particles[:, 1] += 0.03 * np.random.randn(N)
            particles[:, 2] += 0.01 * np.random.randn(N)

            weights = np.ones(N) / N   # reset to uniform after resample

        # Visualize
        viz.update_particles(particles, weights)
        viz.update_true(x, y, th)
        viz.update_estimate(xhat, yhat, that)
        viz.update_lidar_hits(hits)
        viz.redraw()

    # Save error log and show analysis plots
    plt.ioff()

    df = pd.DataFrame(log)
    out_csv = f"mcl_errors_{ACTIVE_MAP}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Error log saved to {out_csv}  ({len(df)} iterations)")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"MCL Localization Error Analysis — {ACTIVE_MAP}")

    axes[0].plot(df['pos_error'], color='steelblue')
    axes[0].set_ylabel("Position error (m)")
    axes[0].set_title("Euclidean distance: estimate vs. truth")
    axes[0].grid(True)

    axes[1].plot(np.degrees(df['ang_error']), color='darkorange')
    axes[1].set_ylabel("Heading error (°)")
    axes[1].set_title("Absolute heading error (wrapped to [0°, 180°])")
    axes[1].grid(True)

    axes[2].plot(df['N_eff'], color='seagreen')
    axes[2].axhline(N * 0.5, color='red', linestyle='--',
                    label=f'Resample threshold (N×0.5 = {N*0.5:.0f})')
    axes[2].set_ylabel("N_eff")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("Effective sample size (drops trigger resampling)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    out_png = f"mcl_errors_{ACTIVE_MAP}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Analysis plot saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    _demo()