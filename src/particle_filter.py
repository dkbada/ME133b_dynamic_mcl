import numpy as np
from math import pi
from config import SIGMA, xmin, xmax, ymin, ymax
from maps import inFreespace
from lidar import all_particle_lidar_distances

def compute_weights(particles, z_real, segs):
    """
    Compute normalised importance weights for all particles.

    Weight model:
      • Particles outside free space get weight ≈ 0  (log-weight = -1e9).
      • For valid particles, weight is the product of per-ray Gaussian
        likelihoods:  w ∝ exp(−Σ (z_real_i - z_expected_i)^2 / (2 σ^2))
        Computed in log-space for numerical stability, then exponentiated.

    If the total weight collapses to near zero (all particles are very
    unlikely), the weights are reset to uniform to avoid filter collapse.
    """
    # Freespace mask: penalise particles that have wandered into obstacles
    in_free = np.array([inFreespace(p[0], p[1]) for p in particles])

    # Expected LiDAR distances from each particle's pose
    all_dists = all_particle_lidar_distances(particles, segs)

    # Log-likelihood under Gaussian noise model (sum over all rays)
    diff        = z_real[None, :] - all_dists        # (N, R)
    log_weights = -0.5 * np.sum(diff**2, axis=1) / SIGMA**2   # (N,)

    # Zero-out out-of-bounds particles in log domain
    log_weights[~in_free] = -1e9

    # Shift by max for numerical stability before exp
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    total   = weights.sum()

    if total <= 1e-300 or not np.isfinite(total):
        # Filter degeneracy: fall back to uniform (global re-localisation)
        weights = np.ones(len(particles)) / len(particles)
    else:
        weights /= total
    return weights

def random_free_particle():
    """
    Sample a uniformly random pose (x, y, theta) from free space.
    Used to initialise the particle set and for random injection during
    resampling (supports global re-localisation after kidnapping).
    """
    while True:
        px = np.random.uniform(xmin + 0.5, xmax - 0.5)
        py = np.random.uniform(ymin + 0.5, ymax - 0.5)
        if inFreespace(px, py):
            return [px, py, np.random.uniform(-pi, pi)]


def resample(particles, weights, N, inject_frac=0.05):
    """
    Low-variance (systematic) resampling with random injection.

    Steps:
      1. Keep `n_keep = N * (1 − inject_frac)` particles drawn
         proportional to weights using the systematic resampling algorithm
         (evenly spaced pointer sweep over the CDF).
      2. Inject `N − n_keep` fresh random particles from free space.
         This prevents filter collapse and enables recovery from poor
         initial localisation or sudden pose changes (kidnapping).

    Parameters
    ----------
    inject_frac : float
        Fraction of the population replaced by random free-space samples
        at every resample step.  Higher values → faster recovery but more
        noise when already localised.  Typical range: 0.01–0.10.
    """
    n_keep = N - int(N * inject_frac)

    # Systematic resampling: O(N) and lower variance than multinomial
    cumsum  = np.cumsum(weights)
    step    = 1.0 / n_keep
    start   = np.random.uniform(0, step)
    pos     = start + step * np.arange(n_keep)
    indices = np.clip(np.searchsorted(cumsum, pos), 0, N - 1)

    kept     = particles[indices].copy()
    injected = np.array([random_free_particle() for _ in range(N - n_keep)])
    return np.vstack([kept, injected])


def motion_update(particles, fwd, dth, alpha=(0.10, 0.04)):
    """
    Probabilistic odometry motion model.

    Applies the commanded motion (fwd, dth) to every particle, plus
    independent Gaussian noise scaled by the motion magnitude.

    Parameters
    ----------
    fwd   : float   — commanded forward translation (world units)
    dth   : float   — commanded rotation (radians)
    alpha : tuple   — noise coefficients
              alpha[0] : forward noise proportional to |fwd|
              alpha[1] : heading noise proportional to |dth| + small |fwd| term
                         (models wheel-slip even during straight driving)

    NOTE: The rotation is applied BEFORE the translation so heading is
          updated first, which matches how a differential-drive robot
          steers then drives.  This can introduce a small systematic
          error for large dth values; a two-step (rot1, trans, rot2) model
          is more accurate.
    """
    N = len(particles)

    fwd_noise = alpha[0] * abs(fwd)
    th_noise  = alpha[1] * abs(dth) + 0.008 * abs(fwd)

    # 1. Rotate each particle (with noise)
    particles[:, 2] += dth + th_noise * np.random.randn(N)

    # 2. Translate each particle along its (now updated) heading (with noise)
    noisy_fwd = fwd + fwd_noise * np.random.randn(N)
    particles[:, 0] += noisy_fwd * np.cos(particles[:, 2])
    particles[:, 1] += noisy_fwd * np.sin(particles[:, 2])

    return particles
