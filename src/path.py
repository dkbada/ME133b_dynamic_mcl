import numpy as np
from config import PATH_SPEED, MAX_TURN_RATE

def move_toward(x, y, th, target):
    """
    Compute a (fwd, dth) command that steers the robot one step toward
    `target = (tx, ty)` using a simple proportional heading controller.

    Strategy:
      1. Compute the bearing to the target.
      2. Find the signed heading error and clip it to ±MAX_TURN_RATE so
         the robot turns smoothly rather than snapping.
      3. Only drive forward when roughly aligned (|error| < 0.3 rad) to
         avoid cutting corners into obstacles.

    Returns
    -------
    fwd : float   — forward displacement this step (0 if mis-aligned)
    dth : float   — heading correction this step (rad)
    """
    tx, ty  = target
    desired = np.arctan2(ty - y, tx - x)
    err     = np.arctan2(np.sin(desired - th), np.cos(desired - th))
    dth     = np.clip(err, -MAX_TURN_RATE, MAX_TURN_RATE)
    fwd     = PATH_SPEED if abs(err) < 0.3 else 0.0
    return fwd, dth


