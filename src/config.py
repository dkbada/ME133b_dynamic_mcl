from math import pi

# Set map 
# Available keys:  "varied" | "hallways" | "circle" | "hexagon"
ACTIVE_MAP = "varied"

# World bounds (meters)
(xmin, xmax) = (0, 25)
(ymin, ymax) = (0, 25)

# Robot start pose (used to initialise the true robot position in _demo)
(xstart, ystart) = (1, 1)

# LIDAR params
NUM_RAYS  = 60       # angular resolution of the virtual LiDAR
FOV       = 2 * pi   # full 360° scan
MAX_RANGE = 36.0     # maximum range in world units
SIGMA     = 0.4      # std-dev of Gaussian measurement noise (world units)

# Predetermined path params
WAYPOINT_RADIUS = 0.5    # meters; close enough to count as "reached"
PATH_SPEED      = 0.15   # forward step size per iteration (world units)
MAX_TURN_RATE   = 0.12   # maximum heading correction per iteration (rad)

# Particle initialization
N = 500