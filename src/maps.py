from math             import pi
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.geometry import Point as _Point 
from shapely.prepared import prep
from config import xmin, ymin, xmax, ymax, ACTIVE_MAP

MAP_REGISTRY = {

    # Varied obstacles
    # Asymmetric rooms — good for testing convergence from different poses.
    "varied": {
        "obstacles": MultiPolygon([
            Polygon([[2, 14], [5, 14], [5, 23], [2, 23]]),
            Polygon([[7, 23], [11, 23], [11, 20], [7, 20]]),
            Polygon([[7, 17], [11, 17], [11, 14], [7, 14]]),
            Polygon([[14, 23], [19, 23], [19, 20], [16, 20], [16, 14], [14, 14]]),
            Polygon([[21, 23], [23, 23], [23, 14], [18, 14], [18, 17], [21, 17]]),
            Polygon([[2, 11], [8, 11], [8, 8], [6, 8], [6, 2], [4, 2], [4, 8], [2, 8]]),
            Polygon([[10, 11], [15, 11], [15, 8], [12, 8], [12, 5], [15, 5], [15, 2], [10, 2]]),
            Polygon([[17, 11], [23, 11], [23, 8], [21, 8], [21, 5], [20, 5], [20, 2], [17, 2]]),
        ]),
        "waypoints": [
             ( 1,  1), (24,  1), (24, 24), ( 1, 24), ( 1,  1),
        ],
    },

    # Long parallel hallways
    # Symmetric map — deliberate ambiguity challenges the particle filter
    # to distinguish nearly-identical corridors; a classic MCL stress test.
    "hallways": {
        "obstacles": MultiPolygon([
            Polygon([[2, 23], [23, 23], [23, 20], [2, 20]]),
            Polygon([[2, 17], [23, 17], [23, 14], [2, 14]]),
            Polygon([[2, 11], [23, 11], [23,  8], [2,  8]]),
            Polygon([[2,  5], [23,  5], [23,  2], [2,  2]]),
        ]),
        # Sweep through each corridor
        "waypoints": [
            ( 0.5,  1), (24.5,  1), 
            (24.5,  7), ( 0.5,  6), 
            ( 0.5,  12.5), (24.5,  12.5),  
            (24.5, 18.5), ( 0.5, 18.5), 
            ( 0.5, 24.5), (24.5, 24.5),  
        ],
    },

    # Circle obstacle
    # Single large circular obstacle centred in the world.
    "circle": {
        "obstacles": MultiPolygon([
            _Point(12.5, 12.5).buffer(7.0)
        ]),
        # Clockwise loop around the outside of the circle
        "waypoints": [
            ( 2,  2), (23,  2), (23, 23), ( 2, 23), ( 2,  2),
        ],
    },

    # Hexagon obstacle
    # Single large hexagonal obstacle — tests how rays handle diagonal walls.
    "hexagon": {
        "obstacles": MultiPolygon([
            Polygon([[7, 4], [18, 4], [22, 12], [18, 21], [7, 21], [3, 12]])
        ]),
        # Loop around the outside of the hexagon
        "waypoints": [
            ( 2,  2), (23,  2), (23, 23), ( 2, 23), ( 2,  2),
        ],
    },
}

# Load the selected map
assert ACTIVE_MAP in MAP_REGISTRY, (
    f"Unknown map '{ACTIVE_MAP}'. "
    f"Valid keys: {list(MAP_REGISTRY.keys())}"
)

_entry    = MAP_REGISTRY[ACTIVE_MAP]
obstacles = prep(_entry["obstacles"])
WAYPOINTS = _entry["waypoints"]


def inFreespace(x, y):
    """Return True iff (x, y) is inside the world bounds and outside every obstacle."""
    if x <= xmin or x >= xmax or y <= ymin or y >= ymax:
        return False
    return obstacles.disjoint(Point(x, y))