import bisect
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
(xmin, xmax) = (0, 20)
(ymin, ymax) = (0, 12)

# Collect all the triangles and prepare (for faster checking).
obstacles = prep(MultiPolygon([
    Polygon([[7,  3], [3,  3], [3,  4], [7,  3]]),
    Polygon([[5,  5], [7,  7], [4,  6], [5,  5]]),
    Polygon([[9,  2], [8,  7], [6,  5], [9,  2]]),
    Polygon([[1, 10], [7, 10], [4,  8], [1, 10]])]))

# Define the start/goal states (x, y, theta)
(xstart, ystart) = (6, 1)
(xgoal,  ygoal)  = (5, 11)


######################################################################
#
#   Visualization Class (No Fixes Needed)
#
#   This renders the world.  In particular it provides the methods:
#     show(text = '')                   Show the current figure
#     drawNode(node,         **kwargs)  Draw a single node
#     drawEdge(node1, node2, **kwargs)  Draw an edge between nodes
#     drawPath(path,         **kwargs)  Draw a path (list of nodes)
#
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the triangles.
        for poly in obstacles.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

        # Show immediately.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, **kwargs):
        plt.plot(node.x, node.y, **kwargs)

    def drawEdge(self, head, tail, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), **kwargs)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)

######################################################################
#
#   Node Definition
#
class Node():
    #################
    # Initialization:
    def __init__(self, x, y):
        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

        # Edges = set of neighbors.  This needs to filled in.
        self.neighbors = set()

        # Clear the status, connection, and costs for the A* search tree.
        #   TRUNK:  done = True
        #   LEAF:   done = False, seen = True
        #   AIR:    done = False, seen = False
        self.done     = False
        self.seen     = False
        self.parent   = None
        self.creach   = 0               # Known/actual cost to get here
        self.ctogoest = inf             # Estimated cost to go from here

    ###############
    # A* functions:
    # Actual cost to connect to a neighbor and estimated to-go cost to
    # a distant (goal) node.
    def costToConnect(self, other):
        return self.distance(other)

    def costToGoEst(self, goal):
        return self.distance(goal)

    # Define the "less-than" to enable sorting in A*.  Use total cost estimate.
    def __lt__(self, other):
        return (self.creach + self.ctogoest) < (other.creach + other.ctogoest)

    ################
    # PRM functions:
    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    # Check whether in free space.
    def inFreespace(self):
        point = Point(self.x, self.y)
        return obstacles.disjoint(point)

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        line = LineString([(self.x, self.y), (other.x, other.y)])
        return obstacles.disjoint(line)

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

######################################################################
#
#  Main Code (No Fixes Needed)
#
def main():
    # Create the figure.  Some computers seem to need an additional show()?
    visual = Visualization()
    visual.show()
    plt.show()

if __name__== "__main__":
    main()