import math


def distance(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def find_max_distance_3(points, a, b, c):
    dist_ab = distance(points[a], points[b])
    dist_ac = distance(points[a], points[c])
    dist_bc = distance(points[b], points[c])
    return max(dist_ab, dist_ac, dist_bc)

def next_counter_clockwise(points, i):
    return (i + 1) % len(points)

def prev_counter_clockwise(points, i):
    return (i - 1) % len(points)

def find_antipodal_index(hull_points, current_index, start_antipodal_index):
    max_dist = -1
    antipodal_index = None
    for i in range(len(hull_points)):
        dist = distance(hull_points[current_index], hull_points[i])
        if dist > max_dist:
            max_dist = dist
            antipodal_index = i
    return antipodal_index

def find_largest_distance(points):
    hull_points = points.copy()

    if len(hull_points) == 3:
        return find_max_distance_3(hull_points, 0, 1, 2)

    current_index = 0
    prev_index = prev_counter_clockwise(hull_points, current_index)
    antipodal_index = find_antipodal_index(hull_points, current_index, 1)

    max_dist = find_max_distance_3(hull_points, current_index, prev_index, antipodal_index)
    dist = 0
    turn_complete = False

    while not turn_complete:
        prev_index = current_index
        current_index = next_counter_clockwise(hull_points, current_index)
        antipodal_index = find_antipodal_index(hull_points, current_index, antipodal_index)
        dist = find_max_distance_3(hull_points, current_index, prev_index, antipodal_index)
        if dist > max_dist:
            max_dist = dist
        if current_index == 0:
            turn_complete = True

    return max_dist

# Example usage:

# random numpy points 
import numpy as np
from airfoil import Airfoil 
af = Airfoil.naca("0012", n=601)
points = np.random.rand(101, 2)
points = af.data

# points = 
print("Largest distance:", find_largest_distance(points))