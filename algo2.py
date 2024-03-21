"""
Script to test the most distant pair of points algorithm.

https://www.baeldung.com/cs/most-distant-pair-of-points
"""

# %%

from copy import copy
import proplot as pplt
from airfoil import Airfoil
import numpy as np
from scipy.spatial import ConvexHull
from splines import Line, Spline, area_tri, distance_pts, farthest_apart_points

af = Airfoil.naca("0012", n=51)
af.add_TE_thickness(0.01)
points = af.data
points = np.roll(af.data, np.random.randint(0, af.n), axis=0)

# random points
points = np.random.rand(50, 2)

# randomize the index of the points
np.random.shuffle(points)


hull = ConvexHull(points)
hull_indices = hull.vertices
p = points[hull_indices]
m = len(p)


def dist(q, r, p):
    """distance between point p and line qr"""

    return np.abs(
        (p[0] - q[0]) * (r[1] - q[1]) - (p[1] - q[1]) * (r[0] - q[0])
    ) / np.sqrt((r[0] - q[0]) ** 2 + (r[1] - q[1]) ** 2)


A = []
k = 2

# Find p_k
while dist(p[m - 1], p[0], p[k]) > dist(p[m - 1], p[0], p[k - 1]):
    k += 1


i = 1
j = copy(k)

while i <= k and j <= m:

    A.append((p[i - 1], p[j - 1]))

    while dist(p[i - 1], p[i], p[j - 1]) > dist(p[i - 1], p[i], p[j - 1]) and j < m:
        A.append((p[i - 1], p[j - 1]))
        j += 1
    i += 1

# Scan A to find the most distant pair of points
dist_max = 0
pair_max = None

for pair in A:
    d = distance_pts(pair[0], pair[1])
    if d > dist_max:
        dist_max = d
        pair_max = pair

pt3, pt4 = farthest_apart_points(points)

pt1 = pair_max[0]
pt2 = pair_max[1]

line_chord = Line(pt1, pt2)


fig, ax = pplt.subplots(dpi=300, aspect="equal", figsize=(7, 4))
ax.plot(p[:, 0], p[:, 1], "-", alpha=1)
ax.set(xlim=(-0.01, 1.01))
ax.plot(points[:, 0], points[:, 1], ".")
line_chord.plot(ax, label="TCS Algo")

try:
    line_ref = Line(pt3, pt4)
    line_ref.plot(ax, label="Ref")
except:
    "error in ref"

ax.legend(ncols=1)


# Split crv by the line


# %%
