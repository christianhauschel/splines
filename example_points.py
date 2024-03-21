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
from splines import farthest_apart_points, area_tri, distance_pts, Line
import numpy as np

# enclose circle 
import smallestenclosingcircle

# %%

# af = Airfoil.naca("0012", n=51)
# af.add_TE_thickness(0.01)
# points = af.data
# points = np.roll(af.data, np.random.randint(0, af.n), axis=0)


# %%

# random points
# points = np.random.rand(6, 2)

points = np.array(
    [
        [0.05328671, 0.75379551],
        [0.22475366, 0.10027644],
        [0.27405434, 0.5932981],
        [0.35790505, 0.88768214],
        [0.25576468, 0.44421372],
        [0.90777934, 0.36292922],
    ]
)

# randomize the index of the points
np.random.shuffle(points)


hull = ConvexHull(points)
hull_indices = hull.vertices
hull_points = points[hull.vertices]

# create circle 
center_x, center_y, radius = smallestenclosingcircle.make_circle(points)


# distance, pt1, pt2 = farthest_apart_points(points)

hull = ConvexHull(points)
hull_indices = hull.vertices
hull_points = points[hull_indices]
n = len(hull_points)

k = 1
while k < n:
    crv_area = area_tri(hull_points[0], hull_points[n - 1], hull_points[k])
    nxt_area = area_tri(hull_points[0], hull_points[n - 1], hull_points[k + 1])

    if crv_area > nxt_area:
        break

    k += 1

p = 0
q = copy(k)

answer = distance_pts(hull_points[p], hull_points[q])

id1 = 0
id2 = 0

flag = False

while p <= k and q < n:
    while q < n:
        crv_area = area_tri(hull_points[p], hull_points[p + 1], hull_points[q])
        nxt_area = area_tri(
            hull_points[p], hull_points[p + 1], hull_points[(q + 1) % n]
        )

        if crv_area > nxt_area:
            break
        
        q += 1

        print(answer)

        if distance_pts(hull_points[p], hull_points[q % n]) > answer:
            answer = distance_pts(hull_points[p], hull_points[q % n])
            id1 = copy(p)
            id2 = copy(q % n)
            print("----")
            print(answer)
            print(id1)
            print(id2)
            print("----")
    p += 1
    

print("answer", answer)
print("diameter", radius*2)

pt1 = hull_points[id1]
pt2 = hull_points[id2]


# center_x, center_y, radius = smallestenclosingcircle.make_circle(points)


# line_chord = Line(pt1, pt2)


fig, ax = pplt.subplots(dpi=300, aspect="equal", figsize=(7, 4))
# plot circle
# ax.plot(center_x, center_y, "o", label="Center of circle")


# circle1 = plt.Circle((center_x, center_y), radius, color='r', fill=None)
# plt.gca().add_patch(circle1)

ax.plot(hull_points[:, 0], hull_points[:, 1], "-", alpha=1, label="Convex Hull")
ax.format(xlim=(-0.01, 1.01))
ax.plot(points[:, 0], points[:, 1], ".", label="Curve")
ax.plot(pt1[0], pt1[1], "x", label="Pt1")
ax.plot(pt2[0], pt2[1], "x", label="Pt2")
# line_chord.plot(ax)


# Split crv by the line


# %%
