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
from splines import Line, Spline 


af = Airfoil.naca("0012", n=31)
af.add_TE_thickness(0.01)
points = af.data
points = np.roll(af.data, 5, axis=0)


# randomize the index of the points
np.random.shuffle(points)



hull = ConvexHull(points)
hull_indices = hull.vertices
hull_points = points[hull.vertices]




from splines import farthest_apart_points, area_tri, distance_pts

pt1, pt2 = farthest_apart_points(points)


line_chord = Line(pt1, pt2)


fig, ax = pplt.subplots(dpi=300, aspect="equal", figsize=(7,4))
ax.plot(hull_points[:, 0], hull_points[:, 1], "-", alpha=1, label="Convex Hull")
ax.plot(points[:, 0], points[:, 1], ".", label="Curve")
line_chord.plot(ax)


# Split crv by the line



# %%

