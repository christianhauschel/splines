"""
https://pomax.github.io/bezierinfo/#projections
"""

# %%

import numpy as np
from splines import Spline, distance_pts
import proplot as pplt
from copy import copy

pt = np.array([-1, 0])


# points = np.random.rand(3, 2)
points = np.array(
    [[0.64332117, 0.7363026], [0.61800902, 0.00214729], [0.17353645, 0.33371567]]
)
crv = Spline(points)

# for (coordinate, index) in LUT:
#   q = distance(coordinate, p)
#   if q < d:
#     d = q
#     i = index

t = np.linspace(0, 1, 50)
pts_eval = crv.evaluate(t)


pt_proj, t_proj = crv.project_point(pt)

fig, ax = pplt.subplots(aspect=1, figsize=(4, 4))
crv.plot(ax, dotted=False)
ax.plot(pt[0], pt[1], "x")
ax.plot(pt_proj[0], pt_proj[1], "o")
# ax.plot(pt0[0], pt0[1], "o")


# %%
