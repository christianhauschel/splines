# %%

from splines import Spline, intersection, Line
import numpy as np
try:
    import proplot as pplt
except:
    pass


# seed numpy
np.random.seed(100)

# random points
pts1 = np.random.rand(3, 2)
pts2 = np.random.rand(3, 2)


crv1 = Spline(pts1)
crv2 = Spline(pts2, spline_type="bspline_inter")

pts3 = np.array([[0, 0], [1, 1]])
crv3 = Line(pts3)

pts_intersection = intersection(crv1, crv2)

pt = np.array([0.5, 0.5])
pt_projected, _ = crv1.project_point(pt)

fig, ax = pplt.subplots(figsize=(5, 4))
crv1.plot(ax, c="C0", label="BSpline")
crv2.plot(ax, c="C1", label="Interpolated BSpline")
crv3.plot(ax, c="C2", label="Line")
ax.plot(pts1[:, 0], pts1[:, 1], "o", c="C0")
ax.plot(pts2[:, 0], pts2[:, 1], "o", c="C1")
ax.plot(pts_intersection[:, 0], pts_intersection[:, 1], "x", c="k", label="Intersection")
ax.plot(pt[0], pt[1], "o", c="g", label="Point to project")
ax.plot(pt_projected[0], pt_projected[1], "o", c="r", label="Projected point")
ax.legend(ncols=1)
fig.savefig("doc/img/example.png", dpi=300)

print(crv1.arclength())
print(crv1.center)
crv1.refine_knotvector(2)

pplt.show()
