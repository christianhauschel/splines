from splines import Spline
import numpy as np
import proplot as pplt

pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

crv = Spline(pts)

print(crv.evaluate(0.5))
print(crv.evaluate(np.array([0.5, 0.7])))

fig, ax = pplt.subplots()
crv.plot(ax)

# pplt.show()
