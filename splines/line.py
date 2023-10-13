import numpy as np
from .spline import Spline

class Line(Spline):
    def __init__(self, pt1, pt2):
        pts = np.array([pt1, pt2])
        super().__init__(pts, spline_type="bspline_inter", degree=1)