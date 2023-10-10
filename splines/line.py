from .spline import Spline

class Line(Spline):
    def __init__(self, pts):
        super().__init__(pts, spline_type="bspline_inter", degree=1)