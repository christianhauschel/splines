from scipy.stats import qmc
from scipy.optimize import Bounds
from sklearn.cluster import DBSCAN
from geomdl.operations import split_curve
from geomdl.visualization import VisMPL as vis
import numpy as np
import proplot as pplt
from geomdl import BSpline, knotvector, NURBS
from geomdl.fitting import interpolate_curve, approximate_curve
from copy import copy


class Spline:
    def __init__(
        self,
        pts: np.array,
        weights: np.array = None,
        knots=None,
        spline_type="bspline",
        degree=2,
    ):
        """Generates a geomdl spline.
        Parameters
        ----------
        pts : np.array
            control/interpolation points
        weights : np.array, optional
            NURBS weighting, by default None
        knots : _type_, optional
            knot vector, by default None
        spline_type : str, optional
            ["bspline", "nurbs", "bspline_inter", "bspline_approx"], by default "bspline"
        degree : int, optional
            degree of the spline, by default 2
        """

        if spline_type in ["bspline", "nurbs"]:
            self.spline = self._spline_geomdl(spline_type, degree, pts, knots)
        elif spline_type == "bspline_inter":
            self.spline = interpolate_curve(pts.tolist(), degree)
        elif spline_type == "bspline_approx":
            self.spline = approximate_curve(pts.tolist(), degree)
        self.spline.vis = vis.VisCurve3D() if pts.shape[1] == 3 else vis.VisCurve2D()

        if spline_type == "nurbs" and weights is not None:
            self.spline.weights = weights.tolist()

    @staticmethod
    def _spline_geomdl(spline_type, degree, pts, knots):
        result = BSpline.Curve() if spline_type == "bspline" else NURBS.Curve()
        result.degree = degree
        result.ctrlpts = pts.tolist()

        if knots is None:
            result.knotvector = knotvector.generate(result.degree, result.ctrlpts_size)
        else:
            result.knotvector = knotvector
        return result

    def plot(self, ax, n_pts=100, dotted=False, **kwargs):
        try:
            t0 = self.spline.start()[0]
            t1 = self.spline.end()[0]
        except:
            t0 = 0.0
            t1 = 1.0
        coords_crv = self.evaluate(np.linspace(t0, t1, n_pts))

        if dotted:
            ax.plot(coords_crv[:, 0], coords_crv[:, 1], ".-", **kwargs)
        else:
            ax.plot(coords_crv[:, 0], coords_crv[:, 1], **kwargs)

    def evaluate(self, t, order=1, derivative=0):
        # check if t is array of single value
        if type(t) == np.ndarray:
            return self._evaluate_extra(t, order=order, d=derivative)
        else:
            return self._evaluate_extra_single(t, order=order, d=derivative)

    def _evaluate_extra_single(self, t_extra: float, order=1, d=0) -> np.array:
        """Extrapolation of spline using Taylor expansion.
        Parameters
        ----------
        self.spline : splipy.Curve
            Curve
        t_extra : float
            Parameter to extrapolate
        order : int, optional
            Taylor expansion order, by default 2
        Returns
        -------
        np.array
            Point
        """

        correction = 0.0

        if type(self.spline) in [BSpline.Curve, NURBS.Curve]:
            t = np.clip(t_extra, 0.0, 1.0)
            for i in range(order + 1):
                tangent = np.array(self.spline.derivatives(u=t, order=d + i)[-1])
                correction += tangent * (t_extra - t) ** i / np.math.factorial(i)
        else:
            t = np.clip(t_extra, self.spline.start(), self.spline.end())
            for i in range(order + 1):
                tangent = self.spline.derivative(t, d=i + d)
                correction += tangent * (t_extra - t) ** i / np.math.factorial(i)
        return correction

    def _evaluate_extra(self, t: np.array, order=1, d=0) -> np.array:
        """Extrapolates and evaluates a curve.
        Parameters
        ----------
        self.spline : splipy.Curve
            Curve
        t : np.array
            Parametes to evaluate
        order : int, optional
            Extrapolation order Taylor expansion, by default 2
        Returns
        -------
        np.array
            Evaluated points
        """
        n = len(t)

        if type(self.spline) in [BSpline.Curve, NURBS.Curve]:
            res = np.zeros((n, np.array(self.spline.ctrlpts).shape[1]))
        else:
            res = np.zeros((n, self.spline.controlpoints.shape[1]))
        for i in range(n):
            res[i, :] = self._evaluate_extra_single(t[i], order=order, d=d)
        return res

    def split(self, t):
        """Splits the spline at parameter t.
        Parameters
        ----------
        self.spline : splipy.Curve
            Curve
        t : float
            Parameter to split
        Returns
        -------
        splipy.Curve
            Splitted curve
        """
        return split_curve(self.spline, t)

    @property
    def bbox(self):
        return self.spline.bbox
