import numpy as np
from .spline import Spline
from scipy.stats import qmc
from scipy.optimize import Bounds
from sklearn.cluster import DBSCAN
# from geomdl.visualization import VisMPL as vis
import numpy as np
from scipy.optimize import minimize, root, differential_evolution
from geomdl import BSpline, knotvector, NURBS
from geomdl.fitting import interpolate_curve, approximate_curve
from scipy.optimize import Bounds
from copy import copy


def check_overlap_bounding_box(crv1: Spline, crv2: Spline):
    bb1 = np.array(crv1.bbox)
    bb2 = np.array(crv2.bbox)

    if bb1[0, 0] > bb2[1, 0] or bb1[1, 0] < bb2[0, 0]:
        return False
    return bb1[0, 1] <= bb2[1, 1] and bb1[1, 1] >= bb2[0, 1]


def plot_bounding_box(ax, bb, **kwargs):
    ax.plot(
        [bb[0, 0], bb[1, 0], bb[1, 0], bb[0, 0], bb[0, 0]],
        [bb[0, 1], bb[0, 1], bb[1, 1], bb[1, 1], bb[0, 1]],
        **kwargs
    )


def _intersection(crv1, crv2, list_pts=None, tol_abs=1e-5):
    """Private function to calculate the intersection points of two curves."""

    # Casteljau's algorithm for intersection of two parametric curves
    if list_pts is None:
        list_pts = []

    bb1 = np.array(crv1.bbox)
    bb2 = np.array(crv2.bbox)

    # plot_bounding_box(ax, bb1, c=f"C{i_level}", ls="--")
    # plot_bounding_box(ax, bb2, c=f"C{i_level}", lw=1)

    centroid1 = np.mean(bb1, axis=0)
    centroid2 = np.mean(bb2, axis=0)
    distance = np.linalg.norm(centroid1 - centroid2)

    if distance < tol_abs:
        # print("success")
        # mean of centroid1 and centroid2
        pt = (centroid1 + centroid2) / 2
        list_pts.append(pt)
        return None
        # return list_pts

    if not check_overlap_bounding_box(crv1, crv2):
        return None
        # return None

    # split curves
    crv1_1, crv1_2 = crv1.split(0.5)
    crv2_1, crv2_2 = crv2.split(0.5)

    # check if bounding boxes overlap
    if check_overlap_bounding_box(crv1_1, crv2_1):
        _intersection(crv1_1, crv2_1, list_pts)
    if check_overlap_bounding_box(crv1_1, crv2_2):
        _intersection(crv1_1, crv2_2, list_pts)
    if check_overlap_bounding_box(crv1_2, crv2_1):
        _intersection(crv1_2, crv2_1, list_pts)
    if check_overlap_bounding_box(crv1_2, crv2_2):
        _intersection(crv1_2, crv2_2, list_pts)


def intersection(crv1, crv2, tol_abs=1e-5, clean=True):
    """Calculates the intersection points of two curves using de Casteljau's
    algorithm.
    References
    ----------
    [1] https://pomax.github.io/bezierinfo/index.html#curveintersection
    """

    list_pts = []
    _intersection(crv1, crv2, list_pts, tol_abs)
    pts = np.array(list_pts)

    return cluster_close_points(pts, epsilon=tol_abs) if clean else pts


def cluster_close_points(points, epsilon):
    """Clusters points that are close to each other.
    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """

    # Initialize and fit the DBSCAN clustering model
    dbscan = DBSCAN(eps=epsilon, min_samples=1)
    dbscan.fit(points)

    # Find unique cluster labels assigned by DBSCAN
    unique_labels = np.unique(dbscan.labels_)

    # Reduce points in each cluster to their mean value
    reduced_points = []
    for label in unique_labels:
        cluster_points = points[dbscan.labels_ == label]
        mean_point = np.mean(cluster_points, axis=0)
        reduced_points.append(mean_point)

    # Convert the reduced points to a NumPy array
    return np.array(reduced_points)


