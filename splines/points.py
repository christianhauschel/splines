from copy import copy
import numpy as np
from scipy.spatial import ConvexHull


def max_distant_pair(points):
    CH = ConvexHull(points)
    N = CH.vertices.size
    K = 1

    while K < N:
        cur_area = area(
            points[CH.vertices[0]], points[CH.vertices[N - 1]], points[CH.vertices[K]]
        )
        nxt_area = area(
            points[CH.vertices[0]],
            points[CH.vertices[N - 1]],
            points[CH.vertices[K + 1]],
        )
        if cur_area > nxt_area:
            break
        K += 1

    P = 0
    Q = K
    answer = distance(points[CH.vertices[P]], points[CH.vertices[Q]])
    pts_idx = (CH.vertices[P], CH.vertices[Q])
    while P <= K and Q < N:
        while Q < N:
            cur_area = area(
                points[CH.vertices[P]],
                points[CH.vertices[P + 1]],
                points[CH.vertices[Q]],
            )
            nxt_area = area(
                points[CH.vertices[P]],
                points[CH.vertices[P + 1]],
                points[CH.vertices[(Q + 1) % N]],
            )
            if cur_area > nxt_area:
                break
            Q += 1
            if distance(points[CH.vertices[P]], points[CH.vertices[Q % N]]) > answer:
                answer = distance(points[CH.vertices[P]], points[CH.vertices[Q % N]])
                pts_idx = (CH.vertices[P], CH.vertices[Q % N])
        P += 1
    return answer, pts_idx


def area(p1, p2, p3):
    return 0.5 * np.abs(np.cross(p2 - p1, p3 - p1))


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
