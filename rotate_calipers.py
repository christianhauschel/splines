# %%

import numpy as np
from scipy.spatial import ConvexHull

def maximum_distance(points):
    CH = ConvexHull(points)
    N = CH.vertices.size
    K = 1

    while K < N:
        cur_area = area(points[CH.vertices[0]], points[CH.vertices[N - 1]], points[CH.vertices[K]])
        nxt_area = area(points[CH.vertices[0]], points[CH.vertices[N - 1]], points[CH.vertices[K + 1]])
        if cur_area > nxt_area:
            break
        K += 1
    
    P = 0
    Q = K
    answer = distance(points[CH.vertices[P]], points[CH.vertices[Q]])
    pts_idx = (CH.vertices[P], CH.vertices[Q])
    while P <= K and Q < N:
        while Q < N:
            cur_area = area(points[CH.vertices[P]], points[CH.vertices[P + 1]], points[CH.vertices[Q]])
            nxt_area = area(points[CH.vertices[P]], points[CH.vertices[P + 1]], points[CH.vertices[(Q + 1) % N]])
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
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

points = np.array([
    [0.05328671, 0.75379551],
    [0.22475366, 0.10027644],
    [0.27405434, 0.59329810],
    [0.35790505, 0.88768214],
    [0.25576468, 0.44421372],
    [0.90777934, 0.36292922],
])





for i in range(20):

    points = np.random.rand(10, 2)

    # from airfoil import Airfoil 
    # # generate random odd int 
    # int_odd = np.random.randint(10, 101) * 2 + 1
    # # random naca number 
    # naca_id = np.random.randint(1000, 10000)
    # af = Airfoil.naca(f"{naca_id:04d}", n=int_odd)
    # points = af.data

    max_dist, pts_idx = maximum_distance(points)
    print("Maximum distance:", max_dist, "Points:" , points[pts_idx[0]], points[pts_idx[1]])

    p1 = points[pts_idx[0]]
    p2 = points[pts_idx[1]]

    import proplot as pplt 

    fig, ax = pplt.subplots(aspect=1, figsize=(4,4))
    ax.plot(points[:,0], points[:,1], 'o', color='black', markersize=2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')
    fig.show


# %%