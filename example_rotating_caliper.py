# %%

import numpy as np
from scipy.spatial import ConvexHull
from splines import max_distant_pair

try:
    import proplot as pplt
except ImportError:
    import matplotlib.pyplot as pplt

for i in range(20):

    points = np.random.rand(10, 2)

    # from airfoil import Airfoil 
    # # generate random odd int 
    # int_odd = np.random.randint(10, 101) * 2 + 1
    # # random naca number 
    # naca_id = np.random.randint(1000, 10000)
    # af = Airfoil.naca(f"{naca_id:04d}", n=int_odd)
    # points = af.data

    max_dist, pts_idx = max_distant_pair(points)
    print("Maximum distance:", max_dist, "Points:" , points[pts_idx[0]], points[pts_idx[1]])

    p1 = points[pts_idx[0]]
    p2 = points[pts_idx[1]]


    fig, ax = pplt.subplots(figsize=(4,4))
    ax.plot(points[:,0], points[:,1], 'o', color='black', markersize=2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')
    fig.show


# %%