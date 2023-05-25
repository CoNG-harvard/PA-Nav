import numpy as np

from shapely import Polygon,LineString
from shapely.ops import nearest_points
from shapely.affinity import translate,rotate

from scipy.spatial import ConvexHull

import cvxpy as cp

def detect_tunnels(env,bloating_r):
    obstacles = env.obstacles
    tunnels = []
    for i in range(len(obstacles)):
        for j in range(i+1,len(obstacles)):
            O1,O2 = obstacles[i],obstacles[j]
        
            tun = bin_search_tunnel(O1,O2,bloating_r)
            if tun is not None:
                tunnels.append(tun)
    return tunnels

def bin_search_tunnel(O1,O2,bloating_r,extension_direction=None,ext_l0=1.0,bin_search_esp = 0.05):
    def search(extension_direction):    
        
        # Binary search for the tunnel region
        left, right = 0, ext_l0*2
        expand = True
        while right-left>bin_search_esp:

            shifted_neck,dist,projs = extension_sticks(O1,O2,
                                           np.sign(extension_direction)*(left+right)/2) 

            # print('dist',dist,'shifted_neck',shifted_neck)
            underhit = dist <= 4*bloating_r

            if underhit:
                if expand: right *=2
                else: left = (left+right)/2
            else:
                right = (left+right)/2
                expand = False

        return shifted_neck,projs,dist
    
    pts = nearest_points(O2.vertices(),O1.vertices())
    obs_dist = pts[0].distance(pts[1])
    if  obs_dist>4*bloating_r or obs_dist==0: # If the two obstacles are too far apart or they intersect, there is no tunnel between them.
        return None

    if extension_direction is None: 
        (p1,_,d1), (p2,_,d2) = search(1),search(-1)
        verts = np.array(p1+p2)
        return Polygon(verts[ConvexHull(verts).vertices,:])
    else: return search(extension_direction)

def extension_sticks(O1,O2,ext_l):
    pts = nearest_points(O2.vertices(),O1.vertices())

    neck = LineString(pts)
    perp = rotate(neck,90)

    direction = np.array(perp.coords[:])
    direction = direction[1]-direction[0]

    direction /= np.linalg.norm(direction)

    shifted_neck = translate(neck, *ext_l*direction)

    shift_origin = np.mean(shifted_neck.coords,axis= 0)

    projs = []
    dists = []
    for O in [O1,O2]:
        A,b = O.A,O.b

        x_proj = cp.Variable(shift_origin.shape)
        constraints = [(x_proj - shift_origin) @ direction ==0,
                      A @ x_proj<=b]
        prob = cp.Problem(cp.Minimize(cp.norm(x_proj-shift_origin)),constraints)

        dist = prob.solve()

        projs.append(x_proj.value)
        dists.append(dist)
    
    if len(projs)==0:
        return shifted_neck.coords[:], np.inf,[]
    
    # print(projs,dists,shift_origin)
    # short_end = projs[np.argmin(dists)]
    # opposite_end = shift_origin - (short_end-shift_origin)
    # return shifted_neck.coords[:],[short_end,opposite_end], np.min(dists)

    return shifted_neck.coords[:], np.sum(dists), projs

