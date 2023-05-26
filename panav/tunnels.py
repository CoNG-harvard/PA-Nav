import numpy as np

from shapely import Polygon,LineString
from shapely.ops import nearest_points
from shapely.affinity import translate,rotate

from scipy.spatial import ConvexHull

import cvxpy as cp

class Tunnel:
    def __init__(self,face1,n1,face2,n2):
        '''
            face1: [point1,point2], a list of two vectors. One face of the tunnel.
            face2: same format as face 1.

            n1: the normal vector of face1 pointing inside the tunnel.
            n2: the normal vector of face2 pointing inside the tunnel.
        '''
        self.faces = [face1,face2]
        self.perps = [n1,n2]
        verts = np.array(face1+face2)
        self.region = Polygon(verts[ConvexHull(verts).vertices,:])

        self.waiting = {}
        self.passing = {}
        for i in range(len(self.faces)):
            for j in range(i+1,len(self.faces)):
                self.waiting[(j,i)]= []
                self.waiting[(i,j)]= []
                self.passing[(j,i)]= []
                self.passing[(i,j)]= []

def get_entry_exit(tun,x):
    '''
        Return trajectory x's entry and exit points of the tunnel.

        Each element in entry/exit list: (fid,i,x[:,i])

            fid: the id of the face to be crossed.
            i: the index of the entry/exit point within trajectory x.
            x[:,i]: the entry/exit point location.
    '''
    
    face_lines = [LineString(f) for f in tun.faces]

    entry = []
    exit = []

    for i in range(x.shape[-1]-1):
        seg = LineString((x[:,i],x[:,i+1]))
        for fid, fl in enumerate(face_lines):
            if seg.intersects(fl):
                ent_sgn = np.sign(tun.perps[fid].dot(x[:,i+1]-x[:,i]))
                # print(fid,ent_sgn)
                if ent_sgn == 1:
                    entry.append((fid,i,x[:,i]))
                elif ent_sgn == -1:
                    exit.append((fid,i+1,x[:,i+1]))

    return entry,exit

def detect_tunnels(env,bloating_r):
    obstacles = env.obstacles
    tunnels = []
    for i in range(len(obstacles)):
        for j in range(i+1,len(obstacles)):
            O1,O2 = obstacles[i],obstacles[j]
        
            tun = bin_search_tunnel(O1,O2,bloating_r)
            if tun is not None:
                (f1,d1,n1),(f2,d2,n2) = tun
                tunnels.append(Tunnel(f1,n1,f2,n2))
    return tunnels

def bin_search_tunnel(O1,O2,bloating_r,extension_direction=None,ext_l0=1.0,bin_search_esp = 0.05):
    '''
        Output: 

    '''
    def search(extension_direction):
        '''
            Output: 
                shifted_neck: the line segment defining the entrance and exit of the tunnel.
                
                dist: the extended distance to O1,O2, along the direction of shifted_neck.
                
                normal_vec: the vector perpendicular to shifted_neck, pointing into the tunnel.
        '''    
        
        # Binary search for the tunnel region
        left, right = 0, ext_l0*2
        expand = True
        while right-left>bin_search_esp:

            shifted_neck,dist,normal_vec = extension_sticks(O1,O2,
                                           np.sign(extension_direction)*(left+right)/2) 

            # print('dist',dist,'shifted_neck',shifted_neck)
            underhit = dist <= 4*bloating_r

            if underhit:
                if expand: right *=2
                else: left = (left+right)/2
            else:
                right = (left+right)/2
                expand = False

        return shifted_neck,dist,normal_vec 
    
    pts = nearest_points(O2.vertices(),O1.vertices())
    obs_dist = pts[0].distance(pts[1])
    if  obs_dist>4*bloating_r or obs_dist==0: # If the two obstacles are too far apart or they intersect, there is no tunnel between them.
        return None

    if extension_direction is None: 
        return search(1),search(-1)
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

    normal_vec = -np.sign(ext_l)*direction
    return shifted_neck.coords[:], np.sum(dists), normal_vec

