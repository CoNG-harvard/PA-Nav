import numpy as np
from panav.env import box_2d_center

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

        self.end_points = [np.mean(face,axis=0)- p*0.1 for face,p in zip(self.faces,self.perps)] # The end points of an hard edge. Set it to be slightly outside of the tunnel.
        self.end_regions = [box_2d_center(ep, side = 0.1) for ep in self.end_points] # The waiting region around end points. Used in path planning.
        
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
    
    def passable(self,ent_fid,ex_fid):
        return len(self.passing[(ex_fid,ent_fid)])==0 # No agent is passing the opposite direction

class TunnelPassingControl:
    def __init__(self,env,agent_plans,bloating_r):
        self.tunnels = detect_tunnels(env,bloating_r)
        self.agents = set(np.arange(len(agent_plans)))
        self.passage = {a:[] for a in self.agents}
        
        for a,x in zip(self.agents, agent_plans):
            for tun_id, tunnel in enumerate(self.tunnels):
                ent,ex = get_entry_exit(tunnel,x)
                
                if ent is not None and ex is not None:
                    self.passage[a].append((tun_id,*ent,*ex))
        
        self.next_passage = {}
        for a in self.agents:
            self.next_passage[a] = None if len(self.passage[a])==0\
                                    else self.passage[a].pop(0)

    def update_passing_state(self,agent,agent_loc,
                             waiting_radius, exit_radius):
        
        next_pass = self.next_passage[agent]
        
        if next_pass is not None:
            tun_id, ent_fid,ent_wpid,ent_loc,ex_fid,ex_wpid,ex_loc \
            = next_pass
            
            tunnel = self.tunnels[tun_id]
            
            # See if the agent has come close to the tunnel entrance.
            if np.linalg.norm(ent_loc-agent_loc)<=waiting_radius\
            and agent not in tunnel.passing[(ent_fid,ex_fid)]:
                if agent not in tunnel.waiting[(ent_fid,ex_fid)]:
                        tunnel.waiting[(ent_fid,ex_fid)].append(agent)
                        # Add the agent to the waiting list

            # print('passing',tunnel.passing[(ent_fid,ex_fid)],tunnel.passing[(ex_fid,ent_fid)])
            if agent in tunnel.waiting[(ent_fid,ex_fid)]:
                if agent == tunnel.waiting[(ent_fid,ex_fid)][0]\
                    and tunnel.passable(ex_fid,ent_fid): 
                    tunnel.waiting[(ent_fid,ex_fid)].remove(agent)
                    tunnel.passing[(ent_fid,ex_fid)].append(agent)
                else:
                    return 'wait'
            
            # See if the agent is at an exit point
            if np.linalg.norm(ex_loc-agent_loc)<=exit_radius:
                
                tunnel.passing[(ent_fid,ex_fid)].remove(agent)

                if len(self.passage[agent])>0:
                    self.next_passage[agent] = self.passage[agent].pop(0)
                else:
                    self.next_passage[agent] = None
                
        return 'pass'
def get_entry_exit(tun,x):
    '''
        Return trajectory x's entry and exit points of the tunnel.

        Each element in entry/exit list: (fid,i,x[:,i])

            fid: the id of the face to be crossed.
            i: the index of the entry/exit point within trajectory x.
            x[:,i]: the entry/exit point location.
    '''
    
    face_lines = [LineString(f) for f in tun.faces]

    entry = None
    exit = None

    for i in range(x.shape[-1]-1):
        seg = LineString((x[:,i],x[:,i+1]))
       
        for fid, fl in enumerate(face_lines):
            if seg.intersects(fl):
                ent_sgn = np.sign(tun.perps[fid].dot(x[:,i+1]-x[:,i]))
                
                if ent_sgn == 1:
                    entry = (fid,i,x[:,i])
                elif ent_sgn == -1:
                    exit = (fid,i+1,x[:,i+1])

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

