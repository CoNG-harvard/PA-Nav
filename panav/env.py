import numpy as np
from scipy.spatial import ConvexHull
import pypoman as ppm
import polytope as pc
import shapely
from shapely import affinity, Polygon
import cvxpy as cp

from panav.util import unit_cube
from polytope import qhull



class NavigationEnv:
    def __init__(self, limits=[], obstacles=[],starts=[],goals=[]):

        self.limits = limits # Boundary limits at each axis. limits[0]-> x axis, limits[1]-> y axis, limits[2]-> z axis.
        # obstacles are Region class objects.
        self.obstacles = obstacles
        
        # starts and goals are Region class objects.
        self.starts = starts
        self.goals = goals       
        

class Region:
    '''
        Regions are always convex polytopes. Visualization code based on the S2M2 repo.
    '''
    def __init__(self,A,b):
        self.A = A
        self.b = b
    
    def vertices(self):
        '''
            Compute the vertices of a general polytope.
        '''
        verts = np.array(ppm.duality.compute_polytope_vertices(self.A, self.b))
        return Polygon(verts[ConvexHull(verts).vertices,:])
    
    def centroid(self):
        return self.vertices().centroid

    def project(self, x):
        '''
            Project a point x onto the region.
        '''
        x_proj = cp.Variable(x.shape)
        prob = cp.Problem(cp.Minimize(cp.norm(x_proj-x)),\
                          [self.A @ x_proj - self.b<=0])
        prob.solve()
        return x_proj.value

class PolygonRegion(Region):
    def __init__(self,vertices):
        self.verts = np.array(vertices) # verts.shape = (n,dim)
        self.verts = self.verts[ConvexHull(self.verts).vertices,:]
        self.poly = pc.qhull(self.verts)

        self.A, self.b = self.poly.A, self.poly.b

    def vertices(self):
        return Polygon(self.verts)
        
class Box2DRegion(Region):
    
    def __init__(self, xlim, ylim):
        self.xlim =  xlim
        self.ylim = ylim
        self.poly = pc.box2poly([xlim,ylim])
        
        self.A, self.b = self.poly.A, self.poly.b
        
    def vertices(self):
        return shapely.geometry.box(self.xlim[0],self.ylim[0],
                                    self.xlim[1],self.ylim[1])

def box_2d_center(center,side):
    '''
        Create a 2D box with specified center coordinates and side lengths.
    '''
    lb = center-side/2
    ub = center+side/2
    return Box2DRegion((lb[0],ub[0]),(lb[1],ub[1]))

def trajectory_to_temp_obstacles(t,xs,bloating_r):
    # Convert the agent's path into temporary obstacle.
    temp_obstacles = []
    for k in range(xs.shape[-1]-1):
        temp_obstacles.append(([t[k],t[k+1]], 
                               line_seg_to_obstacle(xs[:,k],xs[:,k+1],bloating_r)))
    return temp_obstacles

def line_seg_to_obstacle(x1,x2,bloating_r):
    '''
        Convert a line segment x1-x2 to a box obstacle with width 2*bloating_r
    '''
    b = shapely.geometry.LineString([x1,x2])
    if b.length == 0:
        return box_2d_center(x1,2*bloating_r)
    else:
        b_p = affinity.scale(affinity.rotate(b,90),
                     xfact = 2*bloating_r/b.length,
                     yfact = 2*bloating_r/b.length,
                  )

        d = (x2-x1)*(1/2+bloating_r/b.length)

        side_1 = affinity.translate(b_p,*d)
        side_2 = affinity.translate(b_p,*(-d))
        
        return PolygonRegion([*side_1.coords,*side_2.coords])

def wp_to_tube_obstacle(t1,t2,p1,p2,bloating_r):
    '''
        Convert a timed line segment (t1,p1)-(t2,p2) to a space-time tube obstacle with bloating radius bloating_r. 
        Output: (Ap,bp) characterizing the polytopic space-time tube obstacle.
    '''
    p1,p2 = np.array(p1),np.array(p2)
    if len(p1.shape) == 0:
        d = 1
    else:
        d = p1.shape[0]
    
    tube_vertices = [np.hstack([t,p+1.0*bloating_r*unit_vec]) 
                     for t,p in zip([t1,t2],[p1,p2]) 
                     for unit_vec in unit_cube(d)]
    poly = qhull(np.vstack(tube_vertices))

    Ap,bp = poly.A,poly.b
    return Ap,bp


# Convert the agent's path space-time tube obstacles.
def trajectory_to_tube_obstacles(times,xs,bloating_r):
    tube_obs = []
    for k in range(xs.shape[-1]-1):
        tube_obs.append(wp_to_tube_obstacle(times[k],times[k+1],
                                            xs[:,k],xs[:,k+1],bloating_r))
    return tube_obs
