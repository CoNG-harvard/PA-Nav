import numpy as np
from scipy.spatial import ConvexHull
import pypoman as ppm
import polytope as pc
import shapely
from shapely import affinity
import cvxpy as cp


class NavigationEnv:
    def __init__(self, limits, obstacles,starts,goals):

        self.limits = limits # Boundary limits at each axis. limits[0]-> x axis, limits[1]-> y axis, limits[2]-> z axis.
        # obstacles are Region class objects.
        self.obstacles = obstacles
        
        # starts and goals are shapely Polygon objects.
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
        return verts[ConvexHull(vert).vertices,:]

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
        return self.verts
        
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


def line_seg_to_obstacle(x1,x2,bloating_r):
    '''
        Convert a line segment x1-x2 to a box obstacle with width 2*bloating_r
    '''
    b = shapely.geometry.LineString([x1,x2])
    b_p = affinity.scale(affinity.rotate(b,90),
                 xfact = 2*bloating_r/b.length,
                 yfact = 2*bloating_r/b.length,
              )

    d = (x2-x1)*(1/2+bloating_r/b.length)

    side_1 = affinity.translate(b_p,*d)
    side_2 = affinity.translate(b_p,*(-d))
    
    return PolygonRegion([*side_1.coords,*side_2.coords])
