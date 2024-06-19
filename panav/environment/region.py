
import shapely
from shapely import Polygon
import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull
import polytope as pc
import pypoman as ppm


class Region:
    '''
        Regions are always convex polytopes. Visualization code based on the S2M2 repo.
    '''
    def __init__(self,A,b):
        self.A = A
        self.b = b
        verts = np.array(ppm.duality.compute_polytope_vertices(self.A, self.b))
        self.verts = Polygon(verts[ConvexHull(verts).vertices,:])
    
    def vertices(self):
        '''
            Return the vertices of a general polytope.
        '''
        return self.verts
    
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
        self.verts = Polygon(self.verts)
        

        self.A, self.b = self.poly.A, self.poly.b

        
class Box2DRegion(Region):
    
    def __init__(self, xlim, ylim):
        self.xlim =  xlim
        self.ylim = ylim
        self.poly = pc.box2poly([xlim,ylim])
        
        self.A, self.b = self.poly.A, self.poly.b
        self.verts = shapely.geometry.box(self.xlim[0],self.ylim[0],
                                    self.xlim[1],self.ylim[1])
    
    def project(self,x):
        xnew = np.array(x)
        
        xnew[0] = max(self.xlim[0],xnew[0])
        xnew[0] = min(self.xlim[1],xnew[0])
        
        xnew[1] = max(self.ylim[0],xnew[1])
        xnew[1] = min(self.ylim[1],xnew[1])
        
        return xnew