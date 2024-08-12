import shapely
import numpy as np
from numpy import linalg as la



def in_env_boundary(env,x,bloating_r):
    return env.limits[0][0]+bloating_r<=x[0]<=env.limits[0][1]-bloating_r\
          and env.limits[1][0]+bloating_r<=x[1]<=env.limits[1][1]-bloating_r
    
def has_line_collision(env,x1,x2,bloating_r):
    '''
        Output: whether the bloated line segment x1-x2 with bloating radius = bloating_r intersects with any of the obstacles in env.
    '''
    if in_env_boundary(env,x1,bloating_r) and in_env_boundary(env,x2,bloating_r):
        l = shapely.LineString([x1,x2]) 
        for o in env.obstacles:
            if l.distance(o.vertices())<bloating_r:
                return True
        return False
    else:
        Warning("The test locations may not be within the limits of the environment.")
        return True 

def has_point_collision(env,x,bloating_r):
    '''
        Output: whether putting an agent with bloating radius=bloating_r at location x in env induces collision.
    '''
    if in_env_boundary(env,x,bloating_r):
        for o in env.obstacles:           
            if shapely.Point(x).distance(o.vertices())<bloating_r:
                return True
        return False
    else:
        Warning("The test location x={} is not within the limits of the environment.".format(x))
        return True 

def binary_line_search(env,x_start,x_target,bloating_r,
                       eta = None,eps = 1e-5):
    '''
        A heuristic to decide a new extension point.
        Output: min_x ||x-x_target|| such that 
               0) x lies on x_start-x_target
               1) has_line_collision(env,x_start,x,bloating_r) is False
               2) If eta is not None, then also ||x-x_start||<=eta. 
    '''
    
    
    if eta is not None:
        direction = x_target-x_start
        x_target = x_start + eta * direction/la.norm(direction)

    lo = 0
    hi = 1
    while hi-lo>eps:
        mid = (hi+lo)/2
        # print(hi,lo,mid)
        x_cur = x_start * (1-mid) + x_target * mid
        if has_line_collision(env,x_start,x_cur,bloating_r):
            # Shrink the line segment
            hi = mid
        else:
            # Expand the line segment
            lo = mid

    return x_cur

def uniform_rand_loc(env):
    return np.random.uniform(low = [env.limits[0][0],env.limits[1][0]],
                    high = [env.limits[0][1],env.limits[1][1]])
