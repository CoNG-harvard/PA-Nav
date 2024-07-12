import numpy as np
import shapely
from shapely import affinity
from polytope import qhull


from panav.util import unit_cube
from panav.environment.region import Box2DRegion, PolygonRegion

def box_2d_center(center,side):
    '''
        Create a 2D box with specified center coordinates and side lengths.
    '''
    lb = center-side/2
    ub = center+side/2
    return Box2DRegion((lb[0],ub[0]),(lb[1],ub[1]))
def approxCircle(c,R,N=50):
    # Approximate a circular obstacle with multiple boxes
    N = 50
    width = 2*R/N
    out = []
    for k in range(-N//2+1,N//2):
        box_center = np.array([c[0] + width * k, c[1]])
        sides = np.array([width, 2 * np.sqrt(R**2 - (width * (abs(k)+0.5))**2)])
        out.append(box_2d_center(box_center,sides))
    return out

def gate(x_loc,y_loc, width, y_lims,thickness = 2.0):
    # A gate is made of two walls touching the upper and lower limit of the environment
    # with an opening of certain width between them. The y-coordinate of the opening is determined by y_loc. 
    
    O1 = Box2DRegion((-thickness/2+x_loc,thickness/2+x_loc),(y_loc+width/2,y_lims[1]))
    O2 = Box2DRegion((-thickness/2+x_loc,thickness/2+x_loc),(y_lims[0],y_loc-width/2))
    return [O1,O2]

def multi_tunnel_wall(n_tunnel,tunnel_width,y_min,y_max,wall_thickness=5, wall_x_offset=0):
        w = tunnel_width # Tunnel width
        s = (y_max-y_min-w*n_tunnel)/(n_tunnel+1) # Spacing between tunnels

        d = wall_thickness # Thickness of the wall

        obstacles = []
        for i in range(n_tunnel+1):
            side = np.array([d,s])
            center = np.array([wall_x_offset, y_max-s/2-i*(s+w)])
            obstacles.append(box_2d_center(center,side))
        return obstacles

def horizontal_multi_tunnel_wall(x_min,x_max,gap_x_locs,tunnel_width,wall_thickness=5, wall_y_offset=0):
        d = wall_thickness # Thickness of the wall

        l = x_min
            
        obstacles = []

        for i in range(len(gap_x_locs)+1):
          
            if i == len(gap_x_locs):
                r = x_max
            else:
                r = gap_x_locs[i] - tunnel_width/2

            
            side = np.array([r-l, d])
            center = np.array([(l + r)/2, wall_y_offset])
            # print(side,center,l,r)
            obstacles.append(box_2d_center(center,side))
            
            l = r + tunnel_width

        return obstacles

def vertical_multi_tunnel_wall(y_min,y_max,gap_y_locs,tunnel_width,wall_thickness=5, wall_x_offset=0):
        d = wall_thickness # Thickness of the wall

        bottom = y_min
            
        obstacles = []

        for i in range(len(gap_y_locs)+1):
          
            if i == len(gap_y_locs):
                top = y_max
            else:
                top = gap_y_locs[i] - tunnel_width/2

            
            side = np.array([d,top-bottom])
            center = np.array([wall_x_offset,(top + bottom)/2])
            # print(side,center,l,r)
            obstacles.append(box_2d_center(center,side))
            
            bottom = top + tunnel_width

        return obstacles

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