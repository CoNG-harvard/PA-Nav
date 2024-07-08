import numpy as np
from panav.environment.utils import box_2d_center,multi_tunnel_wall
from panav.tunnels import Tunnel


class NavigationEnv:
    def __init__(self, limits=[], obstacles=[],starts=[],goals=[]):

        self.limits = limits # Boundary limits at each axis. limits[0]-> x axis, limits[1]-> y axis, limits[2]-> z axis.
        # obstacles are Region class objects.
        self.obstacles = obstacles
        
        self.starts = starts
        self.goals = goals

    def calc_start_goal_regions(self):
        # The following are for visualization only and not used in planning
        start_box_side = goal_box_side = 1.0
        self.start_regions = [box_2d_center(s,start_box_side) for s in self.starts]
        self.goal_regions = [box_2d_center(g,goal_box_side) for g in self.goals]

      
        
class DefaultEmtpyEnv(NavigationEnv):
    def __init__(self, limits=[(-10.0,10.0),(-10.0,10.0)], N_agent = 6):
       

        top, bottom = limits[1][1]-2.0,limits[1][0]+2.0

        start_x_offset = abs(limits[0][0]) * 0.7
        goal_x_offset = start_x_offset + 2.0

        if N_agent % 2 == 0:
            N1 = N2 = N_agent // 2
        else:
            N1 = N_agent // 2
            N2 = N1+1


        start_locs = np.vstack([
        np.vstack([np.ones(N1)*start_x_offset,np.linspace(top,bottom, N1)]).T,
        np.vstack([np.ones(N2)*(-start_x_offset),np.linspace(top,bottom, N2)]).T])

        goal_locs = np.vstack([
        np.vstack([np.ones(N1)*(-goal_x_offset),np.linspace(bottom,top, N1)]).T,
        np.vstack([np.ones(N2)*goal_x_offset,np.linspace(bottom,top, N2)]).T])

        super().__init__(limits, [], start_locs, goal_locs)

class MultiTunnelEnv(DefaultEmtpyEnv):
    def __init__(self,n_tunnel, tunnel_width, limits=[(-10, 10), (-10, 10)], N_agent=6,wallthickness = 5.0):
        super().__init__(limits, N_agent)
        
        y_min,y_max = min(limits[1]),max(limits[1])
        obstacles = multi_tunnel_wall(n_tunnel,tunnel_width,y_min,y_max,wall_thickness=wallthickness)

        self.obstacles = obstacles
        
class WareHouse(DefaultEmtpyEnv):

    def __init__(self, limits=[(-10,10),(-10,10)], N_agent=6,
                 shelf_region_x_limit = [-5.0,5.0],shelf_region_y_limit = [-5.0,5.0],
                 obs_x_margin = 3 * 0.5,obs_y_margin = 3 * 0.5,
                 n_col = 4,n_row = 4):
        super().__init__(limits, N_agent)
   
        w = (shelf_region_x_limit[1] - shelf_region_x_limit[0] - (n_col-1)*obs_x_margin )/n_col
        h = (shelf_region_y_limit[1] - shelf_region_y_limit[0] - (n_row-1)*obs_y_margin )/n_row

        lower_left_corner = np.array([shelf_region_x_limit[0],shelf_region_y_limit[0]])
        
        for i in range(n_col):
            for j in range(n_row):
                o = box_2d_center(lower_left_corner+np.array([w/2 + i * (w+obs_x_margin), h/2 + j * (h+obs_y_margin)]),np.array([w,h]))
                self.obstacles.append(o)
        
        self.w = w
        self.h = h
        self.low_left_corner = lower_left_corner
        self.x_margin = obs_x_margin
        self.y_margin = obs_y_margin
        self.n_col = n_col
        self.n_row = n_row

    def get_tunnels(self):

        col_tunnels = []
        row_tunnels = []

        col = 0 
        row = 0

        for col in range(self.n_col):
            for row in range(self.n_row-1):
                f1_lo = self.low_left_corner + np.array([col * (self.w + self.x_margin),self.h + row * (self.y_margin+self.h)])
                f1_hi = f1_lo + np.array([0,self.y_margin])
                f1 = [f1_lo,f1_hi]


                f2_lo = f1_lo + np.array([self.w,0])
                f2_hi = f1_hi + np.array([self.w,0])
                f2 = [f2_lo,f2_hi]

                n1 = f2_lo - f1_lo
                n2 = -n1

                col_tunnels.append(Tunnel(f1,n1,f2,n2,end_point_buffer=0.1))

        for col in range(self.n_col-1):
            for row in range(self.n_row):
                f1_l = self.low_left_corner + np.array([self.w + col * (self.w + self.x_margin), row * (self.y_margin+self.h)])
                f2_l = f1_l + np.array([0,self.h])
            

                f1_r = f1_l + np.array([self.x_margin,0])
                f2_r = f2_l + np.array([self.x_margin,0])

                f1 = [f1_l,f1_r]
                f2 = [f2_l,f2_r]

                n1 = f2_l - f1_l
                n2 = -n1

                row_tunnels.append(Tunnel(f1,n1,f2,n2,end_point_buffer=0.1))

        return col_tunnels+row_tunnels