import numpy as np
from panav.environment.utils import box_2d_center,multi_tunnel_wall


class NavigationEnv:
    def __init__(self, limits=[], obstacles=[],starts=[],goals=[]):

        self.limits = limits # Boundary limits at each axis. limits[0]-> x axis, limits[1]-> y axis, limits[2]-> z axis.
        # obstacles are Region class objects.
        self.obstacles = obstacles
        
        self.starts = starts
        self.goals = goals

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