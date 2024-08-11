import numpy as np
from panav.environment.utils import box_2d_center, multi_tunnel_wall, peripheral_start_goals
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
    def __init__(self,n_tunnel, tunnel_width, limits=[(-10, 10), (-10, 10)], 
                 N_agent=6,
                 wallthickness = 5.0):
        super().__init__(limits, N_agent)
        
        y_min,y_max = min(limits[1]),max(limits[1])
        obstacles = multi_tunnel_wall(n_tunnel,tunnel_width,y_min,y_max,wall_thickness=wallthickness)

        self.obstacles = obstacles

        self.calc_start_goal_regions()
        
class WareHouse(DefaultEmtpyEnv):

    def __init__(self, limits=[(-10,10),(-10,10)], N_agent=6,
                 shelf_region_x_limit = [-5.0,5.0],shelf_region_y_limit = [-5.0,5.0],
                 obs_x_margin = 3 * 0.5,obs_y_margin = 3 * 0.5,
                 n_col = 4,n_row = 4,
                 corner_padding_x = 2.0,corner_padding_y = 2.0,bloating_r = 0.5):
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

        self.starts, self.goals = peripheral_start_goals(limits,corner_padding_x,corner_padding_y,bloating_r,N_agent)
        
        self.calc_start_goal_regions()


    def get_tunnels(self,separate_row_col = False):

        col_tunnels = [[] for _ in range(self.n_row-1)]
        row_tunnels = [[] for _ in range(self.n_row)] 


        for row in range(self.n_row-1):
            for col in range(self.n_col):
                f1_lo = self.low_left_corner + np.array([col * (self.w + self.x_margin),self.h + row * (self.y_margin+self.h)])
                f1_hi = f1_lo + np.array([0,self.y_margin])
                f1 = [f1_lo,f1_hi]


                f2_lo = f1_lo + np.array([self.w,0])
                f2_hi = f1_hi + np.array([self.w,0])
                f2 = [f2_lo,f2_hi]

                n1 = f2_lo - f1_lo
                n2 = -n1
                col_tunnels[row].append(Tunnel(f1,n1,f2,n2,end_point_buffer=0.1))

        for row in range(self.n_row):
            for col in range(self.n_col-1):
                f1_l = self.low_left_corner + np.array([self.w + col * (self.w + self.x_margin), row * (self.y_margin+self.h)])
                f2_l = f1_l + np.array([0,self.h])
            

                f1_r = f1_l + np.array([self.x_margin,0])
                f2_r = f2_l + np.array([self.x_margin,0])

                f1 = [f1_l,f1_r]
                f2 = [f2_l,f2_r]

                n1 = f2_l - f1_l
                n2 = -n1

                row_tunnels[row].append(Tunnel(f1,n1,f2,n2,end_point_buffer=0.1))

        if separate_row_col:
            return col_tunnels,row_tunnels
        else:
            col_tunnels = [t for tun in col_tunnels for t in tun]
            row_tunnels = [t for tun in row_tunnels for t in tun]
            return col_tunnels+row_tunnels
        
import networkx as nx
from panav.tunnels import Tunnel
from panav.environment.utils import horizontal_multi_tunnel_wall,vertical_multi_tunnel_wall
from shapely import Point
class Room(DefaultEmtpyEnv):
    def __init__(self, 
                n_col = 8,
                n_row = 8,
                cell_width = 12.0,
                cell_height = 12.0,
                wall_thickness = 2.0,
                gap_width = 3.0,
                dist_2_tunnel = 3.0,
                dist_2_obs = 2.0,
                dist_2_neighbor = 3.0,
                start_goal_dist = 10.0,
                N_agent=100,
                starts_in = None,
                goals_in = None
                ):
        super().__init__()
        
        self.N_agent = N_agent
        self.n_col = n_col
        self.n_row = n_row
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.wall_thickness = wall_thickness
        self.gap_width = gap_width

        total_width = n_col * (cell_width + wall_thickness) - wall_thickness
        total_height = n_row * (cell_height + wall_thickness) - wall_thickness

        self.limits = [(-total_width/2,total_width/2),(-total_height/2,total_height/2)]

        self.total_width = total_width
        self.total_height = total_height

        self.tunnels = []
        self.obstacles = []

        self.dist_2_tunnel = dist_2_tunnel
        self.dist_2_obs = dist_2_obs
        self.dist_2_neighbor = dist_2_neighbor
        self.start_goal_dist = start_goal_dist

        self.__construct_room__(starts=starts_in,goals=goals_in)
        self.calc_start_goal_regions()

    def remove_random_edge(G_in,drop_rate=0.2,max_iter = 50): 
        for _ in range(max_iter):
            G = nx.Graph(G_in)
            n = G.number_of_edges()
            es = [e for e in G.edges]
            to_remove = np.random.choice(np.arange(len(es)),int(n * drop_rate),replace=False)

            for i in to_remove:
                G.remove_edge(*es[i])

            if nx.is_connected(G):
                return G
        return G_in
    
    def point_feasible(self,tunnels, test_pt,dist_2_tunnel,dist_2_obs,dist_2_neighbor):
        '''
            Determine if the test_pt is free from any obstacle or tunnels
        '''
        x_limits = self.limits[0]
        y_limits = self.limits[1]
        if any([test_pt[0] < x_limits[0] + dist_2_obs,
                test_pt[0] > x_limits[1] - dist_2_obs,
                test_pt[1] < y_limits[0] + dist_2_obs,
                test_pt[1] > y_limits[1] - dist_2_obs,
        ]):
            return False
                

        pt = Point(test_pt)
        for obs in self.obstacles:
            if obs.verts.distance(pt) < dist_2_obs:
                return False
                

        for tun in tunnels:
            if tun.region.distance(pt) < dist_2_tunnel:
                return False
            
        for s in self.starts + self.goals:
            if np.linalg.norm(s-test_pt)<dist_2_neighbor:
                return False
        return True

    def random_feasible_point(self,tunnels,
                              dist_2_tunnel,
                              dist_2_obs,
                              dist_2_neighbor,
                              max_iter = 500):
        for _ in range(max_iter):
            test_pt = np.random.rand(2) * np.array([self.total_width,self.total_height]) \
                                        + np.array([self.limits[0][0],self.limits[1][0]])
            if self.point_feasible(tunnels,test_pt,dist_2_tunnel,dist_2_obs,dist_2_neighbor):
                return test_pt
            
        return None
    
    def random_start_goal(self,dist_2_tunnel,dist_2_obs,dist_2_neighbor,start_goal_dist):
        max_iter = 500

        for _ in range(max_iter):
            candidate_start = self.random_feasible_point(self.tunnels,dist_2_tunnel,dist_2_obs,dist_2_neighbor)

            if self.point_feasible(self.tunnels,candidate_start,dist_2_tunnel,dist_2_obs,dist_2_neighbor):
                rand_theta = 2 * np.pi * np.random.rand()
                candidate_goal = candidate_start + start_goal_dist * np.array([np.cos(rand_theta),np.sin(rand_theta)])
            
                # candidate_goal = self.random_feasible_point(self.tunnels,dist_2_tunnel,dist_2_obs,dist_2_neighbor)
                # if np.linalg.norm(candidate_start-candidate_goal)>start_goal_dist:
            
                if self.point_feasible(self.tunnels,candidate_goal,dist_2_tunnel,dist_2_obs,dist_2_neighbor):
                    return candidate_start,candidate_goal
        return None

    
    def __construct_room__(self,edge_drop_rate=0.1,starts = None, goals = None):
        np.random.seed(8)
        G = nx.grid_2d_graph(self.n_col,self.n_row)
        G = Room.remove_random_edge(G,drop_rate=edge_drop_rate)


        pos = {}
        origin =  - np.array([self.total_width,self.total_height])/2
        for i,j in G.nodes:
            pos[(i,j)] = np.array([self.cell_height/2 + j * (self.cell_height+self.wall_thickness),
                                   self.cell_width/2 + i * (self.cell_width + self.wall_thickness) ]) + origin


        col_tunnels = []
        # Add horizontal tunnels
        for i in range(self.n_row-1):
            # Add horizontal tunnels at the ith row
            y = self.cell_height + self.wall_thickness/2 +  i * (self.cell_height+self.wall_thickness) +  origin[1]

            gap_x_locs = []
            for j in range(self.n_col):
                e = ((i,j),(i+1,j))
                if e in G.edges:
                    alpha = np.random.rand() # alpha can be randomly chosen later to be any number between 0 and 1. The random seed is fixed, though, so the graph layout won't change even if the algorithm is called multiple times.
                    gap_x = j * (self.cell_width+self.wall_thickness) + self.gap_width/2 * alpha + (self.cell_width-self.gap_width/2)*(1-alpha) + origin[0]
                    gap_x_locs.append(gap_x)

                    f1_l = np.array([gap_x - self.gap_width/2, y - self.wall_thickness/2])
                    f1_r = f1_l + np.array([self.gap_width,0])
                    f1 = [f1_l,f1_r]
                    
                    f2_l = f1_l + np.array([0,self.wall_thickness])
                    f2_r = f1_r + np.array([0,self.wall_thickness])
                    f2 = [f2_l,f2_r]

                    n1 = f2_l - f1_l
                    n2 = -n1
                    col_tunnels.append(Tunnel(f1,n1,f2,n2,end_point_buffer=0.1))

            self.obstacles += horizontal_multi_tunnel_wall(*self.limits[0],gap_x_locs,self.gap_width,self.wall_thickness,y)

        row_tunnels = []
        # Add vertical tunnels
        for j in range(self.n_col-1):
            # Add vertical tunnels at the ith row
            x = self.cell_width + self.wall_thickness/2 +  j * (self.cell_width+self.wall_thickness) +  origin[1]

            gap_y_locs = []
            for i in range(self.n_row):
                e = ((i,j),(i,j+1))
                if e in G.edges:
                    # alpha can be randomly chosen later to be any number between 0 and 1. The random seed is fixed, though, so the graph layout won't change even if the algorithm is called multiple times.
                    alpha = np.random.rand()
                    gap_y = i * (self.cell_height+self.wall_thickness) + self.gap_width/2 * alpha + (self.cell_height-self.gap_width/2)*(1-alpha) + origin[1]
                    gap_y_locs.append(gap_y)

                    f1_lo = np.array([x - self.wall_thickness/2, gap_y - self.gap_width/2])
                    f1_hi = f1_lo + np.array([0,self.gap_width])
                    f1 = [f1_lo,f1_hi]
                    
                    f2_lo = f1_lo + np.array([self.wall_thickness,0])
                    f2_hi = f1_hi + np.array([self.wall_thickness,0])
                    f2 = [f2_lo,f2_hi]

                    n1 = f2_lo - f1_lo
                    n2 = -n1
                    row_tunnels.append(Tunnel(f1,n1,f2,n2,end_point_buffer=0.1))

            self.obstacles += vertical_multi_tunnel_wall(*self.limits[1],gap_y_locs,self.gap_width,self.wall_thickness,x)

            self.tunnels = col_tunnels +  row_tunnels

            if starts is None or goals is None:    
                self.starts = []
                self.goals = []
                for _ in range(self.N_agent):
                    start,goal = self.random_start_goal(self.dist_2_tunnel,
                                                        self.dist_2_obs,
                                                        self.dist_2_neighbor,
                                                        self.start_goal_dist)
                    self.starts.append(start)
                    self.goals.append(goal)
            else:
                self.starts = starts
                self.goals = goals

            self.starts = np.array(self.starts)
            self.goals = np.array(self.goals)
