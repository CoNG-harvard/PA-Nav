import networkx as nx
import numpy as np
from time import time

from panav.SAMP.solvers import Path_Tracking


def TAHP(HG,vmax,bloating_r,TIMEOUT = 120):
    '''
    HG: a hybrid graph class object.
    
    Output: a conflict-free multi-agent PWL path, in the form of [[ts_i,xs_i] for i in angents]
    '''
    t0 = time()
    continuous_plans = []
    paths = traffic_aware_HG_plan(HG)
    for i,path in enumerate(paths):
        # print("Planning for {}/{}".format(i,len(paths)))
        start = HG.node_loc(path[0])
        goal = HG.node_loc(path[-1])
        
        milestones = np.array([HG.node_loc(u) for u in path[1:-1]]).T
        solver = Path_Tracking(HG.env,start,goal,
                            milestones=milestones,max_dev=0.2,vmax=vmax,bloating_r=bloating_r)

        p = solver.plan(obstacle_trajectories=continuous_plans)    
        
        if time()-t0>TIMEOUT: # Stop early if runtime exceeds TIMEOUT.
            return None
        
        if p is None: # The problem becomes infeasible.
            return None
        
        continuous_plans.append(p)
    return continuous_plans

def traffic_aware_HG_plan(HG,consider_soft_traffic=False):
    '''
        HG: a hybrid graph class object.
        consider_soft_traffic: consider congestion on soft edges when planning. 
                By default, only congestion on hard edges are considered.

        Output: a list multi-agent graph paths on HG.
    '''
    
    ## One-by-one cost aware planning
    HG.__reset_traffic__()
    paths = []
    for s,g in zip(HG.start_nodes,HG.goal_nodes):
        
        # print(s,g)
        path = nx.shortest_path(HG,s,g,weight = "traffic_cost")
        # print(path)

        # Update the edge flow along the path
        for i in range(len(path)-1):
            p,q = path[i],path[i+1]
            HG.edges[p,q]['flow'] += 1

        paths.append(path)
        # Important: update graph traffic
        HG.update_traffic(consider_soft_traffic)

    # Reset HG's state before returning    
    HG.__reset_traffic__()
    
    return paths
