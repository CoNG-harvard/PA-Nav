
from time import time
def sequential_planning(solver,env,vmax,bloating_r,TIMEOUT = 120):
    '''
        Sequential MAMP using a given solver.
    '''
    t0 = time()
    continuous_plans = []

    for i in range(len(env.starts)):
        print('Planning for agent {}/{}'.format(i,len(env.starts)))
        start = env.starts[i,:]
        goal = env.goals[i,:]
        
        
        sol = solver(env,start,goal,vmax = vmax, bloating_r =  bloating_r)
        p = sol.plan(obstacle_trajectories=continuous_plans)    

        if time()-t0>TIMEOUT: # Stop early if the run time has exceeded TIMEOUT
            return None
        
        if p is None:
            return None
        continuous_plans.append(p)
    
    return continuous_plans

import numpy as np
from panav.HybridSIPP import HybridSIPP
def sequential_HybridSIPP(HG,return_graph = False):
    
    agents = np.arange(len(HG.start_nodes))
    graph_plans = []
    continuous_plans = []

    for a in agents:
        print(a)
        result = HybridSIPP(HG,HG.start_nodes[a],HG.goal_nodes[a],graph_plans,continuous_plans)
        if result is not None:
            gp,cp = result
        else:
            print("Solver failed.")
            return []
        graph_plans.append(gp)
        continuous_plans.append(cp)

    if return_graph:
        return continuous_plans,graph_plans
    else:
        return continuous_plans