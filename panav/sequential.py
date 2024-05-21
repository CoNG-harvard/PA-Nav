
from time import time
def sequential_planning(solver,env,vmax,bloating_r,TIMEOUT = 120):
    '''
        Sequential MAMP using a given solver.
    '''
    t0 = time()
    continuous_plans = []

    for i in range(len(env.starts)):
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