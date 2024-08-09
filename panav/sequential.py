
from time import time
import signal
 
def timeout_handler(signum,frame):
    print("Algorithm timeout")
    raise TimeoutError("Timeout")
def sequential_planning(solver,env,vmax,bloating_r,TIMEOUT = 120,lazy = True,return_cumu_times = False):
    '''
        Sequential MAMP using a given solver.
    '''
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT)

    t0 = time()

    cumulative_times = []
    continuous_plans = []

    try:        
        for i in range(len(env.starts)):
            print('Planning for agent {}/{}'.format(i,len(env.starts)))
            start = env.starts[i,:]
            goal = env.goals[i,:]
            
            sol = solver(env,start,goal,vmax = vmax, bloating_r =  bloating_r)
            p = sol.plan(obstacle_trajectories = continuous_plans,lazy = lazy)    

            t_now = time()-t0
            if p is None:
                return None
            if t_now>TIMEOUT or p is None: # Stop early if the run time has exceeded TIMEOUT
                break

            cumulative_times.append(t_now)
            continuous_plans.append(p)
    except TimeoutError:
            print("Timeout")
            signal.alarm(0)

    
    if return_cumu_times:
        n_max = min(len(continuous_plans),len(cumulative_times))
        return continuous_plans[:n_max], cumulative_times[:n_max]
    
    return continuous_plans

import numpy as np
# from panav.HybridSIPP import HybridSIPP
from panav.HybridSIPP_new import HybridSIPP
def sequential_HybridSIPP(HG,return_graph = False,Delta = 2.0):
    
    agents = np.arange(len(HG.start_nodes))
    graph_plans = []
    continuous_plans = []

    U = {e:[] for e in HG.edges} # Edge traversal times
    C = {s:[] for s in HG} # Node visit times

    def update_traversal_records(U,C,gp):
        
        for s,t in gp:
            C[s].append(t)
        
        k = len(gp)
        for i in range(k-1):
            s,t = gp[i]
            sp,tp  = gp[i+1]
            U[s,sp].append((t,tp))
        return U,C
            

    for a in agents:
        print(a)
        result = HybridSIPP(HG,U,C,HG.start_nodes[a],HG.goal_nodes[a],continuous_plans,Delta)
        if result is not None:
            gp,cp = result
        else:
            print("Solver failed.")
            break

        U,C = update_traversal_records(U,C,gp)
        graph_plans.append(gp)
        continuous_plans.append(cp)

    if return_graph:
        return continuous_plans,graph_plans
    else:
        return continuous_plans