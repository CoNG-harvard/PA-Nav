
from time import time
 
def sequential_planning(solver,env,vmax,bloating_r,
                        TIMEOUT = 120,
                        lazy = True,
                        return_runtimes = False,
                        return_int_constrains = False):
    '''
        Sequential MAMP using a given solver.
    '''
  
    t0 = time()
    runtimes = []
    continuous_plans = []
    num_int_constraints = []

    for i in range(len(env.starts)):
        print('Planning for agent {}/{}'.format(i,len(env.starts)))

        start = env.starts[i,:]
        goal = env.goals[i,:]
        
        sol = solver(env,start,goal,vmax = vmax, bloating_r =  bloating_r)
        

        t_cur = time()
        p = sol.plan(obstacle_trajectories = continuous_plans,lazy = lazy)    
        t_fin = time()
    
        runtimes.append(t_fin-t_cur)
        t_cumu = t_fin-t0
        if p is None or t_cumu>TIMEOUT:
            return None # Stop early if the run time has exceeded TIMEOUT
        
        if return_int_constrains:
            num_int_constraints.append(sol.count_int_constraints(continuous_plans))
        continuous_plans.append(p)

        
        
    out = [continuous_plans]
    if return_runtimes:
        out.append(runtimes)
    if return_int_constrains:
        out.append(num_int_constraints)

    if len(out)>1:
        return out
        
    return continuous_plans

import numpy as np
from panav.HybridSIPP import HybridSIPP
def sequential_HybridSIPP(HG,return_graph = False,Delta = 2.0,Kmax = 3,return_on_failure=True):
    
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
        t0 = time()

        result = None
        for Km in range(Kmax+1):
            print('Km=',Km)
            result = HybridSIPP(HG,U,C,HG.start_nodes[a],HG.goal_nodes[a],continuous_plans,Delta,Kmax=Km)
            if result is not None:
                break
            
        if result is not None:
            gp,cp = result
        else:
            print("Solver failed.")
            if return_on_failure:
                break
            else:
                return None
        
        print('Solve time',time()-t0)
        
        U,C = update_traversal_records(U,C,gp)
        graph_plans.append(gp)
        continuous_plans.append(cp)

    if return_graph:
        return continuous_plans,graph_plans
    else:
        return continuous_plans