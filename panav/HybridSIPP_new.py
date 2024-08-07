import itertools
import networkx as nx
import numpy as np
from queue import PriorityQueue
from panav.SIPP import plan_to_transitions
from panav.util import unique_tx
# from panav.SAMP.archaic import Tube_Planning
from panav.SAMP.solvers import Tube_Planning
from copy import deepcopy


def HybridSIPP(HG_in,U,C,start,goal,obs_continuous_paths,Delta):

    HG = deepcopy(HG_in)

    hScore = dict(nx.shortest_path_length(deepcopy(HG_in),weight = 'weight'))
    
    return Hybrid_SIPP_core(HG,U,C,start,goal,obs_continuous_paths,hScore,Delta)

def compute_safe_intervals(HG,v,w,U,C,t0,Delta):
    return []

def Hybrid_SIPP_core(HG,U,C,start,goal,obs_continuous_paths,hScore,Delta):
    
    OPEN = PriorityQueue()

    gScore = {start:0}
    # gScore[(s,i)] keeps track of the travel time from the start node to 
    # the i'th safe interval of node s.

    OPEN.put((hScore[start][goal],start))
    # Items in the priority queue are in the form (gScore, item), sorted by value. 
    # The item with the smallest value is placed on top.

    cameFrom = {}
    def recover_path(final_st,cameFrom): # Helper function for recovering the agents' paths using the cameFrom dictionary.
        g_plan = [(final_st,gScore[final_st])]
        t_plan = []
        x_plan = []
        curr = final_st
        while curr != start:
            
            curr,(tp,xp) = cameFrom[curr]
            g_plan.append((curr,gScore[curr]))
            t_plan.append(tp[1:])
            x_plan.append(xp[:,1:])

        g_plan.append((start,0))
        t_plan.append(0)
        x_plan.append(HG.node_loc(start))

        g_plan.reverse()
        t_plan.reverse()
        x_plan.reverse()

        g_plan = unique_graph_steps(g_plan)

        t_plan = np.hstack(t_plan)
        x_plan = np.hstack(x_plan)
        
        return g_plan, unique_tx(t_plan,x_plan)
        # return path

    while not OPEN.empty():
        _,v = OPEN.get() # Remove the s with the smallest f-score.
        if v == goal:
            return recover_path(v,cameFrom)
                  
        for w in HG[v]:
            t0 = gScore[v]
            S = compute_safe_intervals(HG,v,w,U,C,t0,Delta)
            S.sort(key=lambda x: x[0]) # Sort the intervals by starting values.
 
            for lb,ub in S:     
                if goal not in hScore[v].keys(): # goal not reachable from u
                    continue
                
            
                if HG.edges[v,w]['type'] == 'soft':  
                    planner = Tube_Planning(HG.env, HG.node_loc(v),HG.node_loc(w),
                                            HG.vmax,HG.agent_radius,
                                            t0 = t0, T_end_constraints= [(lb,ub)] , ignore_finished_agents=True,
                                            K_max=12)
                    plan_result = planner.plan(obstacle_trajectories=obs_continuous_paths)
                        

                    if plan_result is None: 
                        continue  # Impossible to safely arrive at w during (lb,ub)
                    else:
                        tp,xp = plan_result
                else:
                    t_min = t0 + HG.edges[v,w]['weight']/HG.vmax
                    if t_min>ub:
                        continue  # Impossible to safely arrive at w during (lb,ub)
                    else:
                        tp = np.array([t0,max(t_min,lb)])
                        xp = np.array([HG.node_loc(v),HG.node_loc(w)]).T
                        
                                       
                # The rest is standard A*
                if w not in gScore.keys():
                    gScore[w] = np.inf

                t_K = np.max(tp)
                if t_K < gScore[w]: # The A* update
                    cameFrom[w] = (v,(tp,xp))
                    gScore[w] = t_K
                    fScore = t_K+hScore[w][goal]
                    OPEN.put((fScore,w))
                else:
                    break  # At later (lb,ub), t_K only increases.
    return None
    

def graph_plan_to_continuous(go_plan,HG):
        
    graph_trans= plan_to_transitions(go_plan)
    full_time = [0]
    full_path = [HG.node_loc(go_plan[0][0][0])[:,np.newaxis]]
    for (u,ui),(v,vi),t0,t1 in graph_trans[:-1]:
        
        if u==v:
            if t1-t0>1e-2:
                full_time.append(t1)
                full_path.append(full_path[-1][:,-1][:,np.newaxis])
        else:
            # if HG.edges[u,v]['type'] == 'soft':
            ct = HG.edges[u,v]['continuous_time'][ui][vi]
            cp = HG.edges[u,v]['continuous_path'][ui][vi]
            # else:
            #     ct = HG.edges[u,v]['continuous_time']
            #     cp = HG.edges[u,v]['continuous_path']
                
            full_time.extend(list(ct[1:]+full_time[-1]))
            full_path.append(cp[:,1:])
    return np.array(full_time),np.hstack(full_path)

def unique_graph_steps(graph_plan):
    '''
        graph_plan: [(node[i], time[i]) for i = 1,2,3,...n]
    '''
    ug = []
    if len(graph_plan)>=1:
        ug.append(graph_plan[0])
        if len(graph_plan)>1:
            for n,t in graph_plan[1:]:
                if t-ug[-1][1]>=1e-3:
                    ug.append((n,t))
    return ug
