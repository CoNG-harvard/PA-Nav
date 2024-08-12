import itertools
import networkx as nx
import numpy as np
from queue import PriorityQueue
from panav.SIPP import plan_to_transitions
from panav.util import unique_tx
# from panav.SAMP.archaic import Tube_Planning
from panav.SAMP.solvers import Tube_Planning
from copy import deepcopy

from panav.conflict import plan_obs_conflict


def HybridSIPP(HG_in,U,C,start,goal,obs_continuous_paths,Delta,Kmax = 3):
    '''
        The Kmax for HybridSIPP has to be kept no more than 3 for this algorithm to be meaningfully efficient.
    '''
    HG = deepcopy(HG_in)

    hScore = dict(nx.shortest_path_length(deepcopy(HG_in),weight = 'weight'))
    
    return Hybrid_SIPP_core(HG,U,C,start,goal,obs_continuous_paths,hScore,Delta,Kmax = Kmax)

from panav.SIPP import merge_intervals, unsafe_to_safe

def compute_safe_intervals(HG,v,w,U,C,tau,Delta,eps = 1e-3):
    US = [] # Container for unsafe intervals
    for t in C[w]:
        US.append((t-Delta, t+Delta))

    # print('v,w',(v,w),'tau',tau)
    # print("US for {}:".format((v,w)),US)


    if HG.edges[v,w]['type'] == 'hard':
        for t1,t2 in U[w,v]: # The opposition scenario
            if tau < t1 - Delta - np.linalg.norm(HG.node_loc(w)-HG.node_loc(v))/ HG.vmax - eps:
                US.append((t1-Delta,np.inf))
            elif tau < t2 + Delta - eps:
                return [] # Conflict with opposing (t1,t2) traversal is not avoidable
            else:
                continue # tau >= t2 + Delta, the agent will always be safe 
        for t1,t2 in U[v,w]: # Slow obstacle scenario
            if tau < t1 - Delta - eps:
                US.append((t2-Delta,np.inf)) 
            elif tau < t1 + Delta - eps:
                return [] 
            else:
                US.append((0,t2+Delta))
    elif HG.nodes[w]['type'] == 'tunnel': # Pre-add surely infeasible intervals to entry time
        for z in HG[w]:
            if HG.edges[w,z]['type'] == 'hard':
                for t1,t2 in U[z,w]: # The opposition scenario
                    US.append((t1 - Delta - np.linalg.norm(HG.node_loc(w)-HG.node_loc(z))/ HG.vmax, t2 + Delta))
                for t1,t2 in U[w,z]: # Slow obstacle scenario
                    US.append((t1 - Delta,t1 + Delta)) 
                        
    US = merge_intervals(US)

    return unsafe_to_safe(US)

from itertools import count

def Hybrid_SIPP_core(HG,U,C,start,goal,obs_continuous_paths,hScore,Delta,Kmax = 3):
    def SearchNode(v,g,f,parent,path):
        return {"v":v,"g":g,"f":f,"parent":parent,"path":path}

    
    unique = count()
    
    OPEN = PriorityQueue()

    # gScore = {start:0}
    # gScore[(s,i)] keeps track of the travel time from the start node to 
    # the i'th safe interval of node s.

    TN_0 = SearchNode(start,0,hScore[start][goal], None, None)
    OPEN.put((TN_0['f'], next(unique), TN_0)) # next(unique) is needed here to avoid some none uniqueness issue for class PriorityQueue

    # Items in the priority queue are in the form (fScore, item), sorted by value. 
    # The item with the smallest value is placed on top.

    # cameFrom = {}
    def recover_path(final_st): # Helper function for recovering the agents' paths using the cameFrom dictionary.
        g_plan = []
        t_plan = []
        x_plan = []
        curr = final_st
        while curr['parent'] is not None:
            
            g_plan.append((curr['v'],curr['g']))
            
            (tp,xp) = curr['path']
            t_plan.append(tp[1:])
            x_plan.append(xp[:,1:])

            curr = curr['parent']

        g_plan.append((start,0))
        t_plan.append(0)
        x_plan.append(HG.node_loc(start).reshape(-1,1))

        g_plan.reverse()
        t_plan.reverse()
        x_plan.reverse()

        g_plan = unique_graph_steps(g_plan)

        t_plan = np.hstack(t_plan)
        x_plan = np.hstack(x_plan)
        
        return g_plan, unique_tx(t_plan,x_plan)
        # return path

    while not OPEN.empty():
        fsc,_,TN = OPEN.get() # Remove the s with the smallest f-score.
        v = TN['v']                  
        t0 = TN['g']
        # print('v',v,'t0',t0,'h',fsc-t0)
        if v == goal:
            return recover_path(TN)
     
        for w in HG[v]:
            if goal not in hScore[w].keys(): # goal not reachable from w
                continue
            
            S = compute_safe_intervals(HG,v,w,U,C,t0,Delta)
            S.sort(key=lambda x: x[0]) # Sort the intervals by starting values.

            soft_plan = False
            if HG.edges[v,w]['type'] == 'soft':
                soft_plan = True
            # if HG.edges[v,w]['type'] == 'hard' and len(S) == 0:
            #     S = [(0,np.inf)]
            #     soft_plan = True
            
            for lb,ub in S:         
                t_min = t0 + HG.edges[v,w]['weight']/HG.vmax
                if t_min>ub:
                    continue  # Impossible to safely arrive at w during (lb,ub)        
                tp = np.array([t0,max(t_min,lb)])
                xp = np.array([HG.node_loc(v),HG.node_loc(w)]).T

                 # Check whether the two-point plan suffices
                if soft_plan and\
                    plan_obs_conflict((tp,xp),obs_continuous_paths,HG.agent_radius):
                        
                        # print('soft planning for',(v,w),
                        #       'edge type',HG.edges[v,w]['type'],
                        #     '(lb,ub)',(lb,ub))
                        planner = Tube_Planning(HG.env, HG.node_loc(v),HG.node_loc(w),
                                            HG.vmax,HG.agent_radius,
                                            t0 = t0, T_end_constraints= [(lb,ub)],
                                            ignore_finished_agents=True,
                                            K_max=Kmax)
                        plan_result = planner.plan(obstacle_trajectories=obs_continuous_paths)
                            
                        if plan_result is None: 
                            continue  # Impossible to safely arrive at w during (lb,ub)
                        else:
                            tp,xp = plan_result
                else:
                    pass
                    # print('no need for soft planning at', (v,w),
                    #       'edge type',HG.edges[v,w]['type'],
                    #       '(lb,ub)',(lb,ub))


                # if soft_plan:  
                #     # print('soft plan')
                #     planner = Tube_Planning(HG.env, HG.node_loc(v),HG.node_loc(w),
                #                             HG.vmax,HG.agent_radius,
                #                             t0 = t0, T_end_constraints= [(lb,ub)],
                #                             ignore_finished_agents=True,
                #                             K_max=Kmax)
                #     plan_result = planner.plan(obstacle_trajectories=obs_continuous_paths)
                        
                #     if plan_result is None: 
                #         continue  # Impossible to safely arrive at w during (lb,ub)
                #     else:
                #         tp,xp = plan_result
                # else:
                #     t_min = t0 + HG.edges[v,w]['weight']/HG.vmax
                #     if t_min>ub:
                #         continue  # Impossible to safely arrive at w during (lb,ub)
                #     else:
                #         # print('hard plan')
                #         tp = np.array([t0,max(t_min,lb)])
                #         xp = np.array([HG.node_loc(v),HG.node_loc(w)]).T
                        
                                       
                # The rest is standard A*
                t_K = np.max(tp)
                fScore = t_K + hScore[w][goal]
                TN_new = SearchNode(w,t_K,fScore,TN, (tp,xp))
                
                # try:
                OPEN.put((fScore, next(unique), TN_new)) # next(unique) is needed here to avoid some none uniqueness issue for class PriorityQueue

                # except Exception:
                #     pass # The exception will occur when two items with the same fScore and TN_new are in the queue
                         # This is a benign error for PriorityQueue and we will ignore it.
             

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
