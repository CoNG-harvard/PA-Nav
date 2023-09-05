import itertools
import networkx as nx
import numpy as np
from queue import PriorityQueue
from panav.SIPP import compute_safe_intervals, compute_edge_weights, interval_intersection, plan_to_transitions
from panav.util import unique_tx
from panav.SAMP import Tube_Planning, SA_MILP_Planning
from copy import deepcopy


def HybridSIPP(HG_in,start,goal, obs_graph_paths, obs_continuous_paths):
    '''
        HG_in: a networkx graph.

        node_locs: a dict in the form {s:loc of s for s in HG}

        start, goal: start and goal node(nodes in HG).
       
        obs_graph_paths: a list of (node, time)-lists [
                                            [(node[i][0],t[i][0]),(node[i][1],t[i][1]),...,(node[i][k_i],t[i][k_i])]
                                             for i = 1,2,...,nAgents   
                                            ]
       
        obs_continuous_path: a list of tuples [(times,xs)] representing the continuous time-space paths of moving obstacles.


        Output: a single-agent graph path on HG in the form of [(s_i,t_i) for i=0,1,2,3...], as well as its continuous space-time realization (t_out, x_out).
    '''
    HG = deepcopy(HG_in)
    node_locs = HG.node_locs()
    
    obs_graph_trans = list(itertools.chain.from_iterable([plan_to_transitions(g) for g in obs_graph_paths]))
    compute_safe_intervals(HG,node_locs,obs_graph_trans,HG.vmax,HG.agent_radius,
                           merge_node_edge_intervals=True) # We must set the merge flag to be True here.
    
    hScore = dict(nx.shortest_path_length(deepcopy(HG_in),weight = 'weight'))
    
    return Hybrid_SIPP_core(HG,start,goal,obs_continuous_paths,hScore)


def Hybrid_SIPP_core(HG,start,goal,obs_continuous_paths,hScore):
    
    OPEN = PriorityQueue()

    gScore = {start:0}
    # gScore[s] keeps track of the travel time from the start node to 
    # node s.

    OPEN.put((0,start))
    # Items in the priority queue are in the form (gScore, item), sorted by value. 
    # The item with the smallest value is placed on top.

    cameFrom = {}
    def recover_path(final_st,cameFrom): # Helper function for recovering the agents' paths using the cameFrom dictionary.
        path = []
        curr = final_st
        while curr != start:
            path.append((curr,gScore[curr]))
            curr = cameFrom[curr]

        path.append((start,gScore[start]))
        path.reverse()

        path = unique_graph_steps(path)

        return path, unique_tx(*graph_plan_to_continuous(path,HG))

    path = []

    for e in HG.edges:
        if HG.edges[e]['type'] == 'soft':
            HG.edges[e]['continuous_time'] = None
            HG.edges[e]['continuous_path'] = None

    while not OPEN.empty():
        curr_fscore,s = OPEN.get() # Remove the s with the smallest gScore.
        if s == goal:
            return recover_path(s,cameFrom)

        for u in HG[s]:
            if goal not in hScore[u].keys(): # goal not reachable from u
                continue
            safe_intervals = HG.nodes[u]['safe_intervals']
            curr_t = gScore[s]
            print("Safe interval of node",s, HG.nodes[s]['safe_intervals'])
            print("Safe interval of node",u, HG.nodes[u]['safe_intervals'])
            print("curr time",curr_t)
            
            
            if HG.edges[s,u]['type'] == 'soft':
                if HG.edges[s,u]['continuous_time'] is None:
                    # for all possible safe intervals at s do
                    # Compute the weight for the travel plan
                    
                    # print('solving for edge', s,u,'curr_t',curr_t)
                    possible_K = [2,3,5,6]

                    for K in possible_K:
                        
                        # print("K",K,"safe intervals",safe_intervals)
                        # print("start",s,HG.node_loc(s),"end",u,HG.node_loc(u))
                        plan_result = Tube_Planning(HG.env, 
                                            HG.nodes[s]['region'],HG.nodes[u]['region'],HG.vmax,HG.agent_radius,
                                            obs_continuous_paths,HG.d,
                                            K,t0 = curr_t,
                                            T_end_constraints=safe_intervals, ignore_finished_agents=True)
                        
                        if plan_result is not None:
                            break

                    # print('plan_result',plan_result)
                    if plan_result is None: # Infeasible. Could be that K value is low.
                        # print(s,u,"Continuous path in open space not found. Consider increasing K value.")
                        HG.edges[s,u]['weight'] = np.inf
                        HG.edges[s,u]['continuous_time'] = np.array([0,np.inf])
                    else:
                        tp,xp = plan_result
                        HG.edges[s,u]['weight'] = np.max(tp)-np.min(tp)
                        HG.edges[s,u]['continuous_path'] = xp
                        HG.edges[s,u]['continuous_time'] = tp-np.min(tp)
                
            elif HG.edges[s,u]['type']=='hard':
                # curr_t = gScore[s]

                # # Slow down continuous time if necessary

                # # print('safe intervals',safe_intervals)
                # # print('curr_t',curr_t,"earliest arrival", curr_t + HG.edges[s,u]['weight'])
                
                # for lb, ub in safe_intervals:
                #     if curr_t + HG.edges[s,u]['weight']<=ub:
                #         eta = np.max([lb-curr_t, HG.edges[s,u]['weight']])
                #         HG.edges[s,u]['continuous_time'] = np.array([0,eta])
                #         # print('Continuous time',(0,eta))
                #         break
                # if HG.edges[s,u]['continuous_time'] is None:
                    # print('solving for hard edge', s,u)
                        
                    # plan_result = Tube_Planning(HG.env, 
                    #                             HG.nodes[s]['region'],HG.nodes[u]['region'],HG.vmax,HG.agent_radius,
                    #                             obs_continuous_paths,HG.d,
                    #                             3,t0 = curr_t,
                    #                             T_end_constraints=safe_intervals)
                        
                    # if plan_result is None: # Infeasible. Could be that K value is low.
                    #     print(s,u,"Continuous path in open space not found. Consider increasing K value.")
                    #     HG.edges[s,u]['weight'] = np.inf
                    #     HG.edges[s,u]['continuous_time'] = np.array([0,np.inf])
                    # else:
                    #     tp,xp = plan_result
                    #     HG.edges[s,u]['weight'] = np.max(tp)-np.min(tp)
                    #     HG.edges[s,u]['continuous_path'] = xp
                    #     HG.edges[s,u]['continuous_time'] = tp-np.min(tp)
                    pass
                
            # The rest is standard A*
            if u not in gScore.keys():
                gScore[u] = np.inf

            travel_time = np.max(HG.edges[s,u]['continuous_time'])
            if gScore[s] + travel_time < gScore[u]: # The A* update
                cameFrom[u] = s
                gScore[u] = gScore[s] + travel_time
                fScore = gScore[u]+hScore[u][goal]
                OPEN.put((fScore,u))
    return None
    

def graph_plan_to_continuous(go_plan,HG):
    graph_trans= plan_to_transitions(go_plan)
    full_time = [0]
    full_path = [HG.node_loc(go_plan[0][0])[:,np.newaxis]]
    for u,v,t0,t1 in graph_trans[:-1]:
       
        if u==v:
            if t1-t0>1e-2:
                full_time.append(t1)
                full_path.append(full_path[-1][:,-1][:,np.newaxis])
        else:
            full_time.extend(list(HG.edges[u,v]['continuous_time'][1:]+full_time[-1]))
            full_path.append(HG.edges[u,v]['continuous_path'][:,1:])

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
