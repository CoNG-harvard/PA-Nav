import networkx as nx
import numpy as np
from queue import PriorityQueue
from panav.SIPP import compute_safe_intervals, compute_edge_weights, interval_intersection, plan_to_transitions
from panav.util import unique_tx
from panav.SAMP import Tube_Planning
from copy import deepcopy


def HybridSIPP(HG_in,node_locs,start,goal, obs_graph_trans, obs_continuous_path, v_max, bloating_r):
    '''
        HG_in: a networkx graph.

        node_locs: a dict in the form {s:loc of s for s in HG}

        start, goal: start and goal node(nodes in HG).
        
        obs_graph_trans: the transitions of the dynamic obstacle(s), in {(s_i,u_i,t1_i,t2_i):i=0,1,...} form.
            
            If s_i!=u_i, then (s_i,u_i) should be an edge in HG.
            
            t1_i<t2_i are positive real numbers representing the time interval during which the dynamic obstacle traverses(or stops at) (s_i,u_i).            
       
       obs_continuous_path: a tuple (times,xs) representing a continuous time-space path.

       v_max: the velocity our agent is expected to travel at.

       bloating_r:  the bloating radius for the agent and obstaclce.

       Output: a single-agent plan on HG in the form of [(s_i,t_i) for i=0,1,2,3...]
    '''
    HG = deepcopy(HG_in)

    if 'weight' not in list(HG.edges(data=True))[0][-1].keys():
        compute_edge_weights(HG,node_locs,v_max)

    compute_safe_intervals(HG,node_locs,obs_graph_trans,v_max,bloating_r)
    
    hScore = dict(nx.shortest_path_length(HG,weight = 'weight'))
    
    return Hybrid_SIPP_core(HG,start,goal,obs_continuous_path,hScore)


def Hybrid_SIPP_core(HG,start,goal,obs_continuous_path,hScore):
    OPEN = PriorityQueue()
   
    def safe_interval_index(t,s):
        '''
            Determine which safe interval of node s does time t fall into.
            
            If t is not in any safe intervals, return None.
        '''
        intervals = HG.nodes[s]['safe_intervals']
        
        for i in range(len(intervals)):
            if intervals[i][0]<=t<=intervals[i][1]:
                return i
        
        return None

    gScore = {(start,0):0}
    # gScore[s,idx,k] keeps track of the travel time from the start node to 
    # node s, arriving at the idx'th safe interval of s.

    OPEN.put((0,(start,0))) 

    # Items in the priority queue are in the form (gScore, item), sorted by value. 
    # The item with the smallest value is placed on top.

    cameFrom = {}
    transition_duration = {}

    def recover_path(final_state,start):
        path_temp = []
        curr = final_state
        while curr != (start,0):
            path_temp.append((curr[0],transition_duration[curr]))
            curr = cameFrom[curr]
        
        path_temp.reverse()

        plan = [(start,0)]
        prev_node = start
        t_prev = 0
        for i in range(len(path_temp)):
            dest_node,duration = path_temp[i]
            
            if t_prev<duration[0]:# The agent waited at prev node
                plan.append((prev_node,duration[0]))
            
            plan.append((dest_node,duration[1]))
            t_prev = duration[1]
            prev_node = dest_node
            
        return plan, unique_tx(*graph_plan_to_continuous(plan,HG))


    path = []

    # Reset the edge weights because they will be re-calculated.
    for e in HG.edges:
        if HG.edges[e]['type'] == 'soft':
            HG.edges[e]['weight'] = None

    while not OPEN.empty():
        curr_fscore,(s,i) = OPEN.get() # Remove the (s, i) with the smallest gScore.
        if s == goal:
            return recover_path((s,i),start)


        t = gScore[(s,i)]
        LB,UB = HG.nodes[s]['safe_intervals'][safe_interval_index(t,s)]

        for u in HG[s]:
            if HG.edges[s,u]['weight'] is None:
                # Re-compute the weights of soft edges using planning.
                ts,xs  = Tube_Planning(HG.env, 
                                   HG.nodes[s]['region'],HG.nodes[u]['region'],HG.vmax,HG.agent_radius,
                                   obs_continuous_path,HG.d,HG.K)
                

                HG.edges[s,u]['weight'] = np.max(ts)

                HG.edges[s,u]['continuous_path'] = xs
                HG.edges[s,u]['continuous_time'] = ts
                
            l = HG.edges[s,u]['weight']
            start_t = t + l
            end_t = UB + l
            for m,(lb,ub) in enumerate(HG.nodes[u]['safe_intervals']):
                int1 = interval_intersection((lb,ub),(start_t,end_t))
                if int1:
                    for lbp,ubp in HG.edges[s,u]['safe_intervals']:
                        int2 = interval_intersection((lbp,ubp),np.array(int1)-l)
                        if int2:
                            a,b = int2
                            if (u,m) not in gScore.keys():
                                gScore[(u,m)] = np.inf

                            if a+l<gScore[(u,m)]:
                                cameFrom[(u,m)] = (s,i)
                                gScore[(u,m)] = a+l
                                transition_duration[(u,m)] = (a,a+l)
                                fScore = a+l+hScore[u][goal]
                                OPEN.put((fScore,(u,m)))
    

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