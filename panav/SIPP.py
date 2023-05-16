import networkx as nx
import numpy as np
from queue import PriorityQueue


def SIPP(G,node_locs,start,goal, obs_trans, v_max, bloating_r):
    '''
        G: a networkx graph.

        node_locs: a dict in the form {s:loc of s for s in G}

        start, goal: start and goal node(nodes in G).
        
        obs_trans: the transitions of the dynamic obstacle(s), in {(s_i,u_i,t1_i,t2_i):i=0,1,...} form.
            
            If s_i!=u_i, then (s_i,u_i) should be an edge in G.
            
            t1_i<t2_i are positive real numbers representing the time interval during which the dynamic obstacle traverses(or stops at) (s_i,u_i).            
       
       v_max: the velocity our agent is expected to travel at.

       bloating_r:  the bloating radius for the agent and obstaclce.

       Output: a single-agent plan on G in the form of [(s_i,t_i) for i=0,1,2,3...]
    '''
    if 'weight' not in list(G.edges(data=True))[0][-1].keys():
        compute_edge_weights(G,node_locs,v_max)

    compute_safe_intervals(G,node_locs,obs_trans,v_max,bloating_r)
    
    hScore = dict(nx.shortest_path_length(G,weight = 'weight'))
    
    return SIPP_core(G,start,goal,hScore)



def plan_to_transitions(plan):
    '''
        plan: (s_i,t_i) pairs, t_i increases with i.

        Output: (s,u,t1,t2) transition pairs
    '''
    transitions = []
    for i in range(len(plan)-1):
        transitions.append((plan[i][0],plan[i+1][0],
                         plan[i][1],plan[i+1][1]))
    
    transitions.append((plan[-1][0],plan[-1][0],plan[-1][1],np.inf)) # Experimental. This lets the agent to stay at its goal and block other agents.

    return transitions

def compute_safe_intervals(G,node_locs,obs_trans,v_max,bloating_r):
    '''
        Compute the safe intervals of the nodes and edges of G, as G's node- and edge-attributes in place.
    '''


    nx.set_edge_attributes(G,{e:[] for e in G.edges},'unsafe_intervals')
    nx.set_edge_attributes(G,{e:[] for e in G.edges},'safe_intervals')

    nx.set_node_attributes(G,{s:[] for s in G},'unsafe_intervals')
    nx.set_node_attributes(G,{s:[] for s in G},'safe_intervals')


    for o in obs_trans:
        u,v,t1,t2 = o
        
        if u==v:
            G.nodes[u]['unsafe_intervals'].append([t1,t2])
        else:
            L = np.linalg.norm(node_locs[u]-node_locs[v])
            G.edges[v,u]['unsafe_intervals'].append([t1-L/v_max,t2])
            
            vel = L/(t2-t1)
            # self_traverse_t = 2*np.sqrt(2)*bloating_r/vel
            # self_traverse_t = 3*bloating_r/vel
            self_traverse_t = 4*bloating_r/vel

            '''
                self_traverse_t is a parameter we can change to tune the conservativeness collision avoidance.
                The coefficient 4 above is the default value, corresponding to sparing a generous space between two agents.
                Use coefficient 4 when calling SIPP in PBS_SIPP.

                A tighter coefficient could be 2*sqrt(2), corresponding to adjacent two edges are in 90-degree angle. See the following illustration.

                O
                  \
             ------s 
                  /
                O 

                In general, the safe interval can be flexibly designed to model the true collision physics between agents.
                The SIPP only solves the problem given the safe intervals.
            '''

            if v_max>=vel: # The agent chases the obstacle
                gap_t = ((v_max-vel)*(t2-t1) + 2* bloating_r)/v_max
                G.edges[u,v]['unsafe_intervals'].append([t1,np.min([t2,t1+gap_t])])
            else: # The obstacle chases the agent
                gap_t = ((vel-v_max)*(t2-t1) + 2* bloating_r)/v_max
                G.edges[u,v]['unsafe_intervals'].append([np.max([0,t1-gap_t]),t1])

            # The following are caused by agent touching the start and nodes when traversing the edge.
            G.nodes[u]['unsafe_intervals'].append([t1,t1+self_traverse_t])
            G.nodes[v]['unsafe_intervals'].append([t2-self_traverse_t,t2])
        

    for i in G:
        G.nodes[i]['unsafe_intervals'] = merge_intervals(G.nodes[i]['unsafe_intervals'])
        G.nodes[i]['safe_intervals'] = unsafe_to_safe( G.nodes[i]['unsafe_intervals'])

    for e in G.edges:
        G.edges[e]['unsafe_intervals']=merge_intervals(G.edges[e]['unsafe_intervals'])
        G.edges[e]['safe_intervals']=unsafe_to_safe(G.edges[e]['unsafe_intervals'])



def SIPP_core(G,start,goal,hScore):
    OPEN = PriorityQueue()

    def safe_interval_index(t,s):
        '''
            Determine which safe interval of node s does time t fall into.
            
            If t is not in any safe intervals, return None.
        '''
        intervals = G.nodes[s]['safe_intervals']
        
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
            
        return plan


    path = []
    while not OPEN.empty():
        curr_fscore,(s,i) = OPEN.get() # Remove the (s, i) with the smallest gScore.
        if s == goal:
            return recover_path((s,i),start)


        t = gScore[(s,i)]
        LB,UB = G.nodes[s]['safe_intervals'][safe_interval_index(t,s)]

        for u in G[s]:
            l = G.edges[s,u]['weight']
            start_t = t + l
            end_t = UB + l
            for m,(lb,ub) in enumerate(G.nodes[u]['safe_intervals']):
                int1 = intersection((lb,ub),(start_t,end_t))
                if int1:
                    for lbp,ubp in G.edges[s,u]['safe_intervals']:
                        int2 = intersection((lbp,ubp),np.array(int1)-l)
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


  
def compute_edge_weights(G,node_locs,v_max):
    nx.set_edge_attributes(G,
                           {e:np.linalg.norm(node_locs[e[0]]-node_locs[e[1]])/v_max\
                              for e in G.edges},
                           'weight')


def merge_intervals(arr):
    '''
        Source: https://www.geeksforgeeks.org/merging-intervals/
    '''
    
    # Sorting based on the increasing order
    # of the start intervals
    arr.sort(key=lambda x: x[0])
 
    # Stores index of last element
    # in output array (modified arr[])
    index = 0
 
    # Traverse all input Intervals starting from
    # second interval
    for i in range(1, len(arr)):
        # If this is not first Interval and overlaps
        # with the previous one, Merge previous and
        # current Intervals
        if (arr[index][1] >= arr[i][0]):
            arr[index][1] = max(arr[index][1], arr[i][1])
        else:
            index = index + 1
            arr[index] = arr[i]
 
    return arr[:index+1]

def unsafe_to_safe(unsafe_intervals,t0=0):
    t_prev = t0
    safe_intervals = []

    for i in range(len(unsafe_intervals)):
        if t_prev<unsafe_intervals[i][0]:
            safe_intervals.append((t_prev,unsafe_intervals[i][0]))

        t_prev = unsafe_intervals[i][1]

    safe_intervals.append((t_prev,np.inf))
    return safe_intervals


def intersection(interval1,interval2):
    lb,ub = max([interval1[0],interval2[0]]),min([interval1[1],interval2[1]])
    if lb>=ub:
        return None
    else:
        return (lb,ub)
                                      
