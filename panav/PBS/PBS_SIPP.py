import itertools
from queue import PriorityQueue
from copy import deepcopy
import numpy as np
import networkx as nx


from panav.MAPFR_conflict import MAPFR_conflict
from panav.SIPP import SIPP,plan_to_transitions

from panav.PBS.HighLevelSearchTree import PriorityTree, SearchNodeContainer

def flowtime(plan):
    return np.sum([wp[-1][1] for wp in plan])
def makespan(plan):
    return np.max([wp[-1][1] for wp in plan])


def PBS_SIPP(G,node_locs,starts,goals,\
             vmax,bloating_r,\
        max_iter = 200,metric = 'flowtime',search_type = 'depth_first'\
        ):
    '''
        The MAMP algorithm using PBS as the high-level search and tube-based SAMP as the low-level search.
        
        Inputs: 
            G: The path planning environment.
            
            node_locs: A dict {s:loc[s] for s in G}, storing the location of each node.
            
            starts, goals: The start and goal nodes.
            
            vmax,bloating_r: SIPP parameters.
                
            max_iter: the maximal iteration of PBS mainloop before force exit and return solution not found.

            metric: the metric used in high-level search, can be either 'flowtime' or 'makespan'.

            search_type: the branching style used in high-level search, can be either 'depth_first'(fast) or 'best_first'(slow).
        
        Output: a multi-agent plan in the form of [(s_i,t_i)]
    '''
    agents = set(np.arange(len(starts)))
    if metric == 'flowtime':
        metric = flowtime
    elif metric == 'makespan':
        metric = makespan
    else:
        print('Metric {} is not supported. Please input "flowtime" or "makespan".'.format(metric))

    # Initialize the agents' plans
    plan0 = []
    for agent in agents:
        start,goal = starts[agent],goals[agent]
        plan = SIPP(G,node_locs,start,goal,[],vmax,bloating_r)
        plan0.append(plan)
    
    cost0 = metric(plan0)

    PT = PriorityTree()
    ROOT = PT.add_node(None,plan0,cost0,ordering=[]) # Adding the root node. 

    OPEN = SearchNodeContainer(search_type)
    OPEN.push(cost0,ROOT)
    
    # Enter the main PBS loop.
    count = 0
    while not OPEN.empty() and count<=max_iter:
        count+=1 

        cost, parent_node = OPEN.pop()
        solution = PT.get_solution(parent_node)
        
        conflict = MAPFR_conflict(G,solution,node_locs,bloating_r) # Look for the first conflict.
        
        if conflict is None:
            return solution, cost
        else:
            a1,a2 = conflict['agents'] # Get the two agents involved in the conflict.

        prev_ordering = PT.get_ordering(parent_node)
        new_PT_nodes = PriorityQueue()

        # Compute new PT nodes
        for (j,i) in [(a1,a2),(a2,a1)]:
            new_plan = deepcopy(solution)
            new_order = [(j,i)] 
            if (j,i) in prev_ordering or len(list(nx.simple_cycles(nx.DiGraph(prev_ordering+new_order))))>0: 
                    # print('Skipping ji',(j,i),'prev_ordering',prev_ordering)
                    continue # Do not add (j,i) to the partial ordering if it introduces a cycle.
            curr_ordering = prev_ordering+new_order

            sorted_agents  = list(nx.topological_sort(nx.DiGraph(curr_ordering))) # Get all the agents with lower orderings than i.

            idx_i = np.where(np.array(sorted_agents)==i)[0][0]
            
            agents_to_avoid = [a for a in sorted_agents[:idx_i]]
            
            success_update = True
            for k in range(idx_i, len(sorted_agents)):
                agent_to_update = sorted_agents[k]
                
                # Update the plan for agent_to_update
                start, goal = starts[agent_to_update],goals[agent_to_update]
                obs_transitions = itertools.chain.from_iterable(\
                                    [plan_to_transitions(new_plan[av]) \
                                     for av in agents_to_avoid])
                
                result = SIPP(G,node_locs,start,goal,obs_transitions,vmax,bloating_r)

                if result is not None:
                    new_plan[agent_to_update] = result
                    agents_to_avoid.append(agent_to_update)
                else:
                    success_update = False 
                    break
            
            if success_update:
                cost = metric(new_plan) 

                neg_of_cost = -cost
                new_node = PT.add_node(parent_node, new_plan, cost,new_order)
                new_PT_nodes.put((neg_of_cost, new_node)) 
                # This is to ensure the PT node with higher cost will be added first to OPEN.
                
        # Put the (at most two) new PT nodes onto OPEN in non-increasing order of the cost.
        while not new_PT_nodes.empty():
            neg_of_cost, PT_node = new_PT_nodes.get()
            cost = -neg_of_cost
            OPEN.push(cost, PT_node)
        
        # Debug purpose
        # if OPEN.empty():
        #     return solution, cost

    if OPEN.empty() and count<max_iter:
        print("PBS fail to find a feasible solution. Please make sure the agents' goals are not blocking other agents paths.")
    return None