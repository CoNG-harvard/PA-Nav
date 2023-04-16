from queue import PriorityQueue
from copy import deepcopy
import networkx as nx
import numpy as np

from panav.PBS.HighLevelSearchTree import PriorityTree, SearchNodeContainer
from panav.SAMP import Tube_Planning
from panav.PBS.conflict import MA_plan_conflict
from panav.util import unique_tx


def flowtime(plan):
    return np.sum([t[-1] for t,x in plan])
def makespan(plan):
    return np.max([t[-1] for t,x in plan])


def PBS(env,vmax,bloating_r,\
        d,K,t0,\
        max_iter = 200,metric = 'flowtime',search_type = 'depth_first'\
        ):
    agents = set(np.arange(len(env.starts)))
    if metric == 'flowtime':
        metric = flowtime
    elif metric == 'makespan':
        metric = makespan
    else:
        print('Metric {} is not supported. Please input "flowtime" or "makespan".'.format(metric))

    # Initialize the agents' plans
    plan0 = []
    for agent in agents:
        start,goal = env.starts[agent],env.goals[agent]

        t, xs = Tube_Planning(env,start,goal,vmax,bloating_r,\
                                     [],\
                                    d,K,t0)
        t,xs = unique_tx(t,xs)
        plan0.append((t,xs))

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

        conflict = MA_plan_conflict(solution,bloating_r) # Look for the first conflict.
        if conflict is None:
            return solution, cost
        else:
            ([a1,_,_],[a2,_,_]) = conflict # Get the two agents involved in the conflict.

        prev_ordering = PT.get_ordering(parent_node)
        new_PT_nodes = PriorityQueue()

        # Compute new PT nodes
        for (j,i) in [(a1,a2),(a2,a1)]:
            new_plan = deepcopy(solution)
            new_order = [(j,i)] 
            if (i,j) in prev_ordering or len(list(nx.simple_cycles(nx.DiGraph(prev_ordering+new_order))))>0: 
                    print('Skipping ij',(j,i),'prev_ordering',prev_ordering)
                    continue # Do not add (j,i) to the partial ordering if it introduces a cycle.
            curr_ordering = prev_ordering+new_order

            sorted_agents  = list(nx.topological_sort(nx.DiGraph(curr_ordering))) # Get all the agents with lower orderings than i.

            idx_i = np.where(np.array(sorted_agents)==i)[0][0]

            agents_to_avoid = [a for a in sorted_agents[:idx_i]]

            success_update = True
            for k in range(idx_i, len(sorted_agents)):
                agent_to_update = sorted_agents[k]
                
                # Update the plan for agent_to_update
                start, goal = env.starts[agent_to_update],env.goals[agent_to_update]
                obs_trajectories = [new_plan[av] for av in agents_to_avoid]

                result = Tube_Planning(env,start,goal,vmax,bloating_r,\
                                     obs_trajectories,\
                                    d,K,t0)

                if result is not None:
                    new_plan[agent_to_update] = unique_tx(*result)
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

    print('Total iterations = ',count,'OPEN empty?',OPEN.empty())
    return None


            