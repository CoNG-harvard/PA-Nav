from queue import PriorityQueue
from copy import deepcopy
import networkx as nx
import numpy as np

from panav.PBS.HighLevelSearchTree import PriorityTree, SearchNodeContainer
from panav.HybridSIPP import HybridSIPP
from panav.conflict import MA_plan_conflict
from panav.util import unique_tx


def flowtime(plan):
    return np.sum([t[-1] for t,x in plan])
def makespan(plan):
    return np.max([t[-1] for t,x in plan])


def PBS_hybrid_SIPP(HG,max_iter = 200,metric = 'flowtime',search_type = 'depth_first'):
    '''
        The MAMP algorithm on the hybrid graph environment, using PBS as the high-level search and tube-based SAMP as the low-level search.
        
        Inputs: 
            HG: panav.hybrid.HybridGraph object. The path planning environment.
                      
            max_iter: the maximal iteration of PBS mainloop before force exit and return solution not found.

            metric: the metric used in high-level search, can be either 'flowtime' or 'makespan'.

            search_type: the branching style used in high-level search, can be either 'depth_first'(fast) or 'best_first'(slow).
    '''
    agents = set(np.arange(len(HG.env.starts)))
    if metric == 'flowtime':
        metric = flowtime
    elif metric == 'makespan':
        metric = makespan
    else:
        print('Metric {} is not supported. Please input "flowtime" or "makespan".'.format(metric))

    # Initialize the agents' plans
    graph_plan0 = []
    cont_plan0 = []
    for agent in agents:
        start,goal = HG.start_nodes[agent],HG.goal_nodes[agent]

        g_plan, c_plan = HybridSIPP(HG,start,goal,[],[])
        
        graph_plan0.append(g_plan)
        cont_plan0.append(c_plan)

    cost0 = metric(cont_plan0)

    PT = PriorityTree()
    ROOT = PT.add_node(None,(graph_plan0,cont_plan0),cost0,ordering=[]) # Adding the root node. 

    OPEN = SearchNodeContainer(search_type)
    OPEN.push(cost0,ROOT)
    
    # Enter the main PBS loop.
    count = 0
    while not OPEN.empty() and count<=max_iter:
        count+=1 

        cost, parent_node = OPEN.pop()
        g_joint_plan,c_joint_plan = PT.get_solution(parent_node)

        conflict = MA_plan_conflict(c_joint_plan,HG.agent_radius) # Look for the first conflict.
        if not conflict:
            return (g_joint_plan,c_joint_plan), cost
        else:
            (a1,a2) = conflict # Get the two agents involved in the conflict.

        prev_ordering = PT.get_ordering(parent_node)
        new_PT_nodes = PriorityQueue()

        # Compute new PT nodes
        for (j,i) in set([(a1,a2),(a2,a1)]):
            new_g_plan,new_c_plan = deepcopy((g_joint_plan,c_joint_plan))
            new_order = [(j,i)] 
            if (i,j) in prev_ordering \
                or (j,i) in prev_ordering\
                or len(list(nx.simple_cycles(nx.DiGraph(prev_ordering+new_order))))>0: 
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
                start, goal = HG.start_nodes[agent_to_update],HG.goal_nodes[agent_to_update]
                
                result = []
              
                obs_graph_paths = [new_g_plan[av] for av in agents_to_avoid]
                obs_cont_paths = [new_c_plan[av] for av in agents_to_avoid]

                result = HybridSIPP(HG, start,goal, obs_graph_paths,obs_cont_paths)

                # print("result",result)
                if result is not None:
                    new_g_plan[agent_to_update],new_c_plan[agent_to_update] = result
                    agents_to_avoid.append(agent_to_update)
                else:
                    success_update = False 
                    break
                    
            if success_update:
                cost = metric(new_c_plan) 

                neg_of_cost = -cost
                new_node = PT.add_node(parent_node, (new_g_plan,new_c_plan), cost,new_order)
                new_PT_nodes.put((neg_of_cost, new_node)) 
                # This is to ensure the PT node with higher cost will be added first to OPEN.


        # Put the (at most two) new PT nodes onto OPEN in non-increasing order of the cost.
        while not new_PT_nodes.empty():
            neg_of_cost, PT_node = new_PT_nodes.get()
            cost = -neg_of_cost
            OPEN.push(cost, PT_node)

    print('Total iterations = ',count,'OPEN empty?',OPEN.empty())
    return None


            