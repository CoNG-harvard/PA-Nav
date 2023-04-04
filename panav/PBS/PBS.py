from queue import PriorityQueue, deque
import networkx as nx
import numpy as np
from copy import deepcopy


from .conflict import find_conflict
from .HighLevelSearch import PriorityTree, SearchNodeContainer

def PBS(G, start_nodes,goal_nodes, edge_weights = None,\
         max_iter = 2000, metric = 'flowtime', search_type = 'depth_first'):
    '''
        Reference used for implementation:
        [Searching with Consistent Prioritization for Multi-Agent Path Finding, AAAI 2019]

        Inputs: 

                G: the graph.
                start_nodes: a tuple of starting positions in the graph. 
                goal_nodes: a tuple of goal positions in the graph.

                Our implementation here assumes the MAPF problem is labeled: that is the goals are pre-assigned to the agents, meaning agent k must go to goal k.
               
                edge_weights: a dictionationary {edge:cost for edge in G.edges}, specifying the travel costs along the edges.
                By default, the edge_weights are all set to 1.
                
                max_iter: the maximal number of iteration allowed to run CBS.
                    CBS does not have a way to determine infeasibility. 
                    To avoid infinite loops, we stop it when it exceeded an iteration threshold..
                
                metric: the MAPF objective to minimize. By default, it is flowtime. But metric.makespan can also be used if desired.
                
                search_type: either 'depth_first' or 'best_first'.
                    If 'depth_first', a stack will be used to contain the PT nodes to visit next.

                    Rigorously speaking, the depth_first option in PBS is a biased depth-first search. 
                    According to the original paper, after creating two children nodes from a parent node, the child with the lower cost will be visited first.
                    Despite this bias, the PT tree still grows in the direction of depth, rather than in breadth or the best OPEN node.
                    Therefore, we still call this option depth-first search.

                    If 'best_first', a PriorityQueue will be used instead.
        Output:
                solution, cost: an numpy array of M single-agent paths(M is the number of agents), conflict-free and with the smallest cost(flowtime by default).
                
                solution[i] is the path for agent i.
    '''
    #### Important convention on ordering tuples #####
    #  An ordering tuple (a1,a2) means agent a1 has higher priority than agent a2,
    #  meaning agent a2, along with all its priority descendents, should yield to a1.
    ##################################################

    if metric == 'flowtime':
        metric = flowtime
    elif metric == 'makespan':
        metric = makespan
    else:
        print('Metric {} is not supported. Please input "flowtime" or "makespan".'.format(metric))

    if edge_weights is None:
        edge_weights = {e:1 for e in G.edges} # Assume uniform weights if None is given.
        edge_weights.update({e[::-1]:1 for e in G.edges})

    nx.set_edge_attributes(G,edge_weights,'weight')
    hScore = dict(nx.shortest_path_length(G,weight = 'weight')) # The heuristic score used in SpaceTimeAStar

    # Initialization: 
    # Plan inidivual paths for agent agent without considering conflicts.
    # We simply call the standard networkx library.
    p = nx.shortest_path(G,weight = 'weight') # The [weight] argument here should be the key to weight values in the edge data dictionary.
    plan0 = [p[s][g] for s, g in zip(start_nodes, goal_nodes)]
    cost0 = metric(G,plan0,goal_nodes)

    PT = PriorityTree()
    ROOT = PT.add_node(None,plan0,metric(G,plan0,goal_nodes),[]) # Adding the root node. 

    OPEN = SearchNodeContainer(search_type)
    OPEN.push(cost0,ROOT)

    count = 0
    while not OPEN.empty() and count<=max_iter:
        count+=1 
        # To avoid infinite loops, we stop the algorithm when it exceeded an iteration threshold..

        cost, parent_node = OPEN.pop()
        solution = PT.get_solution(parent_node)

        # Look for the first conflict.
        conflict = find_conflict(solution,check_edge_conflicts = True)
        if conflict is None:
            return solution, cost
        else:
            a1, a2, c, t = conflict # c could either be a node or an edge.

        # Get orderings upto the current node.
        # print('Ancestors of ', parent_node)
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

                # print('agents_to_avoid',agents_to_avoid,'sorted_agents',sorted_agents,'agent_to_update',agent_to_update)

                node_constraints, edge_constraints, permanent_obstacles \
                = paths_to_constraints(G,[new_plan[av] for av in agents_to_avoid])

                # print('obstacles',[new_plan[av] for av in agents_to_avoid])
                # print('start and goal', start_nodes[agent_to_update], goal_nodes[agent_to_update])

                # print('Solving SAPF',node_constraints,edge_constraints,permanent_obstacles)
                result = SpaceTimeAStar(G,\
                        start_nodes[agent_to_update], goal_nodes[agent_to_update],\
                        node_constraints,edge_constraints,permanent_obstacles, hScore = hScore)
                if result is not None:
                    path,_ = result
                    new_plan[agent_to_update] = path 
                    agents_to_avoid.append(agent_to_update)

                    # print('result',result,'new_plan',new_plan)
                else:
                    success_update = False 
                    # print('Failure SAPF, agent',agent_to_update)
                    break
                    # The PT node is not created if any of the single-agent path is infeasible.

            if success_update:
                cost = metric(G,new_plan,goal_nodes) 
                neg_of_cost = -cost
                new_node = PT.add_node(parent_node, new_plan, cost,new_order)
                # print('Adding PT node', new_node, 'from parent_node', parent_node,'New ordering',PT.get_ordering(new_node),'Order added', new_order)
                new_PT_nodes.put((neg_of_cost, new_node)) # This is to ensure the PT node with higher cost will be removed first.

        # Put the (at most two) new PT nodes onto OPEN in non-increasing order of the cost.
        while not new_PT_nodes.empty():
            neg_of_cost, PT_node = new_PT_nodes.get()
            cost = -neg_of_cost
            OPEN.push(cost, PT_node)
    
    print('Total iterations = ',count,'OPEN empty?',OPEN.empty())
    return None

            