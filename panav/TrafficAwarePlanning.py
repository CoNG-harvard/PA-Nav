import networkx as nx
def traffic_aware_HG_plan(HG,consider_soft_traffic=False):
    '''
        HG: a hybrid graph class object.
        consider_soft_traffic: consider congestion on soft edges when planning. 
                By default, only congestion on hard edges are considered.

        Output: a list multi-agent graph paths on HG.
    '''
    
    ## One-by-one cost aware planning
    HG.__reset_traffic__()
    paths = []
    for s,g in zip(HG.start_nodes,HG.goal_nodes):
        
        # print(s,g)
        path = nx.shortest_path(HG,s,g,weight = "traffic_cost")
        # print(path)

        # Update the edge flow along the path
        for i in range(len(path)-1):
            p,q = path[i],path[i+1]
            HG.edges[p,q]['flow'] += 1

        paths.append(path)
        # Important: update graph traffic
        HG.update_traffic(consider_soft_traffic)

    # Reset HG's state before returning    
    HG.__reset_traffic__()
    
    return paths
