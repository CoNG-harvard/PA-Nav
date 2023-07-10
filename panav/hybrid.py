import networkx as nx
from itertools import product
from functools import partial

from panav.SAMP import Tube_Planning
from panav.tunnels import detect_tunnels, get_entry_exit
from panav.util import unique_tx

import numpy as np

class HybridGraph(nx.Graph):
    def __init__(self, env, agent_radius) -> None:
        super().__init__()
        # env: a panav.env.NavigationEnv object.
        # agent_radius: double, the bloating radius of the agent. Used in tunnel detection.
        
        self.env = env
        self.open_spaces = []

        self.start_nodes = set()
        self.goal_nodes = set()
        self.tunnel_nodes = set()
        
        self.continuous_path_planner = partial(Tube_Planning, 
                                        env = self.env, 
                                        bloating_r = agent_radius, 
                                        obs_trajectories=[], 
                                        d = 2,  # Path planning parameters are hard coded for now.
                                        K = 5,
                                            vmax = 1.0)
        
        self.tunnels = detect_tunnels(env,agent_radius)
        self.__construct_hybrid_graph__()

       
        
    def __construct_hybrid_graph__(self):
        
        # Every node has a region attribute: a panav.env.Region object. To get its location, call self.nodes[s]['region'].centroid()
        # Every node has a type attribute: type \in {'start','goal','tunnel'}. Tunnel endpoints are of type 'tunnel'
        # Every edge has a hardness attribute: type \in {'soft','hard'}.

        # Add hard edges + tunnel nodes
        for i,tunnel in enumerate(self.tunnels): 
            u = 2*i
            v = 2*i+1

            self.add_node(u,type='tunnel',region = tunnel.end_regions[0])
            self.add_node(v,type='tunnel',region = tunnel.end_regions[1])

            self.add_edge(u,v,type='hard')
            self.tunnel_nodes.update({u,v})

        starts,goals = self.env.starts, self.env.goals
        # Add start nodes
        self.start_nodes = set(np.arange(self.number_of_nodes(),
                self.number_of_nodes()+len(starts)))
        self.add_nodes_from(self.start_nodes, type = 'start')
        nx.set_node_attributes(self,{n:{'region':region,'agent':agent} for agent,(n,region) in enumerate(zip(self.start_nodes,starts))})

        # Add goal nodes
        self.goal_nodes = set(np.arange(self.number_of_nodes(),
                self.number_of_nodes()+len(goals)))
        self.add_nodes_from(self.goal_nodes, type = 'goal')
        nx.set_node_attributes(self,{n:{'region':region,'agent':agent} for agent,(n,region) in enumerate(zip(self.goal_nodes,goals))})
        
        # Add soft edges
        G_soft = nx.Graph() # Temporary graph to store soft edges and determine how nodes are grouped by open spaces.
        for u,v in product(self.nodes,self.nodes):
            if u<v and not (u,v) in G_soft.edges:

                # Eliminate start to start, goal to goal connections. 
                if self.nodes[u]['type'] in ['start','goal'] and self.nodes[v]['type'] in ['start','goal']:
                    if self.nodes[u]['type'] != self.nodes[v]['type'] and self.nodes[u]['agent'] == self.nodes[v]['agent']:
                        # If it's a start to goal connection, consider soft edge establishment only when they are the start and goal for the same agent. 
                        # print('Checking start to goal connection for agent', self.nodes[u]['agent'])
                        pass
                    else:
                        # print('Skipping irrelevant connection')
                        continue

                ## Determine if the shortest path between u, v passes through any tunnels
             
                # Plan the shortest continuous path             
                path = self.continuous_path_planner(start = self.nodes[u]['region'],goal = self.nodes[v]['region'])

                if path is None:
                    print("Path not find. Consider increasing the K value. Skipping edge ",u,v)
                    continue
                else:
                    _,x = unique_tx(*path)
                    
                # See if the path passes through any tunnels
                through_some_tunnel = False
                for tunnel in self.tunnels:
                    ent, ex = get_entry_exit(tunnel,x)

                    if not(ent is None and ex is None):
                        # print(u,v,"Pass through tunnel at ", tunnel.region.centroid)
                        through_some_tunnel = True
                        break

                if not through_some_tunnel: # u-v does not pass through any tunnel.
                    # print(u,v,'Does not pass through any tunnel')
                    G_soft.add_edge(u,v,type='soft', continuous_path = x)
        
        self.open_spaces = [c for c in nx.connected_components(G_soft)]
        # Give all nodes in the graph an open space id
        for id, c in enumerate(self.open_spaces):
            for s in c:
                self.nodes[s]['open_space_id'] = id

        # Add soft edges to G
        self.add_edges_from(G_soft.edges(data=True))    
                    