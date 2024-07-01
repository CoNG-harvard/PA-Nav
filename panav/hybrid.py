import networkx as nx
from itertools import product
from functools import partial

from panav.SAMP.archaic import Tube_Planning
# from panav.SAMP.solvers import Tube_Planning
from panav.tunnels import detect_tunnels, get_entry_exit
from panav.util import unique_tx

import numpy as np

class HybridGraph(nx.DiGraph):
    def __init__(self, env, agent_radius,d = 2,  # Path planning parameters are hard coded for now.
                                        vmax = 1.0) -> None:
        ''' 
            env: a panav.env.NavigationEnv object.
            agent_radius: double, the bloating radius of the agent. Used in tunnel detection.
        '''    
        super().__init__()
        
        self.vmax = vmax
        self.d = d
        self.agent_radius = agent_radius

        self.env = env
        self.open_spaces = {}

        self.start_nodes = []
        self.goal_nodes = []
        self.tunnel_nodes = []
        
        self.continuous_path_planner = partial(Tube_Planning, 
                                        env = self.env, 
                                        bloating_r = agent_radius, 
                                        obs_trajectories=[], 
                                        d = d,  # Path planning parameters are hard coded for now.
                                        K = 1,
                                        vmax = vmax)
        # self.continuous_path_planner = Tube_Planning(self.env,None,None,vmax=vmax,bloating_r=agent_radius,d=d,K_max = 10)
        
        self.tunnels = detect_tunnels(env,agent_radius)
        self.__construct_hybrid_graph__()

        # Initialize the traffic flow.
        self.__reset_traffic__()
       
        
    def __reset_traffic__(self):
        # Reset the traffic flow to all zero.
        for e in self.edges:
            self.edges[e]['flow']=0
            self.edges[e]['traffic_cost']=self.edges[e]['weight']

        for id in self.open_spaces.keys():
            self.open_spaces[id]['total_flow']=0


    def update_traffic(self,update_soft=False):
        """
        Update the traffic cost and open space flows based on current flows on the edges.
        """

        # Update the total flow on the open spaces
        for id in self.open_spaces.keys():
            open_space_nodes = self.open_spaces[id]['nodes']
            total_flow = 0
            for u in open_space_nodes:
                for v in open_space_nodes:
                    if (u,v) in self.edges:
                        total_flow+=self.edges[u,v]['flow']
            self.open_spaces[id]['total_flow'] = total_flow

        # Update the traffic cost on the edges
        for k,q in self.edges:
            if self.edges[k,q]['type']=='hard':
                                                  # This is also known as the contra-flow cost
                a = 1
                b = 5
                c = 10
                self.edges[k,q]['traffic_cost'] = (1+\
                                                   a * self.edges[q,k]['flow'] * self.edges[k,q]['flow']+\
                                                   b * self.edges[k,q]['flow']+\
                                                   c * self.edges[q,k]['flow'])\
                                                * self.edges[k,q]['weight'] 
                
            elif update_soft: 
                # Update soft edge if the input specifies they should be updated.
                open_space = self.get_open_space(k)
                total_flow = open_space['total_flow']
                self.edges[k,q]['traffic_cost'] = (1+(total_flow-self.edges[k,q]['flow'])\
                                                * (self.edges[k,q]['flow']+1))\
                                                * self.edges[k,q]['weight']
                


    def __construct_hybrid_graph__(self):
        
        # Every node has a region attribute: a panav.env.Region object.
        # Every node has a type attribute: type \in {'start','goal','tunnel'}. Tunnel endpoints are of type 'tunnel'
        # Every edge has a hardness attribute: type \in {'soft','hard'}.

        # Add hard edges + tunnel nodes
        for i,tunnel in enumerate(self.tunnels): 
            u = 2*i
            v = 2*i+1

            min_travel_time = np.linalg.norm(tunnel.end_points[0]-tunnel.end_points[1])/self.vmax 
            # Won't be modified

            self.add_node(u,type='tunnel',region = tunnel.end_regions[0],occupant = None,wait_offset = np.array([0.0,0.5]))
            self.add_node(v,type='tunnel',region = tunnel.end_regions[1],occupant = None, wait_offset = np.array([0.0,0.5]))

            self.add_edge(u,v,type='hard', weight = min_travel_time,
                          continuous_time = np.array([0, min_travel_time]), continuous_path = np.array(tunnel.end_points).T,
                          occupants = set())
            self.add_edge(v,u,type='hard', weight = min_travel_time, 
                          continuous_time = np.array([0, min_travel_time]), continuous_path = np.array(tunnel.end_points[::-1]).T,
                          occupants = set())
                         # continuous_time is at least min_travel_time on hard edges.

            self.tunnel_nodes.extend([u,v])

        starts,goals = self.env.start_regions, self.env.goal_regions
        # Add start nodes
        self.start_nodes = list(np.arange(self.number_of_nodes(),
                self.number_of_nodes()+len(starts)))
        self.add_nodes_from(self.start_nodes, type = 'start')
        nx.set_node_attributes(self,{n:{'region':region,'agent':agent} for agent,(n,region) in enumerate(zip(self.start_nodes,starts))})

        # Add goal nodes
        self.goal_nodes = list(np.arange(self.number_of_nodes(),
                self.number_of_nodes()+len(goals)))
        self.add_nodes_from(self.goal_nodes, type = 'goal')
        nx.set_node_attributes(self,{n:{'region':region,'agent':agent} for agent,(n,region) in enumerate(zip(self.goal_nodes,goals))})
        
        # Add soft edges
        G_soft = nx.DiGraph() # Temporary graph to store soft edges and determine how nodes are grouped by open spaces.
        legal_endpoint_types = [("tunnel","tunnel"),("start","tunnel"),("tunnel","goal"), ("start","goal")]
        for u,v in product(self.nodes,self.nodes):
            if u!=v and not (u,v) in G_soft.edges:
                u_type,v_type = self.nodes[u]['type'],self.nodes[v]['type']
                if (u_type,v_type) not in legal_endpoint_types:
                    # print("Skipping edge",u,v,"because",(u_type,v_type),"is not a possible edge type. Legal ones are",legal_endpoint_types)
                    continue                    
                
                if u_type =='start' and v_type=='goal' and self.nodes[u]['agent'] != self.nodes[v]['agent']:
                    # If it's a start to goal connection, consider soft edge establishment only when they are the start and goal for the same agent. 
                    # print('Skipping illegal start-goal connection for edge',u,v)
                    continue
                else:
                    pass

                ## Determine if the shortest path between u, v passes through any tunnels
             
                # Plan the shortest continuous path             
                # self.continuous_path_planner.start = self.node_loc(u)
                # self.continuous_path_planner.goal = self.node_loc(v)
                # path = self.continuous_path_planner.plan()
                path = self.continuous_path_planner(start = self.node_loc(u),goal = self.node_loc(v))

                if path is None:
                    print("Path not find. Consider increasing the K value. Skipping edge ",u,v)
                    continue
                else:
                    t,x = unique_tx(*path)
                    
                # See if the path passes through any tunnels
                through_some_tunnel = False
                for tunnel in self.tunnels:
                    ent, ex = get_entry_exit(tunnel,x)

                    if not(ent is None and ex is None):
                        # print(u,v,"Pass through tunnel at ", tunnel.region.centroid)
                        through_some_tunnel = True
                        break

                if not through_some_tunnel: # u-v does not pass through any tunnel.
                    G_soft.add_edge(u,v,type='soft', continuous_path = x, continuous_time= t, weight = np.max(t))
                    
        open_spaces_nodes = [c for c in nx.connected_components(nx.to_undirected(G_soft))]
        
        self.open_spaces = {i:{"nodes":c} for i,c in enumerate(open_spaces_nodes)}

        # Give all nodes in the graph an open space id
        for id, space in self.open_spaces.items():
            for u in space["nodes"]:
                self.nodes[u]['open_space_id'] = id

        # Add soft edges to G
        self.add_edges_from(G_soft.edges(data=True))    

    def node_loc(self, u):
        return np.asarray(self.nodes[u]['region'].centroid().coords[0])
    
    def get_open_space(self,u):
        return self.open_spaces[self.nodes[u]['open_space_id']]
    
    def node_locs(self):
        return [self.node_loc(s) for s in self.nodes]
    