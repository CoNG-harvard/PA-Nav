import faiss
from networkx import DiGraph
from panav.RRT.utils import *

'''
    The RRT algorithm in LaValle 98. Static obstacles only.
'''

def RRT_plan(env,x_start,bloating_r,max_iter = 100000,eta = 0.1):
    def backtrack(i,T):
        plan = []
        pred = list(T.predecessors(i))
        while i is not None:
            plan.append(T.nodes[i]['loc'])
            if len(pred)>0:
                i = pred[0]
                pred = list(T.predecessors(i))
            else:
                i = None

        return plan[::-1]
    # Simulate the RRT loop
    T = RRT(x_start)
    for _ in range(max_iter):
        x_rand = uniform_rand_loc(env)
        i_nearest,x_nearest = T.nearest(x_rand)
        x_new = binary_line_search(env,x_nearest,x_rand,bloating_r,
                                eta = eta)
        # Setting eta = None seems to result in th fast marching tree
        T.add_loc(i_nearest,x_new)
        
        # Stop when one feasible solution is found
        if env.goals[0].vertices().contains(shapely.Point(x_new)):
            return backtrack(T.number_of_nodes()-1,T),T 
    else:
        return [], T

def draw_rrt(ax,T):
    for e in T.edges():
        u,v = e
        seg = np.vstack([T.nodes[u]['loc'],T.nodes[v]['loc']])
        ax.plot(seg[:,0],seg[:,1],color="g")

class RRT(DiGraph):
    def __init__(self,x_init):
        super().__init__()
        self.add_node(0,loc = x_init,tol_cost = 0)
        self.d = len(x_init)
        self.nn_index = faiss.IndexFlatL2(self.d) # The data structure for efficient nearest neighbor computation.
        self.nn_index.add(x_init.reshape(-1,self.d))
    def nearest(self,xq):
        xq = np.array(xq).reshape(1,self.d)
        _, nn_i = self.nn_index.search(xq,1) 

        nn_i = nn_i.item() # Get the scalar in the singleton array
        return nn_i,self.nodes[nn_i]['loc']
    
    def add_loc(self,parent,x_new):
        x_parent = self.nodes[parent]['loc']

        self.add_node(self.number_of_nodes(),loc = x_new,
                      tol_cost = self.nodes[parent]['tol_cost']+la.norm(x_parent-x_new))

        self.add_edge(parent,self.number_of_nodes()-1)
        
        self.nn_index.add(x_new.reshape(-1,self.d))
