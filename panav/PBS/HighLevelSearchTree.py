import networkx as nx
from queue import deque, PriorityQueue

class SearchNodeContainer:
    '''
        The type of the search node container determines the nature of the tree search algorithm.
        Pushing the search nodes onto a stack corresponds to the depth-first search.
        Pushing the search nodes onto a priority queue corresponds to the best-first search.
    '''
    def __init__(self, search_type = 'depth_first'):
        if search_type == 'depth_first':
            self.container = deque()
            self.push = lambda cost, node_ID: self.container.appendleft((cost,node_ID))
            self.pop = lambda : self.container.popleft()
            self.empty = lambda : len(self.container) == 0
        elif search_type == 'best_first':
            self.container = PriorityQueue()
            self.push = lambda cost, node_ID: self.container.put((cost,node_ID))
            self.pop = lambda : self.container.get()
            self.empty = self.container.empty
        else:
            print('Search type {} not supported.'.format(search_type))
            assert(False)

class HighLevelSearchTree(nx.DiGraph):
    def __init__(self):
        super().__init__()
        
    def get_solution(self,ID):
        return self.nodes[ID]['solution']
    
    def get_cost(self,ID):
        return self.nodes[ID]['cost']

        
class PriorityTree(HighLevelSearchTree):
    def __init__(self):
        super().__init__()
    
    def get_ordering(self,ID):
        '''
            Get the orderings stored at the current PT node and its ancestors.
        '''
        return [c for a in nx.ancestors(self,ID).union([ID]) for c in self.__get_ordering_at__(a)]
   
    def __get_ordering_at__(self,ID):
        
        '''
            Get the ordering stored at the current PT node only.
        '''
        return self.nodes[ID]['ordering']
    
    
    def add_node(self,parent_node,solution, cost, ordering):
        '''
            ordering: either [] or a list with a single two-tuple.
        '''
        new_node_ID = self.number_of_nodes()
        super().add_node(new_node_ID, solution=solution,cost=cost, ordering=ordering)
        
        if parent_node is not None:
            super().add_edge(parent_node,new_node_ID)
        
        return new_node_ID