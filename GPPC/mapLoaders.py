import numpy as np
'''
    GPPC 2D Map File format
    ------------------------
    First 4 lines:
        type octile
        height y
        width x
        map
        
    Followed by the grid map, each cell takes value in one of the following symbols.
    ------------------------
    . - passable terrain
    G - passable terrain
    @ - out of bounds
    O - out of bounds
    T - trees (unpassable)
    S - swamp (passable from regular terrain)
    W - water (traversable, but not passable from terrain)
    ------------------------
    Source: https://www.movingai.com/benchmarks/formats.html
'''
class TwoDMap:
    def __init__(self,path_to_file):
        '''
            self.map: a binary grid map. 
            map[x,y]==0 means the cell at (x,y) is free space.
            map[x,y]==1 means the cell at (x,y) is occupied(by obstacles).
            
            self.height: the height of the map.
            
            self.width: the width of the map.
        '''
        
        with open(path_to_file) as f:
            lines = [l.strip().split(' ') for l in f.readlines()]
        
        self.height = int(lines[1][1])
        self.width = int(lines[2][1])
        
        
        def line_to_bin(l):
            return [(c in ['.','G']) for c in l]

        self.map = np.array([line_to_bin(l[0]) for l in lines[4:]])