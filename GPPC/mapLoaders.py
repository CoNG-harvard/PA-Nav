import numpy as np
import pandas as pd

'''
GPPC 2D Scenarior File Format:

A table, each row describes a single-agent motion planning problem, and the optimal length for that problem.

The column names are(from left to right): 'Bucket','map','width','height','start_x','start_y','goal_x','goal_y','opt_length'.

Notes from GPPC website(https://www.movingai.com/benchmarks/formats.html):

The optimal path length is assuming sqrt(2) diagonal costs.
The optimal path length assumes agents cannot cut corners through walls
If the map height/width do not match the file, it should be scaled to that size
(0, 0) is in the upper left corner of the maps
Technically a single scenario file can have problems from many different maps, but currently every scenario only contains problems from a single map

'''

def load_scenario(path_to_file):

    with open(path_to_file) as f:
            lines = [l.strip().split('\t') for l in f.readlines()]

    df = pd.DataFrame(lines)
    df.columns = ['Bucket','map','width','height','start_x','start_y','goal_x','goal_y','opt_length']
    df = df.drop(0)
    df = df.astype({c:t for c,t in zip(df.columns,['int','string']+['float']*7)})
    
    return df


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
def load_map(path_to_file):
        '''
            output: map, a binary grid map. 
            map[x,y]==0 means the cell at (x,y) is free space.
            map[x,y]==1 means the cell at (x,y) is occupied(by obstacles).
            
        '''
        def line_to_bin(l):
            return [(c in ['.','G']) for c in l]

        with open(path_to_file) as f:
            lines = [l.strip().split(' ') for l in f.readlines()]        
        
        map = np.array([line_to_bin(l[0]) for l in lines[4:]])

        return map