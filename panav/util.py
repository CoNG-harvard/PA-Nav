import numpy as np
import itertools


def unit_cube(d):
    '''
        Return the vertices of a d-dimensional unit cube.
        Output: shape = (2^d,d)
    '''
    one = np.ones(d)
    cube_vertices = [np.sum(unit_vec,axis = 1)/d for unit_vec in itertools.product(*([one,-one] for _ in range(d)))]
    return cube_vertices

def unique_tx(t,x):
    '''
        t: shape = K + 1
        x: shape = (d,K+1)
    '''
    times,xs = np.array(t),np.array(x)

    unique_index = []
    for i in range(len(times)-1):
        if np.abs(times[i]-times[i+1])>1e-5:
            unique_index.append(i)
            
    unique_index.append(i+1)
    times = times[unique_index]
    xs = xs[:,unique_index]
    return times, xs

def interpolate_positions(t,x,dt):
    '''
        t: shape = K + 1
        x: shape = (d,K+1)
    '''
    pos = []
    times = []
    # print(x.shape)

    for i in range(len(t)-1):
        n = int((t[i+1]-t[i])/dt)
        pos.append(np.linspace(x[:,i],x[:,i+1],n).T)
        times.append(np.linspace(t[i],t[i+1],n))
    # print(pos,x)
    return  np.hstack(times),np.hstack(pos)