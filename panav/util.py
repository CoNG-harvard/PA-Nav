import numpy as np
import itertools

from numpy.linalg import norm


def flowtime(plan):
    for p in plan:
        if p is None:
            print(p)
    return np.sum([t[-1] for t,x in plan])
def makespan(plan):
    return np.max([t[-1] for t,x in plan])
def count_interger_var(prob):
    '''
        prob: a Problem class object in cvxpy.
        Output: number of integer/binary variables in the problem.
    '''
    return sum([v.size for v in prob.variables() if v.boolean_idx or v.integer_idx])
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

class ParametericCurve:
    '''
        The purpose of this class is to represent the piecewise linear continuous curve C:[0,1]-> X.

        Given waypoints w0,w1,...,wn, there is C(0) = w0, C(1) = wn.

        We want the ability to efficiently compute C(t) for any t in [0,1].
    '''
    def __init__(self, waypoints):
        '''
            waypoints: shape = (space dimension, number of waypoints)
        '''
        self.p = waypoints

        K = waypoints.shape[1]

        self.l = np.array([norm(self.p[:,i]-self.p[:,i-1]) for i in range(1,K)])
        # The lengths of the segments.

        self.L = sum(self.l)
        self.l /= self.L # Normalize the segment lengths

        self.cumu_l = np.array(self.l)
        for i in range(1,len(self.cumu_l)):
            self.cumu_l[i] += self.cumu_l[i-1]
        
    def __call__(self,t):
        return self.at(t) 
    def at(self,t):
        if t<0 or t>1:
            return None
        
        # Determine which segment does t belong to
        seg = 0
        while seg<len(self.cumu_l) and t>=self.cumu_l[seg]:
            seg += 1
        
        if seg == len(self.cumu_l):
            return self.p[:,-1]
        
        cur_seg_length = self.l[seg]

        overshoot = t - self.cumu_l[seg-1] if seg>0 else t

        alpha =  overshoot/cur_seg_length

        return (1-alpha) * self.p[:,seg] + alpha * self.p[:,seg+1]
    