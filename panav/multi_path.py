import cvxpy as cp
import numpy as np
from panav.environment.utils import line_seg_to_obstacle,box_2d_center
from shapely import LineString,Point


def shortest_path(env,start,goal,K=2,d=2,
                  existing_paths = [],
                  bloating_r = 0.5,goal_reach_eps = 0.5,local_plan_radius = None):
    M = 100 * np.max(np.abs(env.limits))

    x = cp.Variable((d, K+2))

    constraints = []

    # Start and goal constraints
    # constraints+=[x[:,0] == start, x[:,-1] ==  goal]

    constraints.append(start == x[:,0])
    gl = box_2d_center(goal,np.ones(2) * goal_reach_eps)
    constraints.append(gl.A @ x[:,-1] <= gl.b)

    # Boundary constraints
    constraints.append(x <= np.array(env.limits)[:,-1].reshape(-1,1) - bloating_r)
    constraints.append(x >= np.array(env.limits)[:,0].reshape(-1,1) + bloating_r)

    # Static obstacle constraints
    obs = env.obstacles+[]

    # Separation constraint
    if len(existing_paths)>0:
        for p in existing_paths:
            # Ignore head and tail
            body = p[:,1:-1]
            for i in range(len(body)-1):
                obs.append(line_seg_to_obstacle(body[:,i],body[:,i+1],bloating_r))
            

    for O in obs:
        if local_plan_radius is not None:
            if O.verts.distance(LineString([start,goal]))>local_plan_radius:
            # if O.verts.distance(Point(start))>local_plan_radius:
            # to_obs = O.project(self.p) - self.p
            # if la.norm(to_obs)<=self.vmax*self.tau+1.5*self.bloating_r: 
                continue

        A, b= O.A,O.b

        H = A @ x-(b+ np.linalg.norm(A,axis=1) * bloating_r).reshape(-1,1) # Bloating radius

        alpha = cp.Variable((H.shape[0],H.shape[1]-1),boolean=True)
        
        constraints.append(H[:,1:] + M * (1-alpha)>=0)
        constraints.append(H[:,:-1] + M * (1-alpha)>=0)
        
        constraints.append(cp.sum(alpha,axis = 0)>=1)
    


    obj_func = cp.sum([cp.norm(x[:,i]-x[:,i+1]) for i in range(x.shape[1]-1)])
    prob = cp.Problem(cp.Minimize(obj_func),constraints)
    val = prob.solve(solver='GUROBI') # The Gurobi solver proves to be more accurate and also faster.
    # print(prob.status)
    return x.value, val

def explore_multi_path(env, start, goal):
    paths = []
    d = len(start)
    for i in range(10):
        # print(i,'num path')
        path = None
        for K in range(2,7):
            # print(K,'K')
            path, val = shortest_path(env, start, goal, K,d,paths)
            if path is not None:
                break
        
        if path is not None:
            paths.append(path)
            # print('path length',path.shape)
        else:
            break
    return paths