import cvxpy as cp
import numpy as np

def SA_MILP_Planning(env, start, goal, vmax, bloating_r,
                    temp_obstacles=[],\
                     d=2,K=10,t0=0):
    '''
        Single-agent path planning via mixed-integer programming.

        env: panav.env.NavigationEnv object. The path planning environment.
        
        start, goal: panav.env.Region object. Start and goal regions.
        
        vmax: maximal speed for all agents.

        bloating_r: the bloating radius of all agents, used for defining collision.

        temp_obstacles: a list of tuples in the format ([lb,ub], O). 
                        Temporary obstacles.
                        O is a panav.env.Region object.
                        Every element in the list means there is an obstacle O showing up in 
                        the environment during time interval [lb,ub].

        d: spatial dimension.

        K: the number of time steps to plan for.

        t0: initial time. By default it is 0.

        Output: t.value(shape = K+1), x.value(shape = (d,K+1))

    '''
    
    M = 100 * np.max(np.abs(env.limits))
    
    x = cp.Variable((d, K+1))
    t = cp.Variable(K+1)

    constraints = []

    # Boundary constraints
    constraints.append(x <= np.array(env.limits)[:,-1].reshape(-1,1) - 3*bloating_r)
    constraints.append(x >= np.array(env.limits)[:,0].reshape(-1,1) + 3* bloating_r)

    # Start and goal constraints
    constraints.append(start.A @ x[:,0] <= start.b)
    constraints.append(goal.A @ x[:,-1] <= goal.b)


    # Static obstacle constraints
    obs = [([],O) for O in env.obstacles] + temp_obstacles
    lb_active = []
    ub_active = []
    for duration,O in obs:
        A, b= O.A,O.b

        H = A @ x-(b+ np.linalg.norm(A,axis=1) * bloating_r).reshape(-1,1) # Bloating radius

        alpha = cp.Variable((H.shape[0],K),boolean=True)
        
        constraints.append(H[:,1:] + M * (1-alpha)>=0)
        constraints.append(H[:,:-1] + M * (1-alpha)>=0)
        
        if len(duration)==0:  # Permanent obstacles
            constraints.append(cp.sum(alpha,axis = 0)>=1)
        else:  # Temporary obstacles.        
            lb_active.append(cp.Variable(K,boolean=True))
            ub_active.append(cp.Variable(K,boolean=True))
            
            lb,ub = duration
    
            constraints.append(t[:-1]-ub+ M * (1-ub_active[-1])>=0)
            constraints.append(lb-t[1:]+ M * (1-lb_active[-1])>=0)

            constraints.append(cp.sum(alpha,axis = 0)+lb_active[-1]+ub_active[-1]>=1)
            

    # Time positivity constraint
    constraints.append(t[0]==t0)
    constraints.append(t[1:]>=t[:-1])

    # Velocity constraints
    vb = vmax*(t[1:]-t[:-1])
    for i in range(d):
        diff = x[i,1:]-x[i,:-1]
        constraints.append(np.sqrt(2) * diff <= vb)
        constraints.append(- vb <= np.sqrt(2) * diff)

    prob = cp.Problem(cp.Minimize(t[-1]),constraints)

    prob.solve(solver='GUROBI') # The Gurobi solver proves to be more accurate and also faster.
    return t.value,x.value