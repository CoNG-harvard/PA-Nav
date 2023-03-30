import cvxpy as cp
import numpy as np

def SA_MILP_Planning(env, start, goal, vmax, bloating_r\
                     ,d=2,K=10,t0=0):
    '''
        Single-agent path planning via mixed-integer programming.
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

    # Velocity constraints
    vb = vmax*(t[1:]-t[:-1])
    for i in range(d):
        diff = x[i,1:]-x[i,:-1]
        constraints.append(np.sqrt(2) * diff <= vb)
        constraints.append(- vb <= np.sqrt(2) * diff)


    # Permanent static obstacle constraints
    for O in env.obstacles:
        A, b= O.A,O.b
        # print(A,b,O.vertices())

        H = A @ x-(b+ np.linalg.norm(A) * bloating_r).reshape(-1,1) # Bloating radius

        alpha = cp.Variable((H.shape[0],K),boolean=True)

        constraints.append(H[:,1:] + M * (1-alpha)>=0)
        constraints.append(H[:,:-1] + M * (1-alpha)>=0)

        constraints.append(cp.sum(alpha,axis = 0)>=1)


    # Time positivity constraint
    constraints.append(t[0]==t0)
    constraints.append(t[1:]>=t[:-1])

    prob = cp.Problem(cp.Minimize(t[-1]),constraints)

    prob.solve()
    return t.value,x.value