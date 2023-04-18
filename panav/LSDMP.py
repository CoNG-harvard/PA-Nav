import numpy as np
import cvxpy as cp

from panav.ORCA import VO, ORCA_Agent

def LSDMP(env,start,goal,neighbor_locs,safe_tau,
          vmax,bloating_r, d=2,K=10,t0=0,
          ORCA_protocol = 0):
    
    M = 100 * np.max(np.abs(env.limits))

    x = cp.Variable((d, K+1))
    t = cp.Variable(K+1)

    constraints = []

    # Boundary constraints
    constraints.append(x <= np.array(env.limits)[:,-1].reshape(-1,1) - 3*bloating_r)
    constraints.append(x >= np.array(env.limits)[:,0].reshape(-1,1) + 3* bloating_r)

    # Start and goal constraints
    constraints.append( x[:,0] == start)
    constraints.append(goal.A @ x[:,-1] <= goal.b)

    # Static obstacle constraints
    obs = [([],O) for O in env.obstacles]
    lb_active = []
    ub_active = []
    for duration,O in obs:
        A, b= O.A,O.b

        H = A @ x-(b+ np.linalg.norm(A,axis=1) * bloating_r).reshape(-1,1) # Bloating radius

        alpha = cp.Variable((H.shape[0],K),boolean=True)

        constraints.append(H[:,1:] + M * (1-alpha)>=0)
        constraints.append(H[:,:-1] + M * (1-alpha)>=0)
        constraints.append(cp.sum(alpha,axis = 0)>=1)

    # Time positivity constraint
    constraints.append(t[0]==t0)
    constraints.append(t[1:]>=t[:-1])

    # Max speed constraints
    vb = vmax*(t[1:]-t[:-1])
    for i in range(d):
        diff = x[i,1:]-x[i,:-1]
        constraints.append(np.sqrt(2) * diff <= vb)
        constraints.append(- vb <= np.sqrt(2) * diff)

    # ORCA constraints
    orca = ORCA_Agent(ORCA_protocol,safe_tau,
                    bloating_r,vmax,
                    init_p=start) 
    
    nb_orca = [ORCA_Agent(ORCA_protocol,safe_tau,
                    bloating_r,vmax,
                    init_p=loc) for loc in neighbor_locs]
    
    us,ns = orca.neighbor_constraints(nb_orca)
    constraints+=[(x[:,1]-x[:,0]- safe_tau * (orca.v_opt+u/2)) @ n >= 0 for u,n in zip(us,ns)]
    constraints.append(t[1]-t[0] == safe_tau) 

    prob = cp.Problem(cp.Minimize(t[-1]),constraints)

    prob.solve(solver='GUROBI')
    
    return t.value,x.value

