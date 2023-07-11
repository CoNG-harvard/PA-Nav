import cvxpy as cp
import numpy as np
from panav.env import trajectory_to_tube_obstacles, box_2d_center, trajectory_to_temp_obstacles

def SA_MILP_Planning(env, start, goal, vmax, bloating_r,
                    obs_trajectories = [],\
                     d=2,K=10,t0=0,T_end_constraints = None,ignore_finished_agents=False,temp_obstacles=[]):
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


    # Create temp_obstalces if needed
    if len(temp_obstacles)==0 and len(obs_trajectories)>0:
        for obs_t,obs_xs in obs_trajectories:
            temp_obstacles.extend(trajectory_to_temp_obstacles(obs_t,obs_xs,bloating_r))

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

    if not ignore_finished_agents:
        for obs_times,obs_xs in obs_trajectories:
            tfin = obs_times[-1]
            xfin = obs_xs[:,-1]
            O = box_2d_center(xfin,2*bloating_r) # The agent that has reached its goal, a static obstacle.
            A, b= O.A,O.b
            
            H = A @ x-(b+ np.linalg.norm(A,axis=1) * bloating_r).reshape(-1,1) # Bloating radius
            alpha = cp.Variable((H.shape[0],K),boolean=True)
            constraints.append(H[:,1:] + M * (1-alpha)>=0)
            constraints.append(H[:,:-1] + M * (1-alpha)>=0)

            tfin_active = cp.Variable(K,boolean=True)
            constraints.append(tfin-t[0,1:]+ M * (1-tfin_active)>=0)

            # Constrain disjunction
            constraints.append(cp.sum(alpha,axis = 0)+tfin_active>=1)    
    # T_end_constraint
    if T_end_constraints is not None:
        ls = np.array(T_end_constraints).flatten()
        ls = ls[np.isfinite(ls)]
        TM = 10 * np.max(ls)

        T_end_alpha = cp.Variable(len(T_end_constraints),boolean = True)
        for i,(lb,ub) in enumerate(T_end_constraints): 
            # print(i,lb,ub)   
            constraints.append(t[0,-1] + TM * (1-T_end_alpha[i])>=lb)
            if np.isfinite(ub):

                constraints.append(t[0,-1] - TM * (1-T_end_alpha[i])<=ub)
            
        constraints.append(cp.sum(T_end_alpha)>=1)

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
    if t.value is not None:
        return t.value,x.value
    else:
        return None

def Tube_Planning(env, start, goal, vmax, bloating_r,
                    obs_trajectories=[],\
                     d=2,K=10,t0=0, T_end_constraints = None,ignore_finished_agents=False):
    '''
        Use tube obstacles to model other moving agents instead of temporary obstacles,

        obs_trajectories: a list of tuples (times,xs), corresponding to the trajectory of another agent.

        T_end_constraints: a list of interval constraint on t[-1]. Typically appear in HybridGraph environment. 
                        - [Data format] T_end_constraint = [(lb_i,ub_i) for i=1,2,3,...,m], lb_i<ub_i are doubles.
                        - The constraint is disjunctive, meaning if lb_i<= t[-1] <=ub_i for some i=1,2,..., then the constraint is satisfied.

        ignore_finished_agents: whether ignore agents that have reached there goals or not.
    '''

    M = 5 * np.max(np.abs(env.limits))
        
    x = cp.Variable((d, K+1))
    t = cp.Variable((1,K+1))

    constraints = []

    # Boundary constraints
    constraints.append(x <= np.array(env.limits)[:,-1].reshape(-1,1) - 3*bloating_r)
    constraints.append(x >= np.array(env.limits)[:,0].reshape(-1,1) + 3* bloating_r)

    # Start and goal constraints
    constraints.append(start.A @ x[:,0] <= start.b)
    constraints.append(goal.A @ x[:,-1] <= goal.b)


    # Static obstacle constraints
    for O in env.obstacles:
        A, b= O.A,O.b

        H = A @ x-(b+ np.linalg.norm(A,axis=1) * bloating_r).reshape(-1,1) # Bloating radius

        alpha = cp.Variable((H.shape[0],K),boolean=True)

        constraints.append(H[:,1:] + M * (1-alpha)>=0)
        constraints.append(H[:,:-1] + M * (1-alpha)>=0)

        
        constraints.append(cp.sum(alpha,axis = 0)>=1)

    tx = cp.vstack([t,x])
    
    tube_obs = []
    for times,xs in obs_trajectories:
        tube_obs+=trajectory_to_tube_obstacles(times,xs,bloating_r)
     
    for Ap,bp in tube_obs:
        Hp = Ap @ tx - (bp + np.linalg.norm(Ap,axis=1)*bloating_r).reshape(-1,1)
        
        alpha = cp.Variable((Hp.shape[0],K),boolean=True)

        constraints.append(Hp[:,1:] + M * (1-alpha)>=0)
        constraints.append(Hp[:,:-1] + M * (1-alpha)>=0)

        constraints.append(cp.sum(alpha,axis = 0)>=1)

    if not ignore_finished_agents:
        for obs_times,obs_xs in obs_trajectories:
            tfin = obs_times[-1]
            xfin = obs_xs[:,-1]
            O = box_2d_center(xfin,2*bloating_r) # The agent that has reached its goal, a static obstacle.
            A, b= O.A,O.b
            
            H = A @ x-(b+ np.linalg.norm(A,axis=1) * bloating_r).reshape(-1,1) # Bloating radius
            alpha = cp.Variable((H.shape[0],K),boolean=True)
            constraints.append(H[:,1:] + M * (1-alpha)>=0)
            constraints.append(H[:,:-1] + M * (1-alpha)>=0)

            tfin_active = cp.Variable(K,boolean=True)
            constraints.append(tfin-t[0,1:]+ M * (1-tfin_active)>=0)

            # Constrain disjunction
            constraints.append(cp.sum(alpha,axis = 0)+tfin_active>=1)
                    

    # Time positivity constraint
    constraints.append(t[0,0]==t0)
    constraints.append(t[0,1:]>=t[0,:-1])

    # T_end_constraint
    if T_end_constraints is not None:
        ls = np.array(T_end_constraints).flatten()
        ls = ls[np.isfinite(ls)]
        TM = 10 * np.max(ls)

        T_end_alpha = cp.Variable(len(T_end_constraints),boolean = True)
        for i,(lb,ub) in enumerate(T_end_constraints): 
            # print(i,lb,ub)   
            constraints.append(t[0,-1] + TM * (1-T_end_alpha[i])>=lb)
            if np.isfinite(ub):

                constraints.append(t[0,-1] - TM * (1-T_end_alpha[i])<=ub)
            
        constraints.append(cp.sum(T_end_alpha)>=1)

    # Velocity constraints
    vb = vmax*(t[0,1:]-t[0,:-1])
    for i in range(d):
        diff = x[i,1:]-x[i,:-1]
        constraints.append(np.sqrt(2) * diff <= vb)
        constraints.append(- vb <= np.sqrt(2) * diff)

    prob = cp.Problem(cp.Minimize(t[0,-1]),constraints)

    
    prob.solve(solver='GUROBI') # The Gurobi solver proves to be more accurate and also faster.
    if t.value is not None:
        return t.value[0,:],x.value
    else:
        return None