import cvxpy as cp
import numpy as np
from panav.env import trajectory_to_tube_obstacles, box_2d_center, trajectory_to_temp_obstacles,wp_to_tube_obstacle

'''
    Mixed-Integer Linear Programming based single agent motion planning algorithms, with dynamic obstacles.
'''
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
            constraints.append(tfin-t[1:]+ M * (1-tfin_active)>=0)

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
            constraints.append(t[-1] + TM * (1-T_end_alpha[i])>=lb)
            if np.isfinite(ub):
                constraints.append(t[-1] - TM * (1-T_end_alpha[i])<=ub)
            
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
                       obs_trajectories=[],
                     d=2,K=10,t0=0, T_end_constraints = None,
                     ignore_finished_agents=False,
                     goal_reach_eps = None,
                     tube_obs=[]
                     ):
    '''
        Use tube obstacles to model other moving agents instead of temporary obstacles,

        start,goal: start and goal coordinates.

        obs_trajectories: a list of tuples (times,xs), corresponding to the trajectory of another agent.

        T_end_constraints: a list of interval constraint on t[-1]. Typically appear in HybridGraph environment. 
                        - [Data format] T_end_constraint = [(lb_i,ub_i) for i=1,2,3,...,m], lb_i<ub_i are doubles.
                        - The constraint is disjunctive, meaning if lb_i<= t[-1] <=ub_i for some i=1,2,..., then the constraint is satisfied.

        ignore_finished_agents: whether ignore agents that have reached there goals or not.
        
        goal_reach_eps: distance tolerance for reaching the goal location. By default, goal_reach_eps is set to be the same as bloating_r.


        tube_obs: additional tube obstacles apart from those in obs_trajectories. 
                  A list in the format of [
                                           ([t_start,t_end],[p_start,p_end]),
                                           ([t_start,t_end],[p_start,p_end]),
                                           ...]
    '''
    # print("t0",t0)
    tubes= []
    for ta,pa in tube_obs:
        tubes.append(wp_to_tube_obstacle(ta[0],ta[1],
                                       pa[:,0],pa[:,1],bloating_r))
    for times,xs in obs_trajectories:
        tubes+=trajectory_to_tube_obstacles(times,xs,bloating_r)

    # Decision variables and standard constraints for the path planning problem.
    t, x, constraints = Standard_Tube_Var_Constraints(env,start,goal,vmax,bloating_r,tubes, d, K, t0, goal_reach_eps)

    # print(obs_trajectories)
    # Additional constraints
    M = 5 * np.max(np.abs(env.limits))
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
                    

    if T_end_constraints is not None:
        ls = np.array(T_end_constraints).flatten()
        ls = ls[np.isfinite(ls)]
        TM = 100 * np.max(ls)

        T_end_alpha = cp.Variable(len(T_end_constraints),boolean = True)
        for i,(lb,ub) in enumerate(T_end_constraints): 
            # print(i,lb,ub)   
            constraints.append(t[0,-1] + TM * (1-T_end_alpha[i])>=lb)
            if np.isfinite(ub):

                constraints.append(t[0,-1] - TM * (1-T_end_alpha[i])<=ub)
            
        constraints.append(cp.sum(T_end_alpha)>=1)

   

    prob = cp.Problem(cp.Minimize(t[0,-1]),constraints)

    # print('number of integer constraints:',count_interger_var(prob))
    prob.solve(solver='GUROBI',reoptimize =True) # The Gurobi solver proves to be more accurate and also faster.
    if t.value is not None:
        return unique_tx(t.value[0,:],x.value)
    else:
        return None
    
def count_interger_var(prob):
    '''
        prob: a Problem class object in cvxpy.
        Output: number of integer/binary variables in the problem.
    '''
    return sum([v.size for v in prob.variables() if v.boolean_idx or v.integer_idx])

def Standard_Tube_Var_Constraints(env, start, goal, vmax, bloating_r, tube_obs, d=2,K=10,t0=0, goal_reach_eps=None):
    if goal_reach_eps is None:
        goal_reach_eps = bloating_r

    x = cp.Variable((d, K+1))
    t = cp.Variable((1, K+1))

    constraints = []

    M = 5 * np.max(np.abs(env.limits))

    # Boundary constraints
    constraints.append(x <= np.array(env.limits)[:,-1].reshape(-1,1) - 2*bloating_r)
    constraints.append(x >= np.array(env.limits)[:,0].reshape(-1,1) + 2*bloating_r)

    # Start and goal constraints
    constraints.append(start == x[:,0])
    gl = box_2d_center(goal,np.ones(2) * goal_reach_eps)
    constraints.append(gl.A @ x[:,-1] <= gl.b)
   


    # Static obstacle constraints
    for O in env.obstacles:
        A, b= O.A,O.b

        H = A @ x-(b+ np.linalg.norm(A,axis=1) * bloating_r).reshape(-1,1) # Bloating radius

        alpha = cp.Variable((H.shape[0],K),boolean=True)

        constraints.append(H[:,1:] + M * (1-alpha)>=0)
        constraints.append(H[:,:-1] + M * (1-alpha)>=0)

        
        constraints.append(cp.sum(alpha,axis = 0)>=1)

    tx = cp.vstack([t,x])
    

    # print("tube_obs: ", len(tube_obs))
    
    for Ap,bp in tube_obs:
        Hp = Ap @ tx - (bp + np.linalg.norm(Ap,axis=1)*bloating_r).reshape(-1,1)
        
        alpha = cp.Variable((Hp.shape[0],Hp.shape[1]-1),boolean = True)
    
        constraints.append(Hp[:,1:] + M * (1-alpha)>=0)
        constraints.append(Hp[:,:-1] + M * (1-alpha)>=0)

        constraints.append(cp.sum(alpha,axis = 0)>=1)

    # Time positivity constraint
    constraints.append(t[0,0]==t0)
    constraints.append(t[0,1:]>=t[0,:-1])
    # Velocity constraints
    vb = vmax*(t[0,1:]-t[0,:-1])
    for i in range(d):
        diff = x[i,1:]-x[i,:-1]
        constraints.append(np.sqrt(2) * diff <= vb)
        constraints.append(- vb <= np.sqrt(2) * diff)
    
    return t,x, constraints

from panav.util import ParametericCurve
def track_ref_path(env, start, goal,ref_path, vmax, bloating_r, obstacle_trajectories,d, alpha = 0.5):
    K0 = ref_path.shape[1]-1
    ref_curve = ParametericCurve(ref_path)
    for K in range(K0,K0+5):
        print(K)
        ref_points = np.array([ref_curve(t) for t in np.linspace(0,1,K+1)]).T

        t,x,constraints = Standard_Tube_Var_Constraints(env, start, goal,vmax, bloating_r,obstacle_trajectories, d, K)

        tracking_loss = cp.norm(x-ref_points,'fro')
        prob = cp.Problem(cp.Minimize(alpha*tracking_loss+t[0,-1]),constraints)
        prob.solve()
        print(tracking_loss.value, t.value[0,-1])
        if t.value is not None:
            return t.value[0,:],x.value
    return None
def track_ref_path_v2(env, start, goal,ref_path, vmax, bloating_r, obstacle_trajectories,d, max_dev = 1.0):
    K0 = ref_path.shape[1]-1
    ref_curve = ParametericCurve(ref_path)
    for K in range(K0,K0+10):
        print(K)
        ref_points = np.array([ref_curve(t) for t in np.linspace(0,1,K+1)]).T

        t,x,constraints = Standard_Tube_Var_Constraints(env, start, goal,vmax, bloating_r,obstacle_trajectories, d, K)

        # constraints.append(cp.norm(x-ref_points,'fro')<=max_dev*(K+1))

        constraints.append(cp.norm(x-ref_points,axis=0)<=max_dev)
        prob = cp.Problem(cp.Minimize(t[0,-1]),constraints)
        prob.solve()
        if t.value is not None:
            return t.value[0,:],x.value
    return None
from panav.conflict import plan_obs_conflict
from panav.util import unique_tx
def auto_K_tube_planning(env, start, goal, vmax, bloating_r,
                       obs_trajectories=[],
                     d=2,t0=0, T_end_constraints = None,
                     ignore_finished_agents=False,
                     goal_reach_eps = None,
                     tube_obs=[],K_max = 10):
    for K in range(1,K_max+1):
        p = Tube_Planning(env, start, goal, vmax, bloating_r,
                       obs_trajectories,
                     d,K,t0, T_end_constraints,
                     ignore_finished_agents,
                     goal_reach_eps,
                     tube_obs)
        if p:
            return p
    
def lazy_optim(planner, env, start, goal, obstacle_trajectories,bloating_r, return_all=True):
    active = []
    m = sum([len(o[0])-1 for o in obstacle_trajectories])

    i = 0
    while i<=m:
        # print("num obstacle trajectories:{}/{}".format(len(active),m))
        p = planner(env,start,goal,active)
        if p is None:
            print('Problem becomes infeasible.')
            break

        p = unique_tx(*p)
        # conflicted_obs = plan_obs_conflict(p, obstacle_trajectories, bloating_r)
        conflicted_obs = plan_obs_conflict(p, obstacle_trajectories, bloating_r,segments_only=True,return_all=return_all)
        if not conflicted_obs:
            return p
        active+=conflicted_obs

    return None

def Efficient_Tube_Planning(env,start,goal,vmax,bloating_r,obstacle_trajectories,
                     d=2,t0=0, T_end_constraints = None,
                     ignore_finished_agents=False,
                     goal_reach_eps = None):
    planner = lambda env,start,goal,tube_obs: auto_K_tube_planning(env,start,goal,vmax,bloating_r,
                                                                   t0=t0,tube_obs=tube_obs,
                                                                   d=d,T_end_constraints=T_end_constraints,
                                                                   ignore_finished_agents=ignore_finished_agents,goal_reach_eps=goal_reach_eps)

    return lazy_optim(planner,env,start,goal,obstacle_trajectories,bloating_r)

def milestone_tracking(env,milestones,obstacle_trajectories,vmax,bloating_r):
    track_path = []
    cur = milestones[0]
    t0 = 0
    for i in range(len(milestones)-1):
        next_milestone = milestones[i+1]
        # print(t0)
        t,x = Efficient_Tube_Planning(env,start = cur, goal = next_milestone,
                                    vmax=vmax,bloating_r=bloating_r,
                                    obstacle_trajectories=obstacle_trajectories,t0 = t0)
        t0 = t[-1]
        # print(t0)
        cur = x[:,-1]
        track_path.append((t[:-1],x[:,:-1]))

        # print('Segment',i)
    track_path.append((t0,cur.reshape(-1,1)))

    ts = []
    xs = []
    for t,x in track_path:
        ts.append(t)
        xs.append(x)

    ts = np.hstack(ts)
    xs = np.hstack(xs)
    track_path = (ts,xs)
    return track_path