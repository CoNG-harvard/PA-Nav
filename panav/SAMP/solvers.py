import cvxpy as cp
import numpy as np
from panav.environment.utils import trajectory_to_tube_obstacles, box_2d_center, wp_to_tube_obstacle,trajectory_to_temp_obstacles
from panav.util import unique_tx, count_interger_var
from panav.conflict import plan_obs_conflict
class SAMP_Base:
    '''
        The base class for Single Agent Motion Planning(SAMP) solvers.
    '''
    def __init__(self,env,start,goal,bloating_r = 0.5,vmax=1.0,K_max=10) -> None:
        self.env = env
        self.start = start
        self.goal = goal
        self.bloating_r = bloating_r
        self.vmax = vmax
        self.K = 1
        self.K_max = K_max
        self.K_INC  = 1
    
    def plan(self,active_obstacles=[],obstacle_trajectories=[],lazy = True):
        if not lazy:
            return self.plan_plain(active_obstacles,obstacle_trajectories)
        else:
            return self.lazy_plan(obstacle_trajectories)

      
    def lazy_plan(self, obstacle_trajectories=[], return_all=True):
        '''
            Assuming the obstacle trajectories can be broken down into active segments.

            Lazily adding the active segments into the problem, resolve until the problem becomes feasible.

            In many cases, leads to faster solution due to less integer constraints.
        '''
        active_obs = []
        m = sum([len(o[0])-1 for o in obstacle_trajectories])

        i = 0
        K_min = 1
        while i<=m:
            # print("num obstacle trajectories:{}/{}".format(len(active_obs),m))
            p = self.plan_plain(active_obstacles=active_obs,K_min=K_min)
           
            if p is None:
                print('Problem becomes infeasible.')
                break

            p = unique_tx(*p)
            # conflicted_obs = plan_obs_conflict(p, obstacle_trajectories, bloating_r)
            conflicted_obs = plan_obs_conflict(p, obstacle_trajectories, self.bloating_r, 
                                               segments_only=True,return_all=return_all)
            if not conflicted_obs:
                return p
            
            K_min = len(p[0])-1 # len(p[0])-1 is the latest K value.
            # print("K_min",K_min)

            active_obs+=conflicted_obs
        return None
    
    def plan_plain(self,active_obstacles=[],obstacle_trajectories = [],K_min = 1):
        
        self.K = K_min
        while self.K<=self.K_max:
            p = self.plan_core(active_obstacles,obstacle_trajectories)
            # print(K,p)
            if p:
                self.K = 1 # Reset self.K 
                return p
            else: 
                self.K += self.K_INC

        # Solution not found even after exhausting all possible K.
        self.K = 1 # Reset self.K
        return None 
    
    def plan_core(self,active_obstacles=[],obstacles_trajectories=[]):
            '''
                active_obstacles: typically individual segments of moving obstacles.
                obstacle_trajectories: typically entire trajectories of moving obstacles.
            '''
            raise NotImplementedError

class Simple_MILP_Planning(SAMP_Base):
    def __init__(self, env, start, goal, vmax = 1.0,bloating_r=0.5, d=2,K_max=10,t0=0,T_end_constraints = None,ignore_finished_agents=False) -> None:
        '''
            Single-agent path planning via mixed-integer programming.

            env: panav.environments.env.NavigationEnv object. The path planning environment.
            
            start, goal: panav.env.Region object. Start and goal regions.
            
            vmax: maximal speed for all agents.

            bloating_r: the bloating radius of all agents, used for defining collision.

            d: spatial dimension.

            K: the number of time steps to plan for.

            t0: initial time. By default it is 0.


        '''
        super().__init__(env, start, goal, bloating_r,vmax,K_max)
 
        self.d = d
        self.t0 = t0
        self.T_end_constraints = T_end_constraints
        self.ignore_finished_agents = ignore_finished_agents

    
    def plan(self,active_obstacles=[],obstacle_trajectories=[]):
        return self.plan_plain(active_obstacles,obstacle_trajectories)


        
    def plan_core(self,active_obstacles=[],obstacle_trajectories=[],solve_inplace = True):
        '''
         active_obstacles: a list of tuples in the format ([lb,ub], O). 
                            Temporary obstacles.
                            O is a panav.env.region.Region object.
                            Every element in the list means there is an obstacle O showing up in 
                            the environment during time interval [lb,ub].

        obstacle_trajectories: a list of tuples (times,xs), corresponding to the trajectory of another agent.

        Output: t.value(shape = K+1), x.value(shape = (d,K+1))

        '''
        
        M = 5 * np.max(np.abs(self.env.limits))
        
        x = cp.Variable((self.d, self.K+1))
        t = cp.Variable(self.K+1)

        constraints = []

        # Boundary constraints
        constraints.append(x <= np.array(self.env.limits)[:,-1].reshape(-1,1) - 1* self.bloating_r)
        constraints.append(x >= np.array(self.env.limits)[:,0].reshape(-1,1) + 1* self.bloating_r)

        # Start and goal constraints
        constraints.append(self.start == x[:,0])
        gl = box_2d_center(self.goal,np.ones(2) * self.bloating_r)
        constraints.append(gl.A @ x[:,-1] <= gl.b)


        obs = []
        for o in active_obstacles:
            obs.append(o)

        # Create temp_obstalces if needed
        for obs_t,obs_xs in obstacle_trajectories:
            obs+=trajectory_to_temp_obstacles(obs_t,obs_xs,self.bloating_r)

        # Static obstacle constraints
        obs = [([],O) for O in self.env.obstacles] + obs
        lb_active = []
        ub_active = []
        for duration,O in obs:
            A, b= O.A,O.b

            H = A @ x-(b+ np.linalg.norm(A,axis=1) * self.bloating_r).reshape(-1,1) # Bloating radius

            alpha = cp.Variable((H.shape[0],self.K),boolean=True)
            
            constraints.append(H[:,1:] + M * (1-alpha)>=0)
            constraints.append(H[:,:-1] + M * (1-alpha)>=0)
            
            if len(duration)==0:  # Permanent obstacles
                constraints.append(cp.sum(alpha,axis = 0)>=1)
            else:  # Temporary obstacles.        
                lb_active.append(cp.Variable(self.K,boolean=True))
                ub_active.append(cp.Variable(self.K,boolean=True))
                
                lb,ub = duration
        
                constraints.append(t[:-1]-ub+ M * (1-ub_active[-1])>=0)
                constraints.append(lb-t[1:]+ M * (1-lb_active[-1])>=0)

                constraints.append(cp.sum(alpha,axis = 0)+lb_active[-1]+ub_active[-1]>=1)

        if not self.ignore_finished_agents:
            for obs_times,obs_xs in obstacle_trajectories:
                tfin = obs_times[-1]
                xfin = obs_xs[:,-1]
                O = box_2d_center(xfin,4 * self.bloating_r) # The agent that has reached its goal, a static obstacle.
                A, b= O.A,O.b
                
                H = A @ x-(b+ np.linalg.norm(A,axis=1) * self.bloating_r).reshape(-1,1) # Bloating radius
                alpha = cp.Variable((H.shape[0],self.K),boolean=True)
                constraints.append(H[:,1:] + M * (1-alpha)>=0)
                constraints.append(H[:,:-1] + M * (1-alpha)>=0)

                tfin_active = cp.Variable(self.K,boolean=True)
                constraints.append(tfin-t[1:]+ M * (1-tfin_active)>=0)

                # Constrain disjunction
                constraints.append(cp.sum(alpha,axis = 0)+tfin_active>=1)    
        # T_end_constraint
        if self.T_end_constraints is not None:
            ls = np.array(self.T_end_constraints).flatten()
            ls = ls[np.isfinite(ls)]
            TM = 10 * np.max(ls)

            T_end_alpha = cp.Variable(len(self.T_end_constraints),boolean = True)
            for i,(lb,ub) in enumerate(self.T_end_constraints): 
                # print(i,lb,ub)   
                constraints.append(t[-1] + TM * (1-T_end_alpha[i])>=lb)
                if np.isfinite(ub):
                    constraints.append(t[-1] - TM * (1-T_end_alpha[i])<=ub)
                
            constraints.append(cp.sum(T_end_alpha)>=1)

        # Time positivity constraint
        constraints.append(t[0]==self.t0)
        constraints.append(t[1:]>=t[:-1])

        # Velocity constraints
        vb = self.vmax*(t[1:]-t[:-1])
        for i in range(self.d):
            diff = x[i,1:]-x[i,:-1]
            constraints.append(np.sqrt(2) * diff <= vb)
            constraints.append(- vb <= np.sqrt(2) * diff)

        prob = cp.Problem(cp.Minimize(t[-1]),constraints)

        if not solve_inplace:
            return t,x,constraints,prob
        else:
            # print('number of integer constraints:',count_interger_var(prob))
            prob.solve(solver='GUROBI',TimeLimit = 100) # The Gurobi solver proves to be more accurate and also faster.
            if prob.status != 'optimal':
                return None
            else:
                out = unique_tx(t.value,x.value)
                del prob
                return out
            
class Tube_Planning(SAMP_Base):
    def __init__(self, env, start, goal, 
                 vmax = 1, bloating_r = 0.5,d = 2, t0 = 0,K_max = 10,
                 T_end_constraints = None,ignore_finished_agents = False,goal_reach_eps = None) -> None:
        
        '''
        Use tube obstacles to model other moving agents instead of temporary obstacles,

        start,goal: start and goal coordinates.

     
        T_end_constraints: a list of interval constraint on t[-1]. Typically appear in HybridGraph environment. 
                        - [Data format] T_end_constraint = [(lb_i,ub_i) for i=1,2,3,...,m], lb_i<ub_i are doubles.
                        - The constraint is disjunctive, meaning if lb_i<= t[-1] <=ub_i for some i=1,2,..., then the constraint is satisfied.

        ignore_finished_agents: whether ignore agents that have reached there goals or not.
        
        goal_reach_eps: distance tolerance for reaching the goal location. By default, goal_reach_eps is set to be the same as bloating_r.

        '''
        
        super().__init__(env,start,goal,bloating_r,vmax,K_max)

         
        
        self.d = d
        self.t0 = t0

        self.T_end_constraints = T_end_constraints
        self.ignore_finished_agents = ignore_finished_agents
        self.goal_reach_eps = goal_reach_eps if goal_reach_eps else bloating_r



    
    def plan_core(self,active_obstacles=[],obstacle_trajectories=[],solve_inplace = True):
        '''
            active_obstacles: obstacles in addition to those in obstacle_trajectories.
                              A list in the format of [
                                           ([t_start,t_end],[p_start,p_end]),
                                           ([t_start,t_end],[p_start,p_end]),
                                           ...]

            obs_trajectories: a list of tuples (times,xs), corresponding to the trajectory of another agent.

            solve_in_place: False -> Return the MILP problem, including the variables, constraints, and prob objects, without calling prob.solve().
                            True -> Return the result of prob.solve().
        '''
        tubes = []
        for ta,pa in active_obstacles:
            tubes.append(wp_to_tube_obstacle(ta[0],ta[1],
                                        pa[:,0],pa[:,1],self.bloating_r))
        for times,xs in obstacle_trajectories:
            tubes+=trajectory_to_tube_obstacles(times,xs,self.bloating_r)

        # Decision variables and standard constraints for the path planning problem.
        t, x, constraints = self.Standard_Tube_Var_Constraints(self.start,self.goal,tubes,self.K)

        # Additional constraints
        M = 5 * np.max(np.abs(self.env.limits))
        if not self.ignore_finished_agents:
            for obs_times,obs_xs in obstacle_trajectories:
                tfin = obs_times[-1]
                xfin = obs_xs[:,-1]
                O = box_2d_center(xfin,4 * self.bloating_r) # The agent that has reached its goal, a static obstacle.
                A, b= O.A,O.b
                
                H = A @ x-(b+ np.linalg.norm(A,axis=1) *self.bloating_r).reshape(-1,1) # Bloating radius
                alpha = cp.Variable((H.shape[0],self.K),boolean=True)
                constraints.append(H[:,1:] + M * (1-alpha)>=0)
                constraints.append(H[:,:-1] + M * (1-alpha)>=0)

                tfin_active = cp.Variable(self.K,boolean=True)
                constraints.append(tfin-t[0,1:]+ M * (1-tfin_active)>=0)

                # Constrain disjunction
                constraints.append(cp.sum(alpha,axis = 0)+tfin_active>=1)
                        

        if self.T_end_constraints is not None:
            ls = np.array(self.T_end_constraints).flatten()
            ls = ls[np.isfinite(ls)]
            TM = 100 * np.max(ls)

            T_end_alpha = cp.Variable(len(self.T_end_constraints),boolean = True)
            for i,(lb,ub) in enumerate(self.T_end_constraints): 
                # print(i,lb,ub)   
                constraints.append(t[0,-1] + TM * (1-T_end_alpha[i])>=lb)
                if np.isfinite(ub):

                    constraints.append(t[0,-1] - TM * (1-T_end_alpha[i])<=ub)
                
            constraints.append(cp.sum(T_end_alpha)>=1)

        prob = cp.Problem(cp.Minimize(t[0,-1]),constraints)

        if not solve_inplace:
            return t,x,constraints,prob
        else:
            # print('number of integer constraints:',count_interger_var(prob))
            prob.solve(solver='GUROBI',TimeLimit = 100) # The Gurobi solver proves to be more accurate and also faster.
            if prob.status == 'optimal':
                out = unique_tx(t.value[0,:],x.value)
                del prob
                return out
            else:
                return None
      
        
    def Standard_Tube_Var_Constraints(self,start,goal,tube_obs,K):
        
        '''
        tube_obs: A list in the format of [
                                           ([t_start,t_end],[p_start,p_end]),
                                           ([t_start,t_end],[p_start,p_end]),
                                           ...]
        '''

        x = cp.Variable((self.d, K+1))
        t = cp.Variable((1, K+1))

        constraints = []

        M = 5 * np.max(np.abs(self.env.limits))

        # Boundary constraints
        constraints.append(x <= np.array(self.env.limits)[:,-1].reshape(-1,1) - 2*self.bloating_r)
        constraints.append(x >= np.array(self.env.limits)[:,0].reshape(-1,1) + 2*self.bloating_r)

        # Start and goal constraints
        constraints.append(start == x[:,0])
        gl = box_2d_center(goal,np.ones(2) * self.goal_reach_eps)
        constraints.append(gl.A @ x[:,-1] <= gl.b)
    


        # Static obstacle constraints
        for O in self.env.obstacles:
            A, b= O.A,O.b

            H = A @ x-(b+ np.linalg.norm(A,axis=1) * self.bloating_r).reshape(-1,1) # Bloating radius

            alpha = cp.Variable((H.shape[0],K),boolean=True)

            constraints.append(H[:,1:] + M * (1-alpha)>=0)
            constraints.append(H[:,:-1] + M * (1-alpha)>=0)

            
            constraints.append(cp.sum(alpha,axis = 0)>=1)

        tx = cp.vstack([t,x])
        

        # print("tube_obs: ", len(tube_obs))
        
        for Ap,bp in tube_obs:
            Hp = Ap @ tx - (bp + np.linalg.norm(Ap,axis=1)*self.bloating_r).reshape(-1,1)
            
            alpha = cp.Variable((Hp.shape[0],Hp.shape[1]-1),boolean = True)
        
            constraints.append(Hp[:,1:] + M * (1-alpha)>=0)
            constraints.append(Hp[:,:-1] + M * (1-alpha)>=0)

            constraints.append(cp.sum(alpha,axis = 0)>=1)

        # Time positivity constraint
        constraints.append(t[0,0]==self.t0)
        constraints.append(t[0,1:]>=t[0,:-1])
        # Velocity constraints
        vb = self.vmax*(t[0,1:]-t[0,:-1])
        for i in range(self.d):
            diff = x[i,1:]-x[i,:-1]
            constraints.append(np.sqrt(2) * diff <= vb)
            constraints.append(- vb <= np.sqrt(2) * diff)
        
        return t,x, constraints



class Path_Tracking(Tube_Planning):
    def __init__(self, env, start, goal, milestones=[],max_dev = 1.0, vmax=1, bloating_r=0.5, d=2, t0=0, K_max=10, T_end_constraints=None, ignore_finished_agents=False, goal_reach_eps=None) -> None:
        super().__init__(env, start, goal, vmax, bloating_r, d, t0, K_max, T_end_constraints, ignore_finished_agents, goal_reach_eps)
        '''
            milestones: a list of locations between start and goal (excluded) to be visited.
            max_dev: maximal deviation from the milestones when visiting.
        '''
        self.milestones = milestones
        self.max_dev = max_dev
    

    def plan_core(self, active_obstacles=[], obstacle_trajectories=[], solve_inplace=True):
        '''
            Planning to track the milestones while observing other tube planning constraints.

            For now, we don't require the milestones be followed in specific order. As long as they are all visited, we deem it okay.
        '''
        # Get the usual constraints from simple tube planning
        t,x,constraints,_ = super().plan_core(active_obstacles, obstacle_trajectories, solve_inplace=False)
        
        # Add the path tracking loss
        if len(self.milestones)>0:
            n_milestones = self.milestones.shape[1]
            alpha = cp.Variable((self.K,n_milestones),boolean =True)
            M = 5 * np.max(np.abs(self.env.limits))

            for m in range(n_milestones):
                for i in range(self.K):
                    constraints.append(cp.norm(x[:,i]-self.milestones[:,m],1)<=self.max_dev+(1-alpha[i,m])*M)

            for m in range(n_milestones):
                constraints.append(cp.sum(alpha[:,m])>=1) # This ensures some x[:,i] is within max_dev of milestones[:,m]

        prob = cp.Problem(cp.Minimize(t[0,-1]),constraints)
      

        if not solve_inplace:
            return t,x,constraints,prob
        else:
            # print('number of integer constraints:',count_interger_var(prob))
            prob.solve(solver='GUROBI',TimeLimit = 100) # The Gurobi solver proves to be more accurate and also faster.
            if prob.status == 'optimal':
                out = unique_tx(t.value[0,:],x.value)
                del prob
                return out
            else:
                return None