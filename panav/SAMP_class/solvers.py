import cvxpy as cp
import numpy as np
from panav.env import trajectory_to_tube_obstacles, box_2d_center, trajectory_to_temp_obstacles,wp_to_tube_obstacle
from panav.util import unique_tx
from panav.conflict import plan_obs_conflict
class SAMP_Base:
    '''
        The base class for Single Angle Motion Planning(SAMP) solvers.
    '''
    def __init__(self,env,start,goal,bloating_r = 0.5) -> None:
        self.env = env
        self.start = start
        self.goal = goal
        self.bloating_r = bloating_r
    
    
    def plan(self,active_obstacles=[],obstacles_trajectories=[]):
        '''
            active_obstacles: typically individual segments of moving obstacles.
            obstacle_trajectories: typically entire trajectories of moving obstacles.
        '''
        raise NotImplementedError
    
    
    def lazy_optim(self, obstacle_trajectories=[], return_all=True):
        '''
            Assuming the obstacle trajectories can be broken down into active segments.

            Lazily adding the active segments into the problem, resolve until the problem becomes feasible.

            In many cases, leads to faster solution due to less integer constraints.
        '''
        active_obs = []
        m = sum([len(o[0])-1 for o in obstacle_trajectories])

        i = 0
        while i<=m:
            # print("num obstacle trajectories:{}/{}".format(len(active),m))
            p = self.plan(active_obstacles=active_obs)
            if p is None:
                print('Problem becomes infeasible.')
                break

            p = unique_tx(*p)
            # conflicted_obs = plan_obs_conflict(p, obstacle_trajectories, bloating_r)
            conflicted_obs = plan_obs_conflict(p, obstacle_trajectories, self.bloating_r, 
                                               segments_only=True,return_all=return_all)
            if not conflicted_obs:
                return p
            active_obs+=conflicted_obs
        return None

class Tube_Planning(SAMP_Base):
    def __init__(self, env, start, goal, 
                 vmax = 1, bloating_r = 0.5,d = 2, t0 = 0,K_max = 10,
                 T_end_constraints = None,ignore_finished_agents = False,goal_reach_eps = None) -> None:
        
        super().__init__(env,start,goal)

         
        self.vmax = vmax
        self.bloating_r = bloating_r
        self.d = d
        self.t0 = t0

        self.K = K_max
        self.K_max = K_max

        self.T_end_constraints = T_end_constraints
        self.ignore_finished_agents = ignore_finished_agents
        self.goal_reach_eps = goal_reach_eps if goal_reach_eps else bloating_r


    def plan(self,active_obstacles=[],obstacle_trajectories = [],K = "auto"):
        if K == "auto":
            for K in range(1,self.K_max+1):
                self.K = K
                p = self.plan_core(active_obstacles,obstacle_trajectories)
                if p: return p
        else:
            assert(type(K) == int)
            self.K = K
            p = self.plan_core(active_obstacles,obstacle_trajectories)
            self.K = self.K_max
            return p
    
    def plan_core(self,active_obstacles=[],obs_trajectories=[],solve_inplace = True):
        '''
            solve_in_place: False -> Return the MILP problem, including the variables, constraints, and prob objects, without calling prob.solve().
                            True -> Return the result of prob.solve().
        '''
        tubes = []
        for ta,pa in active_obstacles:
            tubes.append(wp_to_tube_obstacle(ta[0],ta[1],
                                        pa[:,0],pa[:,1],self.bloating_r))
        for times,xs in obs_trajectories:
            tubes+=trajectory_to_tube_obstacles(times,xs,self.bloating_r)

        # Decision variables and standard constraints for the path planning problem.
        t, x, constraints = self.Standard_Tube_Var_Constraints(self.start,self.goal,tubes,self.K)

        # print(obs_trajectories)
        # Additional constraints
        M = 5 * np.max(np.abs(self.env.limits))
        if not self.ignore_finished_agents:
            for obs_times,obs_xs in obs_trajectories:
                tfin = obs_times[-1]
                xfin = obs_xs[:,-1]
                O = box_2d_center(xfin,2*self.bloating_r) # The agent that has reached its goal, a static obstacle.
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
            prob.solve(solver='GUROBI',reoptimize =True) # The Gurobi solver proves to be more accurate and also faster.
            if t.value is not None:
                return unique_tx(t.value[0,:],x.value)
            else:
                return None
        
    def Standard_Tube_Var_Constraints(self,start,goal,tube_obs,K):

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



