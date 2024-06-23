import numpy as np
from matplotlib.patches import Circle
from numpy import linalg as la
import cvxpy as cp

class ORCA_Agent:
    def __init__(self,protocol,tau,
                bloating_r,vmax,
                 init_p,init_v=None,vmin = None):
        
        valid = {0, 1, 2}
        if protocol not in valid:
            raise ValueError("results: protocol must be one of %r." % valid)
       
        self.protocol = protocol
        
        self.p = np.array(init_p)

        if init_v is None:
            init_v = np.zeros(self.p.shape)
            
        self.v = init_v

        self.tau = tau
        self.bloating_r = bloating_r
        
        self.v_opt = np.zeros(self.v.shape)
        self.vmax = vmax

        if vmin is None:
            self.vmin = 0.01 * vmax

        self.resolving_deadlock = False
    
    def update_v_opt(self,v_pref):
        self.v_opt = self.calc_v_opt(v_pref)
        
    def update_v(self,v_pref,obstacles,neigbor_agents,right_hand_rule=True):
        self.v = self.safe_v(v_pref,obstacles,neigbor_agents,
                                right_hand_rule=right_hand_rule)
    
    def calc_v_opt(self,v_pref):
         # Determine v_opt
        if self.protocol == 0:
            v_opt = np.zeros(v_pref.shape)
        elif self.protocol == 1:
            v_opt = v_pref
        elif self.protocol == 2:
            v_opt = self.v
        return v_opt
    
    def neighbor_constraints(self,neigbor_agents):
        '''
            Return the set of u and n vectors representing the ORCA half planes

            right_hand_rule: deadlock avoidance mechanism.
                             If true, perturb the normal vector to the right hand side by a little.
        '''
        us,ns = [],[]
        for b in neigbor_agents: 
            if la.norm(self.p-b.p)<=(b.vmax+self.vmax)*self.tau+self.bloating_r+b.bloating_r: # Only consider those agents that are close enough to the ego agent.
                vo = VO(self.p,b.p,
                        self.bloating_r,b.bloating_r,self.tau)
                
                v_rel = self.v_opt-b.v_opt
                
                zone_code = vo.zones(v_rel)
                u = vo.u(v_rel)

                if zone_code == 2:
                    n = -u/np.linalg.norm(u)
                    # n = np.zeros(u.shape)
                    # continue # The agent is not in conflict with neighboring agent b.
                else:
                    n = u/np.linalg.norm(u)

                us.append(u)
                ns.append(n)
        return us, ns

    def safe_v(self,v_pref,obstacles,neigbor_agents,right_hand_rule = True):
        
        # Constraints induced by other agents.
        us,ns = self.neighbor_constraints(neigbor_agents)
        # Constraints induced by static obstacles
        obstacle_d = [] 
        for O in obstacles: 
            to_obs = O.project(self.p) - self.p

            if la.norm(to_obs)<=self.vmax*self.tau+1.5*self.bloating_r: # Only consider those static obstacles that are close enough to the ego agent.
                obstacle_d.append(to_obs)

        v = v_pref 


        if np.all([(v-(self.v_opt+u/2)).dot(n) >= 0 for u,n in zip(us,ns)])\
            and np.all([d.dot(v*self.tau)/np.linalg.norm(d) <= (np.linalg.norm(d)-self.bloating_r)
                      for d in obstacle_d]): # v_pref is already a feasible velocity.
            return v
        

        ######## Safe velocity calculation using Gurobi
        v = cp.Variable(v_pref.shape) 
        
        constraints = [(v-(self.v_opt+u/2)) @ n >= 0 for u,n in zip(us,ns)]

        constraints += [d/np.linalg.norm(d) @ (v*self.tau) <= (np.linalg.norm(d)-self.bloating_r)
                      for d in obstacle_d]

        # Maximum speed constraint
        constraints.append(cp.norm(v)<= self.vmax)

        prob = cp.Problem(cp.Minimize(cp.norm(v-v_pref)),constraints)
        
        prob.solve()
        v_out = v.value
        ########

        ############ Safe velocity calculation using grid sampling
        # def grid_solve_v(v_pref,n_grid=10):
        #     candx,candy = np.meshgrid(np.linspace(-self.vmax,self.vmax,n_grid),
        #                         np.linspace(-self.vmax,self.vmax,n_grid))

        #     cand =[]
        #     candx = candx.flatten()
        #     candy = candy.flatten()
        #     for vx,vy in zip(candx,candy):
        #         v = np.array([vx,vy])
        #         constraints = [(v-(self.v_opt+u/2)) @ n >= 0 for u,n in zip(us,ns)]

        #         constraints+=[d/np.linalg.norm(d) @ (v*self.tau) <= (np.linalg.norm(d)-self.bloating_r)
        #                     for d in obstacle_d]
                
        #         constraints.append(la.norm(v)<= self.vmax)
        #         if np.all(constraints):
        #             cand.append(v)
            
        #     return cand[np.argmin([la.norm(v-v_pref) for v in cand])]
        # v_out = grid_solve_v(v_pref)
        #############

        if v_out is None: # The case where the problem is infeasible.
            # print('infeasible') 
            v_out = np.zeros(v_pref.shape) # Temporary solution. To be extended next.
        elif np.linalg.norm(v_out)<=self.vmin and right_hand_rule:
            # print('Potential deadlock')
            # Potential deadlock, engage the right-hand rule
            for theta in np.pi * np.array([1/2,1]):
                # Rotate v_pref clockwise by theta.
                v_right = np.array([[np.cos(-theta),-np.sin(-theta)],
                                    [np.sin(-theta),np.cos(-theta)]]).dot(v_pref)

                prob = cp.Problem(cp.Minimize(cp.norm(v-v_right)),constraints)
                prob.solve()
                v_out = v.value

                # v_out = grid_solve_v(v_right)
                # if np.linalg.norm(v.value)>self.vmin:
                if la.norm(v_out)>self.vmin:
                    # v_out = v.value 
                    break

        return v_out

class Ordered_Agent(ORCA_Agent):
    '''
        The Ordered_Agent overrides the safe_v() method, to avoid higher_ranked_agents, assuming they will not avoid the ego agent.

        Specifically, the hyperplane is not offset by u/2 as in ORCA_Agent, but by u instead.

        Also, in the ordered agent, we don't implement the right-hand-rule since there won't be the issue of deadlock.

        We also don't have the distinction between v and v_opt. v_opt will be v exactly.
    '''
    def __init__(self, protocol, tau, bloating_r, vmax, init_p, init_v=None, vmin=None):
        super().__init__(protocol, tau, bloating_r, vmax, init_p, init_v, vmin)

    
    def update_v(self,v_pref,obstacles,neigbor_agents,right_hand_rule=True):
        self.v_opt = v_pref
        self.v = self.safe_v(v_pref,obstacles,neigbor_agents,
                                right_hand_rule=right_hand_rule)
        self.v_opt = self.v
    
    def safe_v(self, v_pref, obstacles, higher_ranked_agents, right_hand_rule=True):
        us,ns = self.neighbor_constraints(higher_ranked_agents)
        # Constraints induced by static obstacles
        obstacle_d = [] 
        for O in obstacles: 
            to_obs = O.project(self.p) - self.p

            if la.norm(to_obs)<=self.vmax*self.tau+1.5*self.bloating_r: # Only consider those static obstacles that are close enough to the ego agent.
                obstacle_d.append(to_obs)

        v = v_pref 


        if np.all([(v-(self.v_opt+u)).dot(n) >= 0 for u,n in zip(us,ns)])\
            and np.all([d.dot(v*self.tau)/np.linalg.norm(d) <= (np.linalg.norm(d)-self.bloating_r)
                      for d in obstacle_d]): # v_pref is already a feasible velocity.
            return v
        

        ######## Safe velocity calculation using Gurobi
        v = cp.Variable(v_pref.shape) 
        
        constraints = [(v-(self.v_opt+u)) @ n >= 0 for u,n in zip(us,ns)]

        constraints += [d/np.linalg.norm(d) @ (v*self.tau) <= (np.linalg.norm(d)-self.bloating_r)
                      for d in obstacle_d]

        # Maximum speed constraint
        constraints.append(cp.norm(v)<= self.vmax)

        prob = cp.Problem(cp.Minimize(cp.norm(v-v_pref)),constraints)
        
        prob.solve()
        v_out = v.value
        ########

        if v_out is None: # The case where the problem is infeasible.
            print('infeasible') 
            v_out = None # Temporary solution. To be extended next.


        return v_out

class VO:
    '''
        The velocity obstacle of agent a induced by agent b.
        See notebooks/ORCA/Velocity Obstacle.ipynb for a visual documentation of VO.
        See notebooks/ORCA/Calculate u.ipynb for a visual documentation of the calculation of u vector. 
    '''
    def __init__(self,pa,pb,ra,rb,tau):
        '''
            pa: agent a's location.
            pb: the neighboring agent b's location.
            ra: agent a's radius.
            rb: the neighoring agent b's radius.
            tau: the safe time interval. 
                 Agent a is not suppose to crash into b for the future tau seconds. 
        '''
        self.center = (pb-pa)/tau
        self.r = (ra+rb)/tau
        self.center_theta = np.arctan2(self.center[1],self.center[0])
        
        if la.norm(self.center)>0 and self.r <= la.norm(self.center): # The two agents do not overlap
            self.phi = np.arcsin(self.r/la.norm(self.center))
        
    def visualize(self, ax):
        '''
            Visualize the velocity obstacle.
        '''
        xlim = np.array((-1,1)) * la.norm(self.center) *2
        ylim = np.array((-1,1)) * la.norm(self.center) *2
        nx = 200
        ny = 200
        xv,yv = np.meshgrid(np.linspace(*xlim,nx),np.linspace(*ylim,ny))
        test_pt = np.vstack([xv.flatten(),yv.flatten()]).T

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if self.r <= la.norm(self.center): 
            circ = Circle(self.center,self.r,fc = 'green',alpha = 0.2)

            ax.add_artist(circ)

            ax.axline([0,0], slope = np.tan(self.center_theta-self.phi),ls='dotted')
            ax.axline([0,0], slope = np.tan(self.center_theta+self.phi),ls='dotted')


            zone_code = self.zones(test_pt)

            mk = ['o','o','x']

            color = ['blue','orange','grey']

            labels = ['Zone 0','Zone 1','Not in VO']

            for z in [0,1,2]:

                ax.scatter(test_pt[zone_code == z,0],test_pt[zone_code == z,1]\
                           ,s=3,marker=mk[z],fc='none',ec=color[z],label = labels[z]\
                            )

        else:
            ax.scatter(0,0,alpha=0.0,label = 'Two agents overlap')

        ax.grid(True)
        ax.set_aspect('equal')


    def zones(self,test_pt):
        '''
            Classify the test_pt in terms of its relationship to the velocity obstacle.
            
            Output: zone code. 
                   -1: the bloating regions of the two agents overlap. VO is the entire space. 
                    0: test_pt is in VO and is closest to the front arc of VO.
                    1: test_pt in VO and is closest to one of the two sides of VO.
                    2: test_pt is not in VO.
        '''

        test_pt = np.array(test_pt)
        single_input = len(test_pt.shape) == 1
        if single_input:
            test_pt = test_pt[np.newaxis,:]

        zone_code = np.zeros(len(test_pt))
        
        if self.r > la.norm(self.center): 
            zone_code[:] = -1
        else:
            in_circ = la.norm(test_pt - self.center,axis = 1)<=self.r

            in_squeeze = test_pt.dot(self.center)\
                        >= np.cos(self.phi) * (la.norm(test_pt,axis = 1) * la.norm(self.center)) 

            far = np.logical_and(in_squeeze, 
                                 la.norm(test_pt,axis = 1)>= self.r/np.tan(self.phi))

            vo = np.logical_or(in_circ,far)

            in_front = (test_pt - self.center).dot(-self.center)\
                      >= np.cos(np.pi/2-self.phi) * (la.norm(test_pt - self.center,axis = 1) * la.norm(self.center))

            zone0 = np.logical_and(vo,
                                   np.logical_and(in_circ, in_front))

            zone1 = np.logical_and(vo,
                                   np.logical_not(zone0))

            zone2 = np.logical_not(vo)

            zone_code[zone0] = 0
            zone_code[zone1] = 1
            zone_code[zone2] = 2

        if single_input:
            zone_code = zone_code[0]

        return zone_code

    def u(self, test_pt):
        '''
            Compute the u vector of the test_pt, such that u is the projection of
            test_pt on the boundary of this VO.
        '''

        zone = self.zones(test_pt)

        if zone == -1:
            u = -self.center
        else:
            to_center = test_pt - self.center

            if la.norm(to_center)==0:
                u = - self.center/(la.norm(self.center)) * self.r
                print('Closely hit')
            else:
                if zone == 2: # Relative velocity not in VO.
                    # l = la.norm(self.center)*np.cos(self.phi)
                    # thetas = [self.center_theta + self.phi,
                    #           self.center_theta - self.phi]
                    # p1,p2 = [l * np.array([np.cos(theta),np.sin(theta)])
                    #          for theta in thetas]
                    
                    # proj1 = cp.Variable(p1.shape)
                    # constraints = [
                    #     (proj1 - p1) @ (self.center-p1)>=0,
                    #     (proj1 - p2) @ (self.center-p2)>=0,
                    #     (proj1 - (p1+p2)/2) @ (p1+p2)>=0
                    # ]
                    # prob = cp.Problem(cp.Minimize(cp.norm(proj1 - test_pt)),constraints)
                    # prob.solve()
                    # proj1 = proj1.value
                    
                    c = (test_pt-self.center).dot(-self.center)/(la.norm(test_pt-self.center)*la.norm(self.center))
                    if c>=np.cos(np.pi/2-self.phi):
                        proj = self.center+\
                            (test_pt-self.center)/la.norm(test_pt-self.center) * self.r
                    else:
                        side_thetas = [self.center_theta-self.phi, 
                                    self.center_theta+self.phi]
                        sides = np.vstack([np.cos(side_thetas),np.sin(side_thetas)])

                        l = sides.T.dot(test_pt)

                        proj =  sides[:,l>=0]* l[l>=0] # The projection must be on the same direction as the boundary ray


                        dist_2_proj = la.norm((proj.T - test_pt).T,axis = 0)

                        proj = proj[:,np.argmin(dist_2_proj)]
                    
                    # dists = [la.norm(test_pt-proj1),la.norm(test_pt-proj2)]
                    
                    # proj = [proj1,proj2][np.argmin(dists)]
                    u = proj - test_pt

                    # u = np.zeros(self.center.shape)
                elif zone == 1:
                    side_thetas = [self.center_theta-self.phi, 
                                   self.center_theta+self.phi]
                    sides = np.vstack([np.cos(side_thetas),np.sin(side_thetas)])

                    proj = sides.T.dot(test_pt) * sides # proj = [proj_1(a column vector), proj 2(a column vector)]

                    dist_2_proj = la.norm((proj.T - test_pt).T,axis = 0)

                    true_proj = proj[:,np.argmin(dist_2_proj)]

                    u = true_proj - test_pt
                elif zone == 0:
                    u = to_center/(la.norm(to_center))\
                        * (self.r - la.norm(to_center))
        return u
