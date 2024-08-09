
import sys
from os.path import abspath, dirname
sys.path.append(dirname(abspath(__file__)))

from time import time
import numpy as np

from ompl import base as ob
from ompl import geometric as og
from RRT.utils import has_line_collision

sys.path.append(dirname(abspath(__file__))+'../')
from panav.conflict import plan_obs_conflict

def ST_RRTStar_wrapper(env, start, goal, obstacle_trajectories, vmax, bloating_r, MIN_RUN_TIME = 0.1):

    space_dim = len(start)
    bounds = ob.RealVectorBounds(space_dim)

    limits = env.limits

    for axis in range(space_dim):
        bounds.setLow(axis,min(limits[axis]))
        bounds.setHigh(axis,max(limits[axis]))
        
    space = ob.RealVectorStateSpace(space_dim)
    space.setBounds(bounds)
    spaceTime = ob.SpaceTimeStateSpace(space,vmax)
    si = ob.SpaceInformation(spaceTime)

    # Set the object used to check which states in the space are valid
    validityChecker = Validator(si,env,bloating_r,obstacle_trajectories)
    si.setMotionValidator(validityChecker)

    si.setup()
    
    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    start = py_to_state(start,spaceTime)
    goal = py_to_state(goal,spaceTime)

    # Set the start and goal states
    pdef.setStartAndGoalStates(start(), goal())

    # Define the mechanism that allows us intercept the event when ST-RRT* first find a feasible solution
    feasibleSolFound = {'result':False}
    def intermediateSol(_,spath,best_cost): # We know the signature of the callback is like this because we looked into the C++ source code of RRT*
        feasibleSolFound['result'] = True
        return spath, best_cost.value()
    pdef.setIntermediateSolutionCallback(ob.ReportIntermediateSolutionFn(intermediateSol))

    # Initialize the core planner from OMPL
    optimizingPlanner = og.STRRTstar(si)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    t0 = time()

    term_condition = lambda : feasibleSolFound['result'] and time()-t0 > MIN_RUN_TIME

    term_condition = ob.PlannerTerminationCondition(ob.PlannerTerminationConditionFn(term_condition))
    solved = optimizingPlanner.solve(term_condition)

    path = pdef.getSolutionPath().getStates()

    ts = []
    xs = []
    for p in path:
        x = p[0]
        xs.append([x[i] for i in range(space_dim)])
        ts.append(spaceTime.getStateTime(p))
    ts = np.array(ts)
    xs = np.array(xs).T

    return ts,xs

def sequential_ST_RRTStar(env,vmax,bloating_r,return_cumu_times=False):
    t0 = time()

    continuous_plans = []
    cumu_times = []
    

    for i in range(len(env.starts)):
        print('Planning for agent {}/{}'.format(i,len(env.starts)))
        start = env.starts[i,:]
        goal = env.goals[i,:]
        
        p = ST_RRTStar_wrapper(env,start,goal,continuous_plans,vmax,bloating_r)

        if p is None:
            return None

        continuous_plans.append(p)
        
    return continuous_plans


def py_to_state(s,space):
    space_dim = len(s)
    x = ob.CompoundState(space)
    for i in range(space_dim):
        x[i] = s[i]
    return x

def state_to_py(s,space_dim = 2):
    return np.array([s[i] for i in range(space_dim)])

def path_to_py(path,space_dim = 2):
    # n_states = path.getStateCount()
    out = [state_to_py(v,space_dim) for v in path]
    return np.array(out).T

class Validator(ob.MotionValidator):
    def __init__(self,si,env,bloating_r,obstacle_trajectories=[]):
        super().__init__(si)
        self.env = env
        self.bloating_r = bloating_r
        self.obstacle_trajectories = obstacle_trajectories

        self.ST = si.getStateSpace()
        self.vmax = self.ST.getVMax()

    def getTime(self,s):
        return self.ST.getStateTime(s)
    
    def checkMotion(self,s1,s2):
        t1 = self.getTime(s1)
        t2 = self.getTime(s2)

        x1 = state_to_py(s1[0]) # s[0] is the space component, s[1] is the time component
        x2 = state_to_py(s2[0])

        leap = np.linalg.norm(x1-x2)
        
        if t1 == t2:
            return leap  == 0
        
        local_plan = (np.array([t1,t2]), np.vstack([x1,x2]).T)
        
        return all([t2>=t1,
                    not has_line_collision(self.env,x1,x2,self.bloating_r),
                    not plan_obs_conflict(local_plan,self.obstacle_trajectories,self.bloating_r),
                    leap / abs(t2-t1) <= self.vmax
                    ])

