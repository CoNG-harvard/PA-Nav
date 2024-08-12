import numpy as np
import numpy.linalg as la

from panav.TrafficAwarePlanning import traffic_aware_HG_plan
from panav.ORCA import Ordered_Agent

from time import time

def PIBT_plan(env,ref_plan,vmax,bloating_r,TIMEOUT,debug=False,simple_plan=True,
              max_iter = 500,
              tau = 1.0, # The safe time interval. Can be generously long.
              
              
              exec_tau = 0.4,    # The execution time of ORCA velocity.
                                # Should be much shorter than the safe interval tau.
                                # Leaving a slight horizon margin helps avoid numerical inaccuracy in CVXPY optimization results.

            exhaustive_search = True
              ):
    start_T = time()

    plans = ref_plan

    # tau = 1.0 
    # exec_tau = 0.4 * tau 

    start_locs = env.starts
    goal_locs = env.goals
    N = len(start_locs)

    agents = np.arange(N)

    active_agents = set(agents)


    v_prefs = [np.zeros(start_locs[0].shape) for a in agents]

    protocol = 0

    orcas = [Ordered_Agent(protocol,tau,bloating_r,vmax,p,init_v = None,id = id) 
            for id,p in enumerate(start_locs)]

    # We will assume agent i is ranked i among the agents when dealing with conflicts.
    curr_wp_index = [1 for a in agents]


    def calc_pref(agent):
        
        target_wp = plans[agent][:,curr_wp_index[agent]]
        agent_loc = orcas[agent].p

        v_prefs[agent] = towards(agent_loc,target_wp,tau,vmax,env,bloating_r,simple_plan) # If not waiting, then head towards the target waypoint.            

    def PIBT(a):
        if time()-start_T>TIMEOUT:
            print('PIBT TIMEOUT')
            return False


        def find_C_P():
            P = []
            C = []
            for nb in agents: # We cannot ignore retired agents but should still treat them as obstacles.
                # Find all agents in the that could collide with a in the next tau seconds.
                dist = la.norm(orcas[nb].p-orcas[a].p)
                if nb!=a and dist<orcas[nb].bloating_r+orcas[a].bloating_r\
                                                        + 2 * vmax * tau:
                    if dist<orcas[nb].bloating_r+orcas[a].bloating_r:
                        print("Soft Collision. Agents ",a,nb,"Dist",np.linalg.norm(orcas[nb].p-orcas[a].p))
                    if orcas[nb].v is None:
                        C.append(nb)
                    else:
                        P.append(nb)
            return C,P
            
        candidate_v_pref = [] 


        for theta in np.pi * np.linspace(0,2,4)[:-1]:
                v_p = v_prefs[a]
                if la.norm(v_prefs[a])<1e-8:
                    v_p = np.array([1,0]) * vmax
                # Rotate v_pref clockwise by theta.
                v_right = np.array([[np.cos(-theta),-np.sin(-theta)],
                                    [np.sin(-theta),np.cos(-theta)]]).dot(v_p)
                candidate_v_pref.append(v_right)

        candidate_v_pref.append(np.array([0.0,0.0])) # Always have zero velocity as a candidate
        
        for v_pref in candidate_v_pref:
            if debug:
                print('agent',a,'v_pref',v_pref)

            C,P = find_C_P()

            orcas[a].update_v(v_pref,env.obstacles,[orcas[b] for b in P]) 
            if orcas[a].v is None:
                if not exhaustive_search:
                    orcas[a].v = orcas[a].v_opt = np.array([0,0]) # Assign invalidity with zero velocity to avoid exponential computational cost
                return False
            
            children_valid = True
            for c in C:
                if orcas[c].v is None:
                    if debug:
                        print('Agent',a,' calling PIBT for agent', c)
                    children_valid = PIBT(c)
                    if not children_valid:
                        print(r'Children {c} invalid')
                        break
            
            if children_valid:
                return True

        if exhaustive_search:
            orcas[a].v = None
        else:
            orcas[a].v = orcas[a].v_opt = np.array([0.0,0.0]) # Assign invalidity with zero velocity to avoid exponential computational cost
        return False
    
    
    pos = [[] for _ in range(N)]
    times = [[] for _ in range(N)]

    curr_t = 0

    for _ in range(max_iter):
        # if debug:
        if True:
            print("################# Time step {} ################".format(_))
            print("Remaining agents",len(active_agents))
        for a in agents:
            pos[a].append(np.array(orcas[a].p))
            times[a].append(curr_t)

        # Compute the preferred velocity.
        for a in active_agents:
            

            calc_pref(a)         
        
        for a in active_agents:
            if orcas[a].v is None:
                valid = PIBT(a)
                if not valid:
                    print("PIBT failed")
                    return [(np.array(ts),np.array(xs).T) for ts,xs in zip(times,pos)]
        # We assume agent a is ranked a among the agents when dealing with conflicts.

        # Execute the safe velocity.
        all_reached = True
        
        retired_agents = []
        for a in active_agents:
            dist2goal = np.linalg.norm(orcas[a].p - goal_locs[a])
    
            if dist2goal>=0.5*bloating_r:
                # Check if the current waypoint should be updated
                curIdx = curr_wp_index[a]
                target_wp = plans[a][:,curIdx]
                if np.linalg.norm(orcas[a].p-target_wp)<0.5 * bloating_r:
                    curr_wp_index[a] += 1 

                # print('agent',a,'v',orcas[a].v)
                orcas[a].p += orcas[a].v*exec_tau
                all_reached = False
                # Reset all agent's v to be None
                orcas[a].v = None

            else:
                orcas[a].goal_reached = True
                orcas[a].v = orcas[a].v_opt = np.array([0,0])
                retired_agents.append(a) 

        # The active_agents' size cannot change in the loop above, so we have to do the removal outside.
        for ra in retired_agents:
            active_agents.remove(ra)
        
        curr_t += exec_tau

        if all_reached:
            print('PIBT success')
            break
    
    return [(np.array(ts),np.array(xs).T) for ts,xs in zip(times,pos)]

from shapely import LineString
from panav.multi_path import shortest_path
def towards(cur_loc, wp, tau, vmax, env, bloating_r,simple_plan = True):

    obs_in_conflict = []
    if not simple_plan:
        no_conflict = True
        for obs in env.obstacles:
            if la.norm(obs.project(cur_loc)-cur_loc) < tau * vmax:
                if obs.verts.distance(LineString([cur_loc,wp]))<bloating_r:
                    no_conflict = False
                    obs_in_conflict.append(obs)

        
        if no_conflict:
            wp = wp
        else:
            # print('Planning using MILP')
            result = local_MILP_plan(env,obs_in_conflict,cur_loc,wp,vmax,bloating_r)
            if result is not None:
                wp = result
  
    to_wp = wp-cur_loc
    return to_wp/tau if tau * vmax > np.linalg.norm(to_wp) else vmax *  to_wp/(np.linalg.norm(to_wp)+1e-5)


def local_MILP_plan(env,obs_in_conflict,cur_loc,wp,vmax,bloating_r):
    # print(cur_loc,wp)
    r = bloating_r
    while True:
        x,_ = shortest_path(env,obs_in_conflict,cur_loc,wp,K=1,d=2,
                    existing_paths = [],
                    bloating_r = r,local_plan_radius=1.5*bloating_r)
        if x is not None:
            break
        r *= 0.9
    return x[:,1]