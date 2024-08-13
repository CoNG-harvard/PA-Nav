import numpy as np
import numpy.linalg as la

from panav.TrafficAwarePlanning import traffic_aware_HG_plan
from panav.ORCA import Ordered_Agent

from time import time

def PIBT_plan(HG,vmax,bloating_r,TIMEOUT,debug=False,simple_plan=True,
              max_iter = 500,
              tau = 1.0, # The safe time interval. Can be generously long.
              exec_tau = 0.4,    # The execution time of ORCA velocity.
                                # Should be much shorter than the safe interval tau.
                                # Leaving a slight horizon margin helps avoid numerical inaccuracy in CVXPY optimization results.

            exhaustive_search = True,
            ignore_inactive_agents =  True
              ):
    start_T = time()

    paths = traffic_aware_HG_plan(HG)
    # print(paths)
    ref_plan = [np.array([HG.node_loc(u) for u in path]).T for path in paths]
    plans = ref_plan

  
    # tau = 1.0 
    # exec_tau = 0.4 * tau 

    start_locs = HG.env.starts
    goal_locs = HG.env.goals
    N = len(start_locs)

    agents = np.arange(N)

    active_agents = set(agents)


    v_prefs = [np.zeros(start_locs[0].shape) for a in agents]

    protocol = 0

    orcas = [Ordered_Agent(protocol,tau,bloating_r,vmax,p,init_v = None,id = id) 
            for id,p in enumerate(start_locs)]

    # We will assume agent i is ranked i among the agents when dealing with conflicts.
    curr_wp_index = [1 for a in agents]

    entry_r = 6 * bloating_r


    def calc_pref(agent):
        
        target_wp = plans[agent][:,curr_wp_index[agent]]
        agent_loc = orcas[agent].p

        if orcas[agent].state == 'tunnel_waiting':
                curIdx = curr_wp_index[a]
                w = paths[a][curIdx]
                if curIdx == len(paths[a])-1:
                    print('AGENT',agent)
                v = paths[a][curIdx+1]
                
                pw,pv = HG.node_loc(w),HG.node_loc(v)
                
                
                theta = np.pi/4
                
                wait_loc = pw + 0.8 * entry_r * np.array([[np.cos(theta),-np.sin(theta)],
                                                          [np.sin(theta),np.cos(theta)]]) @ (pw-pv)/la.norm(pw-pv+1e-5)

                # wait_loc = agent_loc + (np.random.rand(2)-0.5)*0# Temporary solution: prefer to stay at the current location when waiting.
                v_prefs[agent] = towards(agent_loc,wait_loc,tau,vmax,HG.env,bloating_r,simple_plan)
                if debug:
                    print('agent', agent,'tunnel waiting v_pref',v_prefs[agent])
        else:
            v_prefs[agent] = towards(agent_loc,target_wp,tau,vmax,HG.env,bloating_r,simple_plan) # If not waiting, then head towards the target waypoint.            

    def PIBT(a):
        if time()-start_T>TIMEOUT:
            print('PIBT TIMEOUT')
            return False


        def find_C_P():
            P = []
            C = []
            AG = active_agents if ignore_inactive_agents else agents
            for nb in AG: # We cannot ignore retired agents but should still treat them as obstacles.
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

            orcas[a].update_v(v_pref,HG.env.obstacles,[orcas[b] for b in P]) 
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
                        print(f'Children {c} invalid')
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

        # Check for waypoint reaching and tunnel occupancy
        for a in active_agents:
            if debug:
                print('agent',a,'state',orcas[a].state)
            curIdx = curr_wp_index[a]
            agent_loc = orcas[a].p
            target_wp = plans[a][:,curIdx]

            w = paths[a][curIdx]
            if curIdx < plans[a].shape[1]-1: 
                v = paths[a][curIdx+1]
            else: 
                v = None

            if v is None and orcas[a].state == 'tunnel_waiting':
                return [(np.array(ts),np.array(xs).T) for ts,xs in zip(times,pos)]

            if v is not None and\
                HG.edges[w,v]['type']=='hard' and\
                la.norm(agent_loc-target_wp)<=entry_r and\
                orcas[a].state in ['tunnel_waiting','free']:
                            
                if len(HG.edges[v,w]['occupants'])==0 and HG.nodes[w]['occupant'] == None: 
                        HG.edges[w,v]['occupants'].add(a)
                        HG.nodes[w]['occupant'] = a
                        orcas[a].state = 'tunnel_entry'
                        if debug:
                            print('agent',a,'entering tunnel',w,v)
                else:
                    if debug:
                        print('agent',a,'waiting at tunnel',w,v)
                    orcas[a].state = 'tunnel_waiting'
            
            if debug:
                print('agent',a,'waypoint index',curr_wp_index[a])
            if la.norm(agent_loc-target_wp)<= bloating_r:  
                if curIdx == plans[a].shape[1]-1:
                    if debug:
                        print('target_wp',target_wp,'goal_locs[a]',goal_locs[a],'agent_loc',agent_loc)
                    assert(la.norm(target_wp-goal_locs[a])<bloating_r)
                    assert(la.norm(plans[a][:,-1]-goal_locs[a])<bloating_r)
                    assert(la.norm(agent_loc-goal_locs[a])<bloating_r)
                    
                    orcas[a].state = 'goal'
                else:
                    match orcas[a].state:
                        case 'tunnel_entry':

                            orcas[a].state = 'in_tunnel'
                            orcas[a].cur_edge = (w,v)
                            HG.nodes[w]['occupant'] = None # Release the entry node
                            # HG.edges[w,v]['occupants'].add(a) # This line is actually redundant
                        case 'in_tunnel':
                            orcas[a].state = 'free'
                            e = orcas[a].cur_edge
                            if debug:
                                print('agent',a,'leaving tunnel',e)
                            HG.edges[e]['occupants'].remove(a)
                            orcas[a].cur_edge = None

                    curr_wp_index[a] += 1 
        
        
        for a in active_agents:
            # Compute the preferred velocity.
            calc_pref(a)         

            prev_wp = curr_wp_index[a]-1
            cur_wp = curr_wp_index[a]
            u,v = paths[a][prev_wp:cur_wp+1]
            cur_edge_type = HG.edges[u,v]['type']
            if cur_edge_type == 'soft':
                orcas[a].vmax = 1.0 * vmax # On open space, move slower to avoid congestion at tunnel endpoints
            else:
                orcas[a].vmax = vmax # In tunnel, move quickly 
        
        
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
                # print('agent',a,'v',orcas[a].v)
                orcas[a].p += orcas[a].v*exec_tau
                all_reached = False
                # Reset all agent's v to be None
                orcas[a].v = None
            else:
                print('agent',a,'reached')
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