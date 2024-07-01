import numpy as np
import numpy.linalg as la

from panav.TrafficAwarePlanning import traffic_aware_HG_plan
from panav.ORCA import Ordered_Agent

from time import time

def PIBT_plan(HG,vmax,bloating_r,TIMEOUT,consider_entry=False):
    start_T = time()

    paths = traffic_aware_HG_plan(HG)
    # print(paths)
    ref_plan = [np.array([HG.node_loc(u) for u in path]).T for path in paths]
    plans = ref_plan

    # The execution time of ORCA velocity.
    # Should be much shorter than the safe interval tau.
    tau = 1.0 # The safe time interval. Can be generously long.
    exec_tau = 0.5 * tau # Leaving a slight horizon margin helps avoid numerical inaccuracy in CVXPY optimization results.

    start_locs = HG.env.starts
    goal_locs = HG.env.goals
    N = len(start_locs)

    agents = np.arange(N)


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
                wait_loc = pw + 0.8 * entry_r * (pw-pv)/la.norm(pw-pv+1e-5)

                # wait_loc = agent_loc + (np.random.rand(2)-0.5)*0# Temporary solution: prefer to stay at the current location when waiting.
                v_prefs[agent] = towards(agent_loc,wait_loc,tau,vmax)
                # print('agent', agent,'tunnel waiting v_pref',v_prefs[agent])
        else:
            v_prefs[agent] = towards(agent_loc,target_wp,tau,vmax) # If not waiting, then head towards the target waypoint.            

    def PIBT(a):
        if time()-start_T>TIMEOUT:
            print('PIBT TIMEOUT')
            return False

        P = []
        C = []

        for nb in agents:
            # Find all agents in the that could collide with a in the next tau seconds.
            if nb!=a and np.linalg.norm(orcas[nb].p-orcas[a].p)<orcas[nb].bloating_r+orcas[a].bloating_r\
                                                    + 2 * vmax * tau:
                if np.linalg.norm(orcas[nb].p-orcas[a].p)<orcas[nb].bloating_r+orcas[a].bloating_r:
                    print("Soft Collision. Agents ",a,nb,"Dist",np.linalg.norm(orcas[nb].p-orcas[a].p))
                if orcas[nb].v is None:
                    C.append(nb)
                else:
                    P.append(nb)
            
        candidate_v_pref = [] 


        for theta in np.pi * np.linspace(0,2,4)[:-1]:
                # Rotate v_pref clockwise by theta.
                v_right = np.array([[np.cos(-theta),-np.sin(-theta)],
                                    [np.sin(-theta),np.cos(-theta)]]).dot(v_prefs[a])
                candidate_v_pref.append(v_right)

        candidate_v_pref.append(np.array([0,0])) # Always have zero velocity as a candidate
        
        for v_pref in candidate_v_pref:
            # print('agent',a,'v_pref',v_pref)
            orcas[a].update_v(v_pref,HG.env.obstacles,[orcas[b] for b in P]) 
            if orcas[a].v is None:
                return False
            
            children_valid = True
            for c in C:
                if orcas[c].v is None:
                    children_valid = PIBT(c)
                    if not children_valid:
                        break
            
            if children_valid:
                return True

        orcas[a].v = None
        return False
    
    
    pos = [[] for _ in range(N)]
    times = [[] for _ in range(N)]

    curr_t = 0

    for _ in range(400):
        print("################# Time step {} ################".format(_))
        for a in agents:
            pos[a].append(np.array(orcas[a].p))
            times[a].append(curr_t)

        # Check for waypoint reaching and tunnel occupancy
        for a in agents:
            # print('agent',a,'state',orcas[a].state)
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

                # if _ == 6 and a == 4:
                # print('Occupants of (0,1)',HG.edges[0,1]['occupants'], 'Occupants of (1,0)',HG.edges[1,0]['occupants'], 'Occupants for 0', HG.nodes[0]['occupant'],'Occupants for 1',HG.nodes[1]['occupant'])
                            
                if len(HG.edges[v,w]['occupants'])==0 and HG.nodes[w]['occupant'] == None: 
                        HG.edges[w,v]['occupants'].add(a)
                        HG.nodes[w]['occupant'] = a
                        orcas[a].state = 'tunnel_entry'

                        print('agent',a,'entering tunnel',w,v)
                else:

                    print('agent',a,'waiting at tunnel',w,v)
                    orcas[a].state = 'tunnel_waiting'
            

            if la.norm(agent_loc-target_wp)<= bloating_r:  
                if curIdx == plans[a].shape[1]-1:
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

                            print('agent',a,'leaving tunnel',e)
                            HG.edges[e]['occupants'].remove(a)
                            orcas[a].cur_edge = None
                    
                    curr_wp_index[a] += 1 
        
        
        for a in agents:
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
        
        
        for a in agents: 
            if orcas[a].v is None:
                valid = PIBT(a)
                if not valid:
                    print("PIBT failed")
                    return [(np.array(ts),np.array(xs).T) for ts,xs in zip(times,pos)]
        # We assume agent a is ranked a among the agents when dealing with conflicts.

        # Execute the safe velocity.
        all_reached = True
        
        for a in agents:
            dist2goal = np.linalg.norm(orcas[a].p - goal_locs[a])
    
            if dist2goal>=0.5*bloating_r:
                # print('agent',a,'v',orcas[a].v)
                orcas[a].p += orcas[a].v*exec_tau
                all_reached = False
                # Reset all agent's v to be None
                orcas[a].v = None
            else:
                orcas[a].goal_reached = True
                orcas[a].v = orcas[a].v_opt = np.array([0,0])
        
        curr_t += exec_tau

        if all_reached:
            break
    
    return [(np.array(ts),np.array(xs).T) for ts,xs in zip(times,pos)]

def towards(cur_loc, wp, tau, vmax):
    to_wp = wp-cur_loc
    return to_wp/tau if tau * vmax > np.linalg.norm(to_wp) else vmax *  to_wp/(np.linalg.norm(to_wp)+1e-5)
