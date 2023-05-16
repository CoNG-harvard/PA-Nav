from panav.PBS.conflict import seg_conflict
import numpy as np
from panav.SIPP import plan_to_transitions
def MAPFR_pairwise_conflict(plan1,plan2,node_locs,bloating_r):
    '''
        plan1 and plan2: lists containing (u_i,t_i) waypoints.
        node_locs: a dict of {node:locs}
    '''
    for u,v, t1,t2 in plan_to_transitions(plan1):
        seg1 = ([t1,t2],np.vstack([node_locs[u],node_locs[v]]).T)
        for p,q,l1,l2 in plan_to_transitions(plan2):
            seg2 = ([l1,l2],np.vstack([node_locs[p],node_locs[q]]).T)
            if seg_conflict(*seg1,*seg2,bloating_r,bloating_r):
                return (u,v),(t1,t2),(p,q),(l1,l2)
            
    return None

def MAPFR_conflict(plans,node_locs,bloating_r):
    for i in range(len(plans)):
        planA = plans[i]
        for j in range(i+1,len(plans)):
            planB = plans[j]
            conflict = MAPFR_pairwise_conflict(planA,planB,node_locs,bloating_r)
            if conflict is not None:
                return {'agents':(i,j), 
                        'conflict':
                        [{'agent':i,'transition':conflict[0],'duration':conflict[1]},
                        {'agent':j,'transition':conflict[2],'duration':conflict[3]}]}
    return None