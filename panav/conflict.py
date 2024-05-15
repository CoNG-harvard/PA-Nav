import numpy as np

def MA_plan_conflict(plan,bloating_r,first = True):
    '''
        Input: the plan, a list of (times,locs) pairs, each represented the timed piecewise-linear agent trajectory.
        
        Output: 
        If there is conflict within the plan, output the  the first conflict.
        The conflict includes the involved agents and conflicted timed line segments.
                            output = ([agent1, t1, seg1],[agent2, t2, seg2])
        
        If there is no conflict, return None.
    '''
    for i in range(len(plan)):
        conflicted_obs_trajectory = plan_obs_conflict(plan[i], plan[i+1:], bloating_r,first)
        if first and conflicted_obs_trajectory:
            return conflicted_obs_trajectory
        
    return None

def plan_obs_conflict(plan, obstacle_trajectories,bloating_r, segments_only = False, return_all = False):
    '''
        segments_only: False -> Return the full trajectories that have some conflict with plan.
                       True -> Return only the line segments in these trajectories that conflict with plan.
    '''
    conflicts = []
    if plan:
        t,x = plan
        for j in range(len(obstacle_trajectories)):
            tp,xp = obstacle_trajectories[j]
            p_conflict = path_path_conflict(t,x,tp,xp,bloating_r,bloating_r,return_all)
            if p_conflict:
                # print('conflict', j)
                if return_all:
                    if segments_only:
                        conflicts += p_conflict
                    else:
                        conflicts.append(obstacle_trajectories[j])
                else: 
                    if segments_only:
                        return p_conflict
                    else:
                        return obstacle_trajectories[j]

    return conflicts

def path_path_conflict(t,x,tp,xp,r,rp,return_all=False):
    if return_all:
        conflicts = []
    for k in range(len(t)-1):
        ta = t[k:k+2]
        pa = x[:,k:k+2]
        for m in range(len(tp)-1):
            tb = tp[m:m+2]
            pb = xp[:,m:m+2]
            if seg_conflict(ta,pa,tb,pb,r,rp):
                # print('ta',ta,'pa',pa)
                # print('tb',tb,'pb',pb)
                if return_all:
                    conflicts.append((tb,pb))
                else: # Return the first
                    return (tb,pb)
    if return_all:
        return conflicts
    else:
        return None

def seg_conflict(ta,pa,tb,pb,ra,rb):
    ''' 
        Determine if two space-time segments have conflict.
        
        Inputs:
        
            ta: shape = (2,). The start and end times of segment A.
            pa: shape = (dim, 2). The start and end coordinates of segment A.

            tb, pb: start and end times/coordinates of segment B.

            tb.shape = ta.shape
            pb.shape = pb.shape

            ra,rb: the bloating radii.
        
        Output: boolean. 
        
            True if the segments A and B with bloating radii ra, rb overlaps.
    '''
    if ta[0]>tb[1] or ta[1]<tb[0]:
        return False # The time intervals of these two segments are disjoint.
    
    lb = max([ta[0],tb[0]])
    ub = min([ta[1],tb[1]])

    va = (pa[:,1]-pa[:,0])/(ta[1]-ta[0])
    vb = (pb[:,1]-pb[:,0])/(tb[1]-tb[0])

    v = va-vb
    u = pa[:,0]-pb[:,0] - (va*ta[0]-vb*tb[0])

    if v.dot(v)>0:
        t_star = -u.dot(v)/v.dot(v)
        if lb<=t_star<=ub:
            return np.linalg.norm(u + t_star*v)<= ra+rb  # The minimal distance is attained when t=t_star.
    
    return np.linalg.norm(u + lb * v)<= ra+rb \
        or np.linalg.norm(u + ub * v)<= ra+rb 
        # The minimal distance is attained when t=lb or ub.
        # This happens when v=0, or v!=0 and t_star is outside (lb,ub)