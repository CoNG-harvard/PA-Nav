import numpy as np

def MA_plan_conflict(times,plans):
    for i in range(len(plans)):
        t,x=times[i],plans[i]
        for j in range(i+1,len(plans)):
            tp,xp = times[j],plans[j]
            if path_conflict(t,x,tp,xp,bloating_r,bloating_r):
                return True
    return False

def path_conflict(t,x,tp,xp,r,rp):
    for k in range(len(t)-1):
        ta = t[k:k+2]
        pa = x[:,k:k+2]
        for m in range(len(tp)-1):
            tb = tp[m:m+2]
            pb = xp[:,m:m+2]
            if seg_conflict(ta,pa,tb,pb,r,rp):
                # print('ta',ta,'pa',pa)
                # print('tb',tb,'pb',pb)
                return True
    return False

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

    t_star = -u.dot(v)/v.dot(v)
    
    if lb<=t_star<=ub:
        return np.linalg.norm(u + t_star*v)<= ra+rb  # The minimal distance is attained when t=t_star.
    else:
        return np.linalg.norm(u + lb * v)<= ra+rb \
            or np.linalg.norm(u + ub * v)<= ra+rb 
            # The minimal distance is attained when t=lb or ub.