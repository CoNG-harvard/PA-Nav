import numpy as np
def unique_tx(t,x):
    times,xs = np.array(t),np.array(x)

    unique_index = []
    for i in range(len(times)-1):
        if times[i]<times[i+1]:
            unique_index.append(i)
            
    unique_index.append(i+1)
    times = times[unique_index]
    xs = xs[:,unique_index]
    return times, xs

def interpolate_positions(t,x,dt):

    pos = []
    times = []

    for i in range(len(t)-1):
        n = int((t[i+1]-t[i])/dt)
        pos.append(np.linspace(x[:,i],x[:,i+1],n))
        times.append(np.linspace(t[i],t[i+1],n))
        
    return  np.hstack(times),np.vstack(pos)