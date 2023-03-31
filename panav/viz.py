from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry.polygon import Polygon

def draw_env(env,paths=[],ax = None):
    '''
        env: the path planning environment.

        paths: a list of agents' paths. 
               
               paths[i](shape = (dim,n_steps)) is the path for agent i. 

        ax: the axis to plot on.
    '''
    if ax is None:
        ax = plt.gca()
    
    # Plot boundaries
    xlim = env.limits[0]
    ylim = env.limits[1]
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for y in ylim:
        ax.axhline(y,0,1)
    for x in xlim:
        ax.axvline(x,0,1)
        
    # Plot the obstacles
    for o in env.obstacles:
        draw_obstacle(o,ax)
   
    # Plot start and goal zones.
    for agent_ID,s in enumerate(env.starts):
        x,y = draw_start(s,ax)
        ax.text(x,y,str(agent_ID),ha='center',va='center')
    
    for agent_ID,g in enumerate(env.goals):
        x,y = draw_goal(g,ax)
        ax.text(x,y,str(agent_ID),ha='center',va='center')

    agents = range(len(paths))

    for a in agents:    
        ax.plot(paths[a][0,:],paths[a][1,:],alpha = 0.5)
    

    # Switch off boundaries
    # ax.axis(False)
    
    # Use a square aspect ratio
    ax.set_aspect('equal', adjustable='box')

def draw_obstacle(o,ax):
    verts = o.vertices()
    poly = Polygon(verts)
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor='r', alpha=0.3)
    
def draw_start(o,ax):
    verts = o.vertices()
    poly = Polygon(verts)
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor='blue', alpha=0.3)
    return np.mean(x[1:]),np.mean(y[1:])

def draw_goal(o,ax):
    verts = o.vertices()
    poly = Polygon(verts)
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor='g', alpha=0.3)
    return np.mean(x[1:]),np.mean(y[1:])



from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
def animation(env,paths,bloating_r,dt):
    '''
        Animate multi-agent trajectories in the given env.
        
        paths: paths[i](shape = (dim,n_steps)) is the path for agent i.
        
        bloat_r: the bloating radius of all agents.
        
        dt: time interval between two consecutive frames, measured in seconds.
    '''
    fig = plt.figure()
    ax = plt.gca()
    draw_env(env,paths, ax)

    agents = range(len(paths))

    # for a in agents:
    #     ax.plot(paths[a][:,0],paths[a][:,1],alpha = 0.5)

    agent_discs = []
    for a in agents:
        disc = Circle(paths[a][0,:],bloating_r)
        ax.add_artist(disc)
        agent_discs.append(disc)

    def init_func():
        return agent_discs

    def animate(t):
        for a,disc in zip(agents,agent_discs):
            if t<paths[a].shape[-1]:
                disc.center = paths[a][0,t],paths[a][1,t]
        return agent_discs

    anim = FuncAnimation(fig,animate,frames = max([p.shape[-1] for p in paths]),blit=True,interval = dt*1000)
    return anim