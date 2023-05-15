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

    if env.limits:
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
        x,y = draw_start(s,ax, label = 'Starts' if agent_ID==0 else '')
        ax.text(x,y,str(agent_ID),ha='center',va='center')
    
    for agent_ID,g in enumerate(env.goals):
        x,y = draw_goal(g,ax, label = 'Goals' if agent_ID == 0 else '')
        ax.text(x,y,str(agent_ID),ha='center',va='center')

    agents = range(len(paths))

    for a in agents:    
        ax.plot(paths[a][0,:],paths[a][1,:],alpha = 0.5)
    

    # Switch off boundaries
    # ax.axis(False)
    
    # Use a square aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # if env.starts or env.goals:
    #     ax.legend()

def draw_obstacle(o,ax):
    verts = o.vertices()
    poly = Polygon(verts)
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor='r', alpha=0.3)
    
def draw_start(o,ax,label = ''):
    verts = o.vertices()
    poly = Polygon(verts)
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor='blue', alpha=0.3,label = label)
    return np.mean(x[1:]),np.mean(y[1:])

def draw_goal(o,ax,label = ''):
    verts = o.vertices()
    poly = Polygon(verts)
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor='g', alpha=0.3,label = label)
    return np.mean(x[1:]),np.mean(y[1:])



from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
def animation(env,paths,bloating_r,dt,fig=None,ax=None,agent_discs = None):
    '''
        Animate multi-agent trajectories in the given env.
        
        paths: paths[i](shape = (dim,n_steps)) is the path for agent i.
        
        bloat_r: the bloating radius of all agents.
        
        dt: time interval between two consecutive frames, measured in seconds.
    '''
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()

    

    agents = range(len(paths))

    # for a in agents:
    #     ax.plot(paths[a][:,0],paths[a][:,1],alpha = 0.5)

    if agent_discs is None:
        agent_discs = []
        for a in agents:
            disc = Circle(paths[a][0,:],bloating_r)
            agent_discs.append(disc)

    for disc in agent_discs:
        ax.add_artist(disc)
        
    draw_env(env,paths, ax)

    def init_func():
        return agent_discs

    def animate(t):
        for a,disc in zip(agents,agent_discs):
            if t<paths[a].shape[-1]:
                disc.center = paths[a][0,t],paths[a][1,t]
        return agent_discs

    handles, labels = ax.get_legend_handles_labels()
    if len(labels)>0:
        ax.legend()

    anim = FuncAnimation(fig,animate,frames = max([p.shape[-1] for p in paths]),blit=True,interval = dt*1000)
    return anim



import networkx as nx
from panav.util import interpolate_positions
from panav.env import NavigationEnv
def animate_MAPF_R(G,node_locs,obs_paths,agent_paths,dt,bloating_r):
    
    def path_to_traj(G_plan):
        x = np.vstack([node_locs[s] for s,t in G_plan]).T
        t = np.array([t for s,t in G_plan])
        t,x = interpolate_positions(t,x,dt)
        return x
    
    obs_trajs,agent_trajs = [[path_to_traj(p) for p in paths] for paths in (obs_paths,agent_paths)]
        
    fig = plt.figure()
    ax = plt.gca()
    
    obs_discs,agent_discs = [],[]
    
    for i,traj in enumerate(obs_trajs):
        obs_discs.append(Circle(traj[0],bloating_r,fc='red',ec = 'red'
                                ,label='Obstacle' if i==0 else None))

    for i,traj in enumerate(agent_trajs):
        agent_discs.append(Circle(traj[0],bloating_r,fc='green',ec = 'green'
                                ,label='Agent' if i==0 else None))
    
        
    ax.set_aspect('equal')

    nx.draw_networkx(G, {n:node_locs[n] for n in G},ax,node_size=150)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    
    env = NavigationEnv()
    anim  = animation(env,obs_trajs + agent_trajs,bloating_r,dt,
                      fig,ax,obs_discs + agent_discs)
    return anim