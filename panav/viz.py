from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry.polygon import Polygon

def draw_env(env,ax = None):
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
    

    # Switch off boundaries
    ax.axis(False)
    
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