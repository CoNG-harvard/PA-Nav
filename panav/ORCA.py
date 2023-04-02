import numpy as np
from matplotlib.patches import Circle
from numpy import linalg as la


class VO:
    '''
        The velocity obstacle of agent a induced by agent b.
        See notebooks/Velocity Obstacle.ipynb for a visual documentation.
    '''
    def __init__(self,pa,pb,ra,rb,tau):
        '''
            pa: agent a's location.
            pb: the neighboring agent b's location.
            ra: agent a's radius.
            rb: the neighoring agent b's radius.
            tau: the safe time interval. 
                 Agent a is not suppose to crash into b for the future tau seconds. 
        '''
        self.center = (pb-pa)/tau
        self.r = (ra+rb)/tau
        self.center_theta = np.arctan2(self.center[1],self.center[0])
        
        if la.norm(self.center)>0 and self.r <= la.norm(self.center):
            self.phi = np.arcsin(self.r/la.norm(self.center))

         
        self.center_line_theta = np.arctan2(-self.center[1],-self.center[0])
        
    def visualize(self, ax):
        '''
            Visualize the velocity obstacle.
        '''
        xlim = np.array((-1,1)) * la.norm(self.center) *2
        ylim = np.array((-1,1)) * la.norm(self.center) *2
        nx = 200
        ny = 200
        xv,yv = np.meshgrid(np.linspace(*xlim,nx),np.linspace(*ylim,ny))
        test_pt = np.vstack([xv.flatten(),yv.flatten()]).T

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if self.r <= la.norm(self.center): 
            circ = Circle(self.center,self.r,fc = 'green',alpha = 0.2)

            ax.add_artist(circ)

            ax.axline([0,0], slope = np.tan(self.center_theta-self.phi),ls='dotted')
            ax.axline([0,0], slope = np.tan(self.center_theta+self.phi),ls='dotted')


            zone_code = self.zones(test_pt)

            mk = ['o','o','x']

            color = ['blue','orange','grey']

            labels = ['VO Zone 0','VO Zone 1','Not in VO']

            for z in [0,1,2]:

                ax.scatter(test_pt[zone_code == z,0],test_pt[zone_code == z,1]\
                           ,s=3,marker=mk[z],fc='none',ec=color[z],label = labels[z])

        else:
            ax.scatter(0,0,alpha=0.0,label = 'Two agents overlap')

        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_aspect('equal')


    def zones(self,test_pt):
        '''
            Classify the test_pt in terms of its relationship to the velocity obstacle.
            
            Output: zone code. 
                   -1: the bloating regions of the two agents overlap. VO is the entire space. 
                    0: test_pt is in VO and is closest to the front arc of the circle.
                    1: test_pt in VO and is closest to one of the two sides.
                    2: test_pt is not in VO.
        '''


        zone_code = np.zeros(len(test_pt))
        
        if self.r > la.norm(self.center): 
            zone_code = -1
        else:
            in_circ = la.norm(test_pt - self.center,axis = 1)<=self.r

            in_squeeze = test_pt.dot(self.center)\
                        /(la.norm(test_pt,axis = 1) * la.norm(self.center)) >= np.cos(self.phi)

            far = np.logical_and(in_squeeze, 
                                 la.norm(test_pt,axis = 1)>= self.r/np.tan(self.phi))

            vo = np.logical_or(in_circ,far)

            in_front = (test_pt - self.center).dot(-self.center)\
                      /(la.norm(test_pt - self.center,axis = 1) * la.norm(self.center)) \
                      >= np.cos(np.pi/2-self.phi)

            zone0 = np.logical_and(vo,
                                   np.logical_and(in_circ, in_front))

            zone1 = np.logical_and(vo,
                                   np.logical_not(zone0))

            zone2 = np.logical_not(vo)

            zone_code[zone0] = 0
            zone_code[zone1] = 1
            zone_code[zone2] = 2

        return zone_code
