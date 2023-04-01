import numpy as np
from matplotlib.patches import Circle


class VO:
    '''
        The class for velocity obstacle.
        See notebooks/Velocity Obstacle.ipynb for a visual documentation.
    '''
    def __init__(self,pa,pb,ra,rb,tau):
        '''
            pa: the agent's location.
            pb: the neighboring agent's ocation.
            ra: the agent's radius.
            rb: the neighoring agent's radius.
            tau: the safe time interval. 
                 Agent a is not suppose to crash into b for the future tau seconds.
        '''
        self.center = (pb-pa)/tau
        self.r = (ra+rb)/tau
        self.center_theta = np.arctan2(self.center[1],self.center[0])
        self.phi = np.arcsin(self.r/np.linalg.norm(self.center))
         
        self.center_line_theta = np.arctan2(-self.center[1],-self.center[0])
        
    def visualize(self, ax):
        '''
            Visualize the velocity obstacle.
        '''

        circ = Circle(self.center,self.r,fc = 'green',alpha = 0.2)

        ax.add_artist(circ)

        ax.axline([0,0], slope = np.tan(self.center_theta-self.phi),ls='dotted')
        ax.axline([0,0], slope = np.tan(self.center_theta+self.phi),ls='dotted')

        ax.grid(True)
        ax.set_aspect('equal')

    def zones(self,test_pt):
        '''
            Classify the test_pt in terms of its relationship to the velocity obstacle.
            
            Output: zone code. 
                    0: in VO and is closest to the front arc of the circle.
                    1: in VO and is closest to one of the two sides.
                    2: not in VO.
        '''


        in_circ = np.linalg.norm(test_pt - self.center,axis = 1)<=self.r

        in_squeeze = np.abs(np.arctan2(test_pt[:,1],test_pt[:,0]) \
                            - self.center_theta) <= self.phi

        far = np.logical_and(in_squeeze, 
                             np.linalg.norm(test_pt,axis = 1)>= self.r/np.tan(self.phi))

        vo = np.logical_or(in_circ,far)


        theta = np.arctan2((test_pt - self.center)[:,1],\
                         (test_pt - self.center)[:,0] )

        in_front = np.abs(theta-self.center_line_theta)<= np.pi/2-self.phi

        zone0 = np.logical_and(vo,
                               np.logical_and(in_circ, in_front))

        zone1 = np.logical_and(vo,
                               np.logical_not(zone0))

        zone2 = np.logical_not(vo)

        zone_code = np.zeros(len(test_pt))

        zone_code[zone0] = 0
        zone_code[zone1] = 1
        zone_code[zone2] = 2

        return zone_code
