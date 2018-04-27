from mujoco_py import load_model_from_xml, load_model_from_path, MjSim, MjViewer
import math
import os
import numpy as np

from mujoco_py.modder import TextureModder

class manipulator():
    def __init__(self):

        self.DoF = DoF = 6
        self.model = load_model_from_path("jaco.xml")
        self.state_dim = DoF*3+3
        self.action_dim = DoF
        
        self.rx, self.ry, self.rz = 0.1, 0.4, 0.5
        
        self.sim = MjSim(self.model)
        self.sim_state = self.sim.get_state()
        
        self.viewer = MjViewer(self.sim)
        
    def reset(self):
        self.sim.set_state(self.sim_state)
        
        self.qvel=np.zeros( self.DoF )
        
        s = []
        for i in range(self.DoF):
            s.append( self.qvel[i] )
        for i in range(self.DoF):
            s.append( self.sim.data.qvel[i] )
        for i in range(self.DoF):
            s.append( self.sim.data.qpos[i] )
        s.append( self.rx )
        s.append( self.ry )
        s.append( self.rz )
        s = np.array(s)
        
        return s
    def step(self, a):
        for i in range(self.DoF):
            self.qvel[i] += a[i]
            
            # The limitation of velocity is important
            if self.qvel[i]>1.0:
                self.qvel[i] = 1.0
            if self.qvel[i]<-1.0:
                self.qvel[i] = -1.0
                
            self.sim.data.qvel[i] = self.qvel[i]    

        for i in [6,7,8]:
            self.sim.data.qvel[i] = 0
            self.sim.data.qpos[i] = 0
                
        self.sim.step()
        # if IS_RENDER: self.viewer.render()

        dis = np.linalg.norm(self.sim.data.sensordata-[self.rx, self.ry, self.rz])

        s = []
        for i in range(self.DoF):
            s.append( self.qvel[i] )
        for i in range(self.DoF):
            s.append( self.sim.data.qvel[i] )
        for i in range(self.DoF):
            s.append( self.sim.data.qpos[i] )
        s.append( self.rx )
        s.append( self.ry )
        s.append( self.rz )
        s = np.array(s)
        
        r = -(  dis+( 0.1*sum(abs(self.sim.data.qvel[:self.DoF])) + 0.01*sum(abs(self.qvel)) )*0.005 )
        d = 0
        info = [dis]
        return s, r, d, info