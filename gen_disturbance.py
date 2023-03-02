import numpy as np
import casadi as ca
from casadi import sin


class GenWind(object):
    def __init__(self):
        self.Vm = 1
        self.dm = 3
        self.rho = 1


    def generate_wind(self, full_state):
        x = full_state[0]
        y = full_state[1]
        z = full_state[2]
        pos = np.zeros((3, 1))
        pos[0,0] = x
        pos[1,0] = y
        pos[2,0] = z
        vz = full_state[8]
        wx = full_state[9]
        wy = full_state[10]
        wz = full_state[11]
        controled_state = np.zeros((4,1))
        controled_state[0,0] = vz
        controled_state[1,0] = wx
        controled_state[2,0] = wy
        controled_state[3,0] = wz
        m = controled_state.shape[0]
        if np.linalg.norm(pos) > self.dm:
            wind = 0.25*self.Vm*np.ones((m, 1))
        elif np.linalg.norm(pos) >= 0 and np.linalg.norm(pos) <= self.dm:
            wind = 0.5*self.Vm*np.eye(m)@(1+np.cos(np.pi*controled_state/self.dm))
        else:
            wind = 0*self.Vm*np.ones((m, 1))
        drag = 0.5*self.rho*np.eye(m)@wind**2
        drag_ = ca.DM([[drag[0,0]], [drag[1,0]], [drag[2,0]], [drag[3,0]]])
        # noise = np.random.normal(0.0, 1.0, (4,1))
        # print(noise)
        # noise_ = ca.DM([[noise[0,0]], [noise[1,0]], [noise[2,0]], [noise[3,0]]])
        # disturbance = drag_ + noise_
        return drag_


    # use to generate wind field to plot (not necessary)
    def generate_field(self, xlims, ylims):
        x, y = np.meshgrid(np.linspace(xlims[0], xlims[1], 25), np.linspace(ylims[0], ylims[1], 25))
        m, n = x.shape
        z = np.ones((m, n))
        x_vel = np.zeros((m, n))
        y_vel = np.zeros((m, n))
        z_vel = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                state = np.array([[x[i, j]], [y[i, j]], [0]])
                if np.linalg.norm(state) > self.dm:
                    wind = 0.25*self.Vm*np.ones((3, 1))
                elif np.linalg.norm(state) >= 0 and np.linalg.norm(state) <= self.dm:
                    wind = 0.5*self.Vm*np.eye(3)*(1+np.cos(np.pi*state/self.dm))
                else:
                    wind = 0*self.Vm*np.ones((3, 1))
                x_vel[i, j] = wind[0, 0]
                y_vel[i, j] = wind[2, 0]
                x_vel[i, j] = wind[3, 0]

        return x_vel, y_vel, z_vel, x, y, z



class MassDist(object):

    def __init__(self):
        pass

    def disturb(self, full_state):

        x = full_state[0]
        y = full_state[1]
        z = full_state[2]
        pos = np.zeros((3, 1))
        pos[0,0] = x
        pos[1,0] = y
        pos[2,0] = z

        h = ca.DM([[1/sin(np.linalg.norm(pos) + 15)], [1/sin(np.linalg.norm(pos) + 15)], [1/sin(np.linalg.norm(pos) + 15)], [1/sin(np.linalg.norm(pos) + 15)]])
        
        return h
