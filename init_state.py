import casadi as ca

class Init(object):

    def __init__(self):
        pass

    def initial(self, x, y, z):
        x_init = x
        y_init = y
        z_init = z
        psi_init = 0.
        theta_init = 0.
        phi_init = 0.
        vx_init = 0.
        vy_init = 0.
        vz_init = 0.
        wy_init = 0.
        wx_init = 0.
        wz_init = 0.
        state_init = ca.DM([x_init, y_init, z_init, psi_init, theta_init, phi_init, vx_init, vy_init, vz_init, wx_init, wy_init, wz_init]) 
        
        return state_init

    def target(self, x, y, z):
        x_target = x
        y_target = y
        z_target = z
        psie = 0.0
        thetae = 0.0
        phie = 0.0
        vxe = 0.0
        vye = 0.0
        vze = 0.0
        wye = 0.0
        wxe = 0.0
        wze = 0.0
        state_target = ca.DM([x_target, y_target, z_target, psie, thetae, phie, vxe, vye, vze, wxe, wye, wze])

        return state_target 