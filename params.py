import casadi as ca
from casadi import sin, cos, tan, inv


class Params(object):

    def __init__(self):

        self.step_horizon = 0.05  # time between steps in seconds
        self.Nref = 200
        self.Ntube = 10      # number of look ahead steps
        self.Nsim = 190
        self.sim_time = 200  

        # state variables
        self.x = ca.SX.sym('x')          # (m)
        self.y = ca.SX.sym('y')          # (m)
        self.z = ca.SX.sym('z')          # (m)
        self.phi = ca.SX.sym('phi')      # (rad)
        self.theta = ca.SX.sym('theta')  # (rad)
        self.psi = ca.SX.sym('psi')      # (rad)
        self.vx = ca.SX.sym('vx')        # (m/s)
        self.vy = ca.SX.sym('vy')        # (m/s)
        self.vz = ca.SX.sym('vz')        # (m/s)
        self.wx = ca.SX.sym('wx')        # (rad/s)
        self.wy = ca.SX.sym('wy')        # (rad/s)
        self.wz = ca.SX.sym('wz')        # (rad/s)
        self.states = ca.vertcat(self.x, self.y, self.z, self.phi, self.theta, self.psi, self.vx, self.vy, self.vz, self.wx, self.wy, self.wz)
        self.n_states = self.states.numel()
        self.p = ca.vertcat(
            self.x, self.y, self.z
        )
        self.q = ca.vertcat(
            self.phi, self.theta, self.psi
        )
        self.vb = ca.vertcat(
            self.vx, self.vy, self.vz
        )
        self.w = ca.vertcat(
            self.wx, self.wy, self.wz
        )
        # control variables
        self.w1 = ca.SX.sym('w1')        # (krpm)
        self.w2 = ca.SX.sym('w2')        # (krpm)
        self.w3 = ca.SX.sym('w3')        # (krpm)
        self.w4 = ca.SX.sym('w4')        # (krpm)
        self.controls = ca.vertcat(self.w1, self.w2, self.w3, self.w4)
        self.n_controls = self.controls.numel()

        # matrix containing all states over all time steps +1 (each column is a state vector)
        self.Xref = ca.SX.sym('X', self.n_states, self.Nref + 1)
        # matrix containing all control actions over all time steps (each column is an action vector)
        self.Uref = ca.SX.sym('U', self.n_controls, self.Nref)
        # coloumn vector for storing initial state and target state
        self.P = ca.SX.sym('P', self.n_states + self.n_states)

        # matrix containing all states over all time steps +1 (each column is a state vector)
        self.Xtube = ca.SX.sym('X', self.n_states, self.Ntube + 1)
        # matrix containing all control actions over all time steps (each column is an action vector)
        self.Utube = ca.SX.sym('U', self.n_controls, self.Ntube)

        # physical parameters
        self.g = 9.81 # m/s2
        self.m = 0.0719/2 # kg
        self.l = 0.0325 # m
        self.km = 1.8580e-5
        self.kf = 0.005022
        self.Jxx = (2.3951e-5) # kg/m^2
        self.Jyy = (2.3951e-5) # kg/m^2
        self.Jzz = (3.2347e-5) # kg/m^2
        self.J = ca.DM([
            [self.Jxx, 0, 0], 
            [0, self.Jyy, 0],
            [0, 0, self.Jzz]
        ])

        # control influence matrix
        self.gox = ca.DM.zeros((12, 4))
        self.gox[8, :] = (self.kf*self.l)/self.m
        self.gox[9, 0] = -(self.kf*self.l)/self.Jxx
        self.gox[9, 1] = -(self.kf*self.l)/self.Jxx
        self.gox[9, 2] = (self.kf*self.l)/self.Jxx
        self.gox[9, 3] = (self.kf*self.l)/self.Jxx
        self.gox[10, 0] = -(self.kf*self.l)/self.Jyy
        self.gox[10, 1] = (self.kf*self.l)/self.Jyy
        self.gox[10, 2] = (self.kf*self.l)/self.Jyy
        self.gox[10, 3] = -(self.kf*self.l)/self.Jyy
        self.gox[11, 0] = -(self.kf*self.l)/self.Jzz
        self.gox[11, 1] = (self.kf*self.l)/self.Jzz
        self.gox[11, 2] = -(self.kf*self.l)/self.Jzz
        self.gox[11, 3] = (self.kf*self.l)/self.Jzz

        # control equilibrium (hover speed of rotor krpm)
        self.uref = ca.vertcat(
            4.190,
            4.190,
            4.190,
            4.190
        )

        # adaptive control network params
        self.lr = 0.01
        self.gamma = 0.7
        self.input_size = 12
        self.hidden_size = 256
        self.last_hidden_size = 4
        self.output_size = 4
        self.batch_size = 32

        # buffer params
        self.buffer_size = 100

        # K_max for weight update of DNN (for now leave as such)
        self.K_max = 2.0
        self.freq_ratio = 30



    def dynamics(self):

        # forces 
        Fz = self.kf*(self.w1**2 + self.w2**2 + self.w3**2 + self.w4**2)
        Fb = ca.vertcat(
            0,
            0,
            Fz
        )
        # moments
        Mx = self.kf*self.l*(-self.w1**2 - self.w2**2 + self.w3**2 + self.w4**2)
        My = self.kf*self.l*(-self.w1**2 + self.w2**2 + self.w3**2 - self.w4**2)
        Mz = self.km*(-self.w1**2 + self.w2**2 - self.w3**2 + self.w4**2)
        Mb = ca.vertcat(
            Mx, My, Mz
        )
        # gravity
        accel = ca.vertcat(
            0,
            0,
            self.g
        )

        # rotation matrices
        Rphi = ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, cos(self.phi), sin(self.phi)),
            ca.horzcat(0, -sin(self.phi), cos(self.phi))
        )
        Rtheta = ca.vertcat(
            ca.horzcat(cos(self.theta), 0, -sin(self.theta)),
            ca.horzcat(0, 1, 0),
            ca.horzcat(sin(self.theta), 0, cos(self.theta))
        )
        Rpsi = ca.vertcat(
            ca.horzcat(cos(self.psi), sin(self.psi), 0),
            ca.horzcat(-sin(self.psi), cos(self.psi), 0),
            ca.horzcat(0, 0, 1)
        )
        R = Rpsi @ Rtheta @ Rphi
        R2_ = ca.vertcat(
            ca.horzcat(1, sin(self.phi)*tan(self.theta), cos(self.phi)*tan(self.theta)),
            ca.horzcat(0, cos(self.phi), -sin(self.phi)),
            ca.horzcat(0, sin(self.phi)/cos(self.theta), cos(self.phi)/cos(self.theta))
        )

        # equations of motion
        position_eqs = R@self.vb
        orientation_eqs = R2_@self.w
        velocity_eqs = (1/self.m)*(Fb) - (R.T@accel) - (self.w*self.vb)
        angular_eqs = inv(self.J) @ (Mb - (self.w * self.J@self.w))
        RHS = ca.vertcat(
            position_eqs,
            orientation_eqs,
            velocity_eqs,
            angular_eqs
        )

        # final casadi function for xdot
        f = ca.Function('f', [self.states, self.controls], [RHS])

        return f


    
    


        