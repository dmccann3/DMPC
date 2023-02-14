import casadi as ca
from casadi import pi
import numpy as np
from params import Params


class ref_MPC(Params):

    def DM2Arr(self, dm):
        return np.array(dm.full())

    def constraints(self):

        # get dynamics
        f = self.dynamics()

        # cost function
        cost_fn = 0  
        # constraints in the equation
        g = self.Xref[:, 0] - self.P[:self.n_states]  

        # weight matrices
        Q1 = 1.4      # x 
        Q2 = 1.4      # y
        Q3 = 1.4    # z 
        Q4 = 1      # phi 
        Q5 = 1      # theta
        Q6 = 0.6      # psi
        Q7 = 0.5      # vx
        Q8 = 0.5      # vy
        Q9 = 0.5      # vz
        Q10 = 0.5     # wx 
        Q11 = 0.5    # wy
        Q12 = 0.5    # wz
        Q = ca.diagcat(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12)

        p = 1e-1
        R1 = 1
        R2 = 1
        R3 = 1
        R4 = 1
        R = p*ca.diagcat(R1, R2, R3, R4)

        
        # runge kutta
        for k in range(self.Nref):
            st = self.Xref[:, k]
            con = self.Uref[:, k]
            cost_fn = cost_fn \
                + (st - self.P[self.n_states:]).T @ Q @ (st - self.P[self.n_states:]) \
                + (con - self.uref).T @ R @ (con - self.uref) 
            st_next = self.Xref[:, k+1]
            k1 = f(st, con)
            k2 = f(st + self.step_horizon/2*k1, con)
            k3 = f(st + self.step_horizon/2*k2, con)
            k4 = f(st + self.step_horizon * k3, con)
            st_next_RK4 = st + (self.step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

        
        # create bounds on state and control
        lbx = ca.DM.zeros((self.n_states*(self.Nref+1) + self.n_controls*self.Nref, 1))
        ubx = ca.DM.zeros((self.n_states*(self.Nref+1) + self.n_controls*self.Nref, 1))
        bounds = [(-3, 3), (-3, 3), (0.01, 3), (-pi/3, pi/3), (-pi/3, pi/3), (-pi, pi), (-1000, 1000), (-1000, 1000), (-100, 100), (-pi/3, pi/3), (-pi/3, pi/3), (-pi, pi)]
        for i, (lower, upper) in enumerate(bounds):
            lbx[i: self.n_states * (self.Nref + 1): self.n_states] = lower
            ubx[i: self.n_states * (self.Nref + 1): self.n_states] = upper

        minspeed = 0.0
        lbx[self.n_states*(self.Nref+1):] = minspeed
        maxspeed = 100 #krpm 
        ubx[self.n_states*(self.Nref+1):] = maxspeed

        args = {
            'lbg': ca.DM.zeros((self.n_states*(self.Nref+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((self.n_states*(self.Nref+1), 1)) ,  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }

        return g, cost_fn, args


    def build_solver(self):
        
        g, cost_fn, args = self.constraints()

        OPT_variables = ca.vertcat(
            self.Xref.reshape((-1, 1)),   
            self.Uref.reshape((-1, 1))
        )

        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': self.P
        }

        opts = {
            'ipopt': {
                'max_iter': 1000,
                'print_level': 2,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        return solver, args


    def solve_ref(self, state_init, state_target):

        # initial state and control
        u0 = ca.DM.zeros((self.n_controls, self.Nref)) 
        X0 = ca.repmat(state_init, 1, self.Nref+1)   
        solver, args = self.build_solver()
        
        # fill in arguements of casadi nlp solver
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        args['x0'] = ca.vertcat(
            ca.reshape(X0, self.n_states*(self.Nref+1), 1),
            ca.reshape(u0, self.n_controls*self.Nref, 1)
        )
        
        # get solution
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        

        u = ca.reshape(sol['x'][self.n_states * (self.Nref + 1):], self.n_controls, self.Nref)
        uref = u
        u_ = self.DM2Arr(uref)

        X0 = ca.reshape(sol['x'][: self.n_states * (self.Nref + 1)], self.n_states, self.Nref+1)
        xref = X0
        x_ = self.DM2Arr(X0)

        print('Reference trajectory calculated')
        x = []
        y = []
        z = []
        phi = []
        theta = []
        psi = []
        vx = []
        vy = []
        vz = []
        wx = []
        wy = []
        wz = []
        w1 = []
        w2 = []
        w3 = []
        w4 = []
        
        for i in range(x_.shape[1]-1):
            x.append(x_[0, i])
            y.append(x_[1, i])
            z.append(x_[2, i])
            phi.append(x_[3, i])
            theta.append(x_[4, i])
            psi.append(x_[5, i])
            vx.append(x_[6, i])
            vy.append(x_[7, i])
            vz.append(x_[8, i])
            wx.append(x_[9, i])
            wy.append(x_[10, i])
            wz.append(x_[11, i])

        for j in range(u_.shape[1]):
            w1.append(u_[0, j])
            w2.append(u_[1, j])
            w3.append(u_[2, j])
            w4.append(u_[3, j])
    

        print(f'x: {x[-1]}, y: {y[-1]}, z: {z[-1]}')
        print(f'phi: {phi[-1]}, theta: {theta[-1]}, psi:{psi[-1]}')
        print(f'vx: {vx[-1]}, vy: {vy[-1]}, vz:{vz[-1]}')
        print(f'wx: {wx[-1]}, wy: {wy[-1]}, wz:{wz[-1]}')
        print(len(x))

        ref_states_and_controls = {
        'xref': x,
        'yref': y,
        'zref': z,
        'phiref': phi,
        'thetaref': theta,
        'psiref': psi,
        'vxref': vx,
        'vyref': vy,
        'vzref': vz,
        'wxref': wx,
        'wyref': wy,
        'wzref': wz,
        'w1ref': w1,
        'w2ref': w2,
        'w3ref': w3,
        'w4ref': w4
    }

        return ref_states_and_controls, xref, uref