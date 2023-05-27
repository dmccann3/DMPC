import casadi as ca
import numpy as np
from casadi import pi

from params import Params


class tube_MPC(Params):

    def DM2Arr(self, dm):
        return np.array(dm.full())

    def constraints(self, xref, uref, xe):

        # get nominal dynamics
        f = self.dynamics()

        # cost function
        cost_fn = 0  
        # constraints in the equation
        g = self.Xtube[:, 0] - self.P[:self.n_states]  

        # weight matrices
        Q1 = 1.5     # x 
        Q2 = 1.5      # y
        Q3 = 1    # z 
        Q4 = 1      # phi 
        Q5 = 1      # theta
        Q6 = 1.0      # psi
        Q7 = 0.5      # vx
        Q8 = 0.5      # vy
        Q9 = 1.0      # vz
        Q10 = 1.0     # wx 
        Q11 = 1.0    # wy
        Q12 = 1.0    # wz
        Q = ca.diagcat(Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12)

        p = 1e-1
        R1 = 1
        R2 = 1
        R3 = 1
        R4 = 1
        R = p*ca.diagcat(R1, R2, R3, R4)

        # runge kutta
        for k in range(self.Ntube):
            st = self.Xtube[:, k]
            con = self.Utube[:, k]
            cost_fn = cost_fn \
                + (st - xref[self.n_states * k: (self.n_states * k) + self.n_states]).T @ Q \
                    @ (st - xref[self.n_states * k: (self.n_states * k) + self.n_states]) \
                + (con - uref[self.n_controls * k: (self.n_controls * k) + self.n_controls]).T @ R \
                    @ (con - uref[self.n_controls * k: (self.n_controls * k) + self.n_controls]) 
            st_next = self.Xtube[:, k+1]
            k1 = f(st, con)
            k2 = f(st + self.step_horizon/2*k1, con)
            k3 = f(st + self.step_horizon/2*k2, con)
            k4 = f(st + self.step_horizon * k3, con)
            st_next_RK4 = st + (self.step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

        # compute terminal cost and add it to cost function (CAN CHANGE Q HERE IF NEED BE)
        # st = self.Xtube[:, self.Ntube]
        # cost_fn = cost_fn  + ((st - xe).T @ Q @ (st - xe))

        # constraint bounds (only on control, no constraints on the state)
        lbx = ca.DM.zeros((self.n_states*(self.Ntube+1) + self.n_controls*self.Ntube, 1))
        ubx = ca.DM.zeros((self.n_states*(self.Ntube+1) + self.n_controls*self.Ntube, 1))
        bounds = [(-2, 2), (-2, 2), (0.01, 2), (-pi/3, pi/3), (-pi/3, pi/3), (-pi/2, pi/2), (-20, 20), (-20, 20), (-20, 20), (-pi/3, pi/3), (-pi/3, pi/3), (-pi/2, pi/2)]
        for i, (lower, upper) in enumerate(bounds):
            lbx[i: self.n_states * (self.Ntube + 1): self.n_states] = lower
            ubx[i: self.n_states * (self.Ntube + 1): self.n_states] = upper

        minspeed = 0.0
        lbx[self.n_states*(self.Ntube+1):] = minspeed
        maxspeed = 10 #krpm 
        ubx[self.n_states*(self.Ntube+1):] = maxspeed

        args = {
            'lbg': ca.DM.zeros((self.n_states*(self.Ntube+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((self.n_states*(self.Ntube+1), 1)) ,  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }

        return g, cost_fn, args


    # NEED TO EITHER UPDATE WHICH XREF AND UREF HERE OR SOMEWHERE ELSE
    def build_solver(self, xref, uref, xe):
        g, cost_fn, args = self.constraints(xref, uref, xe)

        OPT_variables = ca.vertcat(
            self.Xtube.reshape((-1, 1)),   
            self.Utube.reshape((-1, 1))
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

    
    def solve(self, state_init, state_target, solver, args):

        u0 = ca.DM.zeros((self.n_controls, self.Ntube))   
        X0 = ca.repmat(state_init, 1, self.Ntube+1)   # initial state full


        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )

        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, self.n_states*(self.Ntube+1), 1),
            ca.reshape(u0, self.n_controls*self.Ntube, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][self.n_states * (self.Ntube + 1):], self.n_controls, self.Ntube)

        return u[:, 0]

        


        


