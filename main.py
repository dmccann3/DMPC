import numpy as np
import random as rand
import casadi as ca
import torch
import torch.nn as nn
import torch.optim as optim

from init_state import Init
from params import Params
from ref_mpc import ref_MPC
from tube_mpc import tube_MPC
from gen_disturbance import GenWind
from replay_buffer import Buffer
from adap_control import DNN
import adap_control
import plot


def main():

    state_init = Init.initial(Init, x=rand.randint(-2, 2), y=rand.randint(-2, 2), z=0.1)
    state_target = Init.target(Init, x=0, y=0, z=1)
    t0 = 0

    # get nominal dynamics
    params = Params()
    f_bar = params.dynamics()

    # create ref_mpc and make trajectory
    ref_mpc = ref_MPC()
    ref_states_and_controls, xref, uref = ref_mpc.solve_ref(state_init, state_target)

    # create tube_mpc and build solver
    tube_mpc = tube_MPC()

    # create wind model 
    wind_model = GenWind()

    # create buffer instance
    buffer = Buffer(params.buffer_size)

    # create adaptive network
    adap_model = DNN(params.input_size, params.hidden_size, params.last_hidden_size, params.output_size)
    optimizer = optim.SGD(adap_model.parameters(), params.lr)
    loss = nn.MSELoss()
    K_a = np.zeros(params.output_size)

    # list for storing final set of states and controls
    final_states = np.zeros((params.Nsim + 1, 12))
    final_states[0,:] = np.array(state_init.full())[:,0]
    final_controls = np.zeros((params.Nsim, 4))
    time = np.zeros(params.Nsim + 1)
    time[0] = t0

    for i in range(params.Nsim):

        # generate disturbance (for simulation)
        drag = wind_model.generate_wind(state_init)

        # solve adaptive control
        input = torch.from_numpy(np.array(state_init.full())).T
        adap_model.double()
        u_a, sigma = adap_model.forward(input)

        # add to buffer
        buffer.add(input, u_a)

        # solve mpc
        xsp = xref[:,i:i+params.Ntube]
        usp = uref[:,i:i+params.Ntube]
        solver, args = tube_mpc.build_solver(xsp, usp, state_target)
        u_mpc = tube_mpc.solve(state_init, state_target, solver, args)

        # sum controls
        u_a_tmp = u_a.detach().numpy()
        u_a_DM = ca.DM([[u_a_tmp[0, 0]], [u_a_tmp[0, 1]], [u_a_tmp[0, 2]], [u_a_tmp[0, 3]]])
        u_total = u_mpc + u_a_DM

        # simulate plant 
        # f_bar_val = f_bar(state_init, u_total)  + params.gox @ (drag) # for sim with tube mpc only
        f_val = f_bar(state_init, u_mpc)  + params.gox @ (u_a_DM + drag)
        state_init = ca.DM(state_init + (params.step_horizon *f_val))
        t0 = t0 + params.step_horizon

        # update outer layer wieghts
        delta = (params.gamma * sigma.detach().numpy() * (np.array(drag.full()) + u_a.detach().numpy().T))/(np.linalg.norm(sigma.detach().numpy())**2)
        K_a_bar = K_a + delta

        # check bounds on K
        if np.linalg.norm(K_a_bar) > params.K_max:
            K_a = (params.K_max / np.linalg.norm(K_a_bar)) * K_a_bar
        else:
            K_a = K_a_bar

        # update outer layer
        adap_model.adapt_outer_layer(-K_a)

        # collect final data
        state = np.array(state_init.full())
        control = np.array(u_total.full())
        final_states[i + 1,:] = state[:,0]
        final_controls[i,:] = control[:,0]
        time[i+1] = t0
        
        # train hidden layers
        if i != 0 and i % params.freq_ratio == 0:
            print('here')
            s, a = buffer.sample_batch(params.batch_size)
            output, losses = adap_control.train_layers(adap_model, optimizer, loss, s, a)


        print(i)

    print('Final States')
    print(f'x: {final_states[-1, 0]}, y: {final_states[-1, 1]}, z: {final_states[-1, 2]}')
    print(f'phi: {final_states[-1, 3]}, theta: {final_states[-1, 4]}, psi:{final_states[-1, 5]}')
    print(f'vx: {final_states[-1, 6]}, vy: {final_states[-1, 7]}, vz:{final_states[-1, 8]}')
    print(f'wx: {final_states[-1, 9]}, wy: {final_states[-1, 10]}, wz:{final_states[-1, 11]}')


    plot.plot(time, ref_states_and_controls, final_states, final_controls, params.Nsim)



if __name__ == '__main__':
    main()

