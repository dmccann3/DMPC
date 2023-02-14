import matplotlib.pyplot as plt
import numpy as np


def plot(time, ref_states_and_controls, final_states, final_controls, Nsim):

    # plot results
    # Just starting plots (Need to include comparisons to shallow mpc, regular mpc, and reference governer)
    fig, (ax_posX, ax_posY, ax_posZ) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_posX.plot(time, ref_states_and_controls['xref'][:Nsim+1], label='x_ref')
    ax_posX.plot(time, final_states[:, 0], label='x')

    ax_posX.set_ylabel('Position (m)')
    ax_posX.set_ylim(-2.5, 2.5)
    ax_posX.legend()
    ax_posX.grid()
    ax_posY.plot(time, ref_states_and_controls['yref'][:Nsim+1], label='y_ref')
    ax_posY.plot(time, final_states[:,1], label='y')
    
    ax_posY.set_ylabel('Position (m)')
    ax_posY.set_ylim(-2.5, 2.5)
    ax_posY.legend()
    ax_posY.grid()
    ax_posZ.plot(time, ref_states_and_controls['zref'][:Nsim+1], label='z_ref')
    ax_posZ.plot(time, final_states[:, 2], label='z')
    
    ax_posZ.set_ylabel('Position (m)')
    ax_posZ.set_ylim(0, 2)
    ax_posZ.legend()
    ax_posZ.grid()
    ax_posZ.set_xlabel('time (s)')
    # fig.savefig('Plots/ref_test_pos.png')
    fig.savefig('Plots/dmpc_test_pos.png')


    fig2, (ax_oriX, ax_oriY, ax_oriZ) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_oriX.plot(time, ref_states_and_controls['phiref'][:Nsim+1], label='phi_ref')
    ax_oriX.plot(time, final_states[:, 3], label='phi')
    
    ax_oriX.set_ylabel('Orientation (rad)')
    ax_oriX.set_ylim(-3.14/2, 3.14/2)
    ax_oriX.legend()
    ax_oriX.grid()
    ax_oriY.plot(time, ref_states_and_controls['thetaref'][:Nsim+1], label='theta_ref')
    ax_oriY.plot(time, final_states[:, 4], label='theta')
    
    ax_oriY.set_ylabel('Orientation (rad)')
    ax_oriY.set_ylim(-3.14/2, 3.14/2)
    ax_oriY.legend()
    ax_oriY.grid()
    ax_oriZ.plot(time, ref_states_and_controls['psiref'][:Nsim+1], label='psi_ref')
    ax_oriZ.plot(time, final_states[:, 5], label='psi')
    
    ax_oriZ.set_ylabel('Orientation (rad)')
    ax_oriZ.set_ylim(-6.28/2, 6.28/2)
    ax_oriZ.legend()
    ax_oriZ.grid()
    ax_oriZ.set_xlabel('time (s)')
    # fig2.savefig('Plots/ref_test_ori.png')
    fig2.savefig('Plots/dmpc_test_ori.png')


    fig3, (ax_velX, ax_velY, ax_velZ) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_velX.plot(time, ref_states_and_controls['vxref'][:Nsim+1], label='vx_ref')
    ax_velX.plot(time, final_states[:, 6], label='vx')
   
    ax_velX.set_ylabel('Velocity (m/s)')
    ax_velX.legend()
    ax_velX.grid()
    ax_velY.plot(time, ref_states_and_controls['vyref'][:Nsim+1], label='vy_ref')
    ax_velY.plot(time, final_states[:, 7], label='vy')

    ax_velY.set_ylabel('Velocity (m/s)')
    ax_velY.legend()
    ax_velY.grid()
    ax_velZ.plot(time, ref_states_and_controls['vzref'][:Nsim+1], label='vz_ref')
    ax_velZ.plot(time, final_states[:, 8], label='vz')

    ax_velZ.set_ylabel('Velocity (m/s)')
    ax_velZ.legend()
    ax_velZ.grid()
    ax_velZ.set_xlabel('time (s)')
    # fig3.savefig('Plots/ref_test_vel.png')
    fig3.savefig('Plots/dmpc_test_vel.png')


    fig4, (ax_angX, ax_angY, ax_angZ) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_angX.plot(time, ref_states_and_controls['wxref'][:Nsim+1], label='wx_ref')
    ax_angX.plot(time, final_states[:,9], label='wx')

    ax_angX.set_ylabel('Velocity (rad/s)')
    ax_angX.legend()
    ax_angX.grid()
    ax_angY.plot(time, ref_states_and_controls['wyref'][:Nsim+1], label='wy_ref')
    ax_angY.plot(time, final_states[:,10], label='wy')

    ax_angY.set_ylabel('Velocity (rad/s)')
    ax_angY.legend()
    ax_angY.grid()
    ax_angZ.plot(time, ref_states_and_controls['wzref'][:Nsim+1], label='wz_ref')
    ax_angZ.plot(time, final_states[:,11], label='wz')

    ax_angZ.set_ylabel('Velocity (rad/s)')
    ax_angZ.legend()
    ax_angZ.grid()
    ax_angZ.set_xlabel('time (s)')
    # fig4.savefig('Plots/ref_test_ang.png')
    fig4.savefig('Plots/dmpc_test_ang.png')


    fig5, (ax_rpm) = plt.subplots(1, 1, figsize=(12, 12), sharex=True)
    ax_rpm.plot(time[:-1], final_controls[:, 0], label='w1')
    ax_rpm.plot(time[:-1], final_controls[:, 1], label='w2')
    ax_rpm.plot(time[:-1], final_controls[:, 2], label='w3')
    ax_rpm.plot(time[:-1], final_controls[:, 3], label='w4')

    ax_rpm.plot(time[:-1], ref_states_and_controls['w1ref'][:Nsim], label='w1', color='black')
    ax_rpm.plot(time[:-1], ref_states_and_controls['w2ref'][:Nsim], label='w2', color='black')
    ax_rpm.plot(time[:-1], ref_states_and_controls['w3ref'][:Nsim], label='w3', color='black')
    ax_rpm.plot(time[:-1], ref_states_and_controls['w4ref'][:Nsim], label='w4', color='black')
    # ax_rpm.plot([0, t[-1]], [4.190, 4.190], label='Trim Condition')
    ax_rpm.set_ylabel('Rotor speed (krpm)')
    ax_rpm.legend()
    ax_rpm.grid()
    ax_rpm.set_xlabel('time (s)')
    # fig5.savefig('Plots/ref_test_rpm.png')
    fig5.savefig('Plots/dmpc_test_rpm.png')