import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
import os

# Import do_mpc package:
import do_mpc

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

m0 = 1 # kg, massa do carro
m1 = 1 # kg, massa do pêndulo inferior
m2 = 1# kg, massa do pêndulo superior
L1 = 2*0.05 # m, comprimento do pêndulo inferior
L2 = 2*0.05 # m, comprimento do pêndulo superior

g = 9.80665 # m/s^2, aceleração da gravidade

f0 = 0.01 # Coef. Atrito do carro
f1 = 0.007 # Coef. atrito do pêndulo inferior
f2 = 0.007 # Coef. atrito do pêndulo superior

l1 = L1/2 # m
l2 = L2/2 #m
J1 = (m1*l1**2)/3  #Inércia
J2 = (m2*l2**2)/3  #Inércia

h1 = m0 + m1 + m2
h2 = l1*m1 + L1*m2
h3 = m2*l2


h4 = m1*l1*l1 + J1 + m2*4*l1*l1
h5 = 2*m2*l1*l2

h6 = m2*l2*l2 + J2

w1 = g*l1*(m1 + 2*m2)
w2 = g*m2*l2

pos = model.set_variable('_x', 'pos')
phi = model.set_variable('_x', 'phi', (2,1))
dpos = model.set_variable('_x', 'dpos')
dphi = model.set_variable('_x', 'dphi', (2,1))

u = model.set_variable('_u', 'force')

ddpos = model.set_variable('_z', 'ddpos')
ddphi = model.set_variable('_z', 'ddphi', (2,1))

model.set_rhs('pos', dpos)
model.set_rhs('phi', dphi)
model.set_rhs('dpos', ddpos)
model.set_rhs('dphi', ddphi)

euler_lagrange = vertcat(
    #1
    h1*ddpos + h2*ddphi[0]*cos(phi[0]) + h3*ddphi[1]*cos(phi[1])
    + f0*dpos - h2*dphi[0]**2*sin(phi[0]) + h3*dphi[1]**2*sin(phi[1]) - u,

    #2
    h2*cos(phi[0])*ddpos + h4*ddphi[0] + h5*cos(phi[0]-phi[1])*ddphi[1]
    + h5*dphi[1]**2*sin(phi[0]-phi[1]) - w1*sin(phi[0]) - f2*dphi[1] + (f1+f2)*dphi[0] ,
    
    #3
    h3*cos(phi[1])*ddpos + h5*cos(phi[0]-phi[1])*ddphi[0] + h6*ddphi[1]
    - h5*dphi[0]**2*sin(phi[0]-phi[1]) - w2*sin(phi[1]) + f2*dphi[0] - f2*dphi[1]
    )

########################################################################
model.set_alg('euler-lagrange', euler_lagrange)
print(euler_lagrange)

E_c = 1/2 * m0 * dpos**2

E_p1 =1 / 2 * m1 * (
    (dpos + l1 * dphi[0] * cos(phi[0]))**2 +
    (l1 * dphi[0] * sin(phi[0]))**2) + 1 / 2 * J1 * dphi[0]**2

E_p2 = 1 / 2 * m2 * (
    (dpos + L1 * dphi[0] * cos(phi[0]) + l2 * dphi[1] * cos(phi[1]))**2 +
    (L1 * dphi[0] * sin(phi[0]) + l2 * dphi[1] * sin(phi[1]))**
    2) + 1 / 2 * J2 * dphi[0]**2

E_ct = E_c + E_p1 + E_p2

print(E_ct)
E_pot = m1*g*l1*cos(phi[0]) + m2*g*(2*l1*cos(phi[0])+
              l2*cos(phi[1]))

model.set_expression('E_ct', E_ct)
model.set_expression('E_pot', E_pot)

# Construir o modelo
model.setup()
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 100,
    'n_robust': 0,
    'open_loop': 0,
    't_step': 0.01,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 1,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
}
mpc.set_param(**setup_mpc)


mterm = model.aux['E_ct'] - model.aux['E_pot'] # terminal cost
lterm = model.aux['E_ct'] - model.aux['E_pot'] # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

# Input force is implicitly restricted through the objective.
mpc.set_rterm(force=1)
mpc.bounds['lower','_u','force'] = -3
mpc.bounds['upper','_u','force'] = 3

mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)

simulator = do_mpc.simulator.Simulator(model)


params_simulator = {
    # Note: cvode doesn't support DAE systems.
    'integration_tool': 'idas',
    'abstol': 1e-8,
    'reltol': 1e-8,
    't_step': 0.01
}

simulator.set_param(**params_simulator)

simulator.setup()

simulator.x0['phi'] = np.deg2rad(5)#0.99*np.pi

x0 = simulator.x0.cat.full()

mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()


import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

def pendulum_bars(x):
    x = x.flatten()
    # Get the x,y coordinates of the two bars for the given state x.
    line_1_x = np.array([
        x[0],
        x[0]+L1*np.sin(x[1])
    ])

    line_1_y = np.array([
        0,
        L1*np.cos(x[1])
    ])

    line_2_x = np.array([
        line_1_x[1],
        line_1_x[1] + L2*np.sin(x[2])
    ])

    line_2_y = np.array([
        line_1_y[1],
        line_1_y[1] + L2*np.cos(x[2])
    ])

    line_1 = np.stack((line_1_x, line_1_y))
    line_2 = np.stack((line_2_x, line_2_y))

    return line_1, line_2


fig = plt.figure(figsize=(16,9))

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1))
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax5 = plt.subplot2grid((4, 2), (3, 1))

ax2.set_ylabel('$E_{ct}$ [J]')
ax3.set_ylabel('$E_{pot}$ [J]')
ax4.set_ylabel('Angulo  [rad]')
ax5.set_ylabel('Força de Entrada [N]')

# Axis on the right.
for ax in [ax2, ax3, ax4, ax5]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if ax != ax5:
        ax.xaxis.set_ticklabels([])

ax5.set_xlabel('time [s]')

mpc_graphics.add_line(var_type='_aux', var_name='E_ct', axis=ax2)
mpc_graphics.add_line(var_type='_aux', var_name='E_pot', axis=ax3)
mpc_graphics.add_line(var_type='_x', var_name='phi', axis=ax4)
mpc_graphics.add_line(var_type='_u', var_name='force', axis=ax5)

ax1.axhline(0,color='black')

bar1 = ax1.plot([],[], '-o', linewidth=5, markersize=10)
bar2 = ax1.plot([],[], '-o', linewidth=5, markersize=10)

ax1.set_xlim(-1.8,1.8)
ax1.set_ylim(-1.2,1.2)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()

u0 = mpc.make_step(x0)

line1, line2 = pendulum_bars(x0)
bar1[0].set_data(line1[0],line1[1])
bar2[0].set_data(line2[0],line2[1])
mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()

fig

import matplotlib.animation as animation

#%%capture
# Quickly reset the history of the MPC data object.
mpc.reset_history()

n_steps = 300
for k in range(n_steps):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)


from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter

# The function describing the gif:
x_arr = mpc.data['_x']
def update(t_ind):
    line1, line2 = pendulum_bars(x_arr[t_ind])
    bar1[0].set_data(line1[0],line1[1])
    bar2[0].set_data(line2[0],line2[1])
    mpc_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()


anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
gif_writer = animation.PillowWriter(fps=30)
anim.save('anim_meuDPI_10.gif', writer=gif_writer)


