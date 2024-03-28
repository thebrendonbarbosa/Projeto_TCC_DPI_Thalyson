import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
#rel_do_mpc_path = os.path.join('..','..')
#sys.path.append(rel_do_mpc_path)
import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
# Plot settings
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'


import time

from mpc_dpi import mpc_dpi
from simulador_dpi import simulador_dpi
from modelo_dpi import modelo_dpi

""" User settings: """
show_animation = True
store_animation = False
store_results = False

# Definir obstáculos a evitar (ciclos)
obstaculos = [
    {'x': 0., 'y': 0.6, 'r': 0.3},
]

cenario = 1  # 1 = partida descendente, 2 = partida ascendente, ambos com alteração de setpoint.

"""
Obtenha módulos do-mpc configurados:
"""

modelo = modelo_dpi(obstaculos)
simulador = simulador_dpi(modelo)
mpc = mpc_dpi(modelo)
estimator = do_mpc.estimator.StateFeedback(modelo)

"""
Set initial state
"""

if cenario == 1:
    simulador.x0['phi'] = .9*np.pi
    simulador.x0['pos'] = 0
elif cenario == 2:
    simulador.x0['phi'] = 0.
    simulador.x0['pos'] = 0.8
else:
    raise Exception('Cenário não definido.')

x0 = simulador.x0.cat.full()

mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()
z0 = simulador.init_algebraic_variables()

"""
Gráfico de configuração:
"""

# Função para criar linhas:
L1 = 0.5  #m, comprimento da primeira haste
L2 = 0.5  #m, comprimento da segunda haste
def barras_pendulo(x):
    x = x.flatten()
    # Obtenha as coordenadas x,y das duas barras para o estado x fornecido.
    linha_1_x = np.array([
        x[0],
        x[0]+L1*np.sin(x[1])
    ])

    linha_1_y = np.array([
        0,
        L1*np.cos(x[1])
    ])

    linha_2_x = np.array([
        linha_1_x[1],
        linha_1_x[1] + L2*np.sin(x[2])
    ])

    linha_2_y = np.array([
        linha_1_y[1],
        linha_1_y[1] + L2*np.cos(x[2])
    ])

    linha_1 = np.stack((linha_1_x, linha_1_y))
    linha_2 = np.stack((linha_2_x, linha_2_y))

    return linha_1, linha_2

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

fig = plt.figure(figsize=(16,9))
plt.ion()

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1))
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax5 = plt.subplot2grid((4, 2), (3, 1))

ax2.set_ylabel('$E_{kin}$ [J]')
ax3.set_ylabel('$E_{pot}$ [J]')
ax4.set_ylabel('position  [m]')
ax5.set_ylabel('Input force [N]')

mpc_graphics.add_line(var_type='_aux', var_name='E_kin', axis=ax2)
mpc_graphics.add_line(var_type='_aux', var_name='E_pot', axis=ax3)
mpc_graphics.add_line(var_type='_x', var_name='pos', axis=ax4)
mpc_graphics.add_line(var_type='_tvp', var_name='pos_set', axis=ax4)
mpc_graphics.add_line(var_type='_u', var_name='force', axis=ax5)

ax1.axhline(0,color='black')

# Axis on the right.
for ax in [ax2, ax3, ax4, ax5]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    if ax != ax5:
        ax.xaxis.set_ticklabels([])

ax5.set_xlabel('time [s]')

barra1 = ax1.plot([],[], '-o', linewidth=5, markersize=10)
barra2 = ax1.plot([],[], '-o', linewidth=5, markersize=10)


for obs in obstaculos:
    circle = Circle((obs['x'], obs['y']), obs['r'])
    ax1.add_artist(circle)

ax1.set_xlim(-1.8,1.8)
ax1.set_ylim(-1.2,1.2)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()


"""
Run MPC main loop:
"""
time_list = []

n_steps = 240
for k in range(n_steps):
    tic = time.time()
    u0 = mpc.make_step(x0)
    toc = time.time()
    y_next = simulador.make_step(u0)
    x0 = estimator.make_step(y_next)

    time_list.append(toc-tic)


    if show_animation:
        linha1, linha2 = barras_pendulo(x0)
        barra1[0].set_data(linha1[0],linha1[1])
        barra2[0].set_data(linha2[0],linha2[1])
        mpc_graphics.plot_results()
        mpc_graphics.plot_predictions()
        mpc_graphics.reset_axes()
        plt.show()
        plt.pause(0.04)

time_arr = np.array(time_list)
mean = np.round(np.mean(time_arr[1:])*1000)
var = np.round(np.std(time_arr[1:])*1000)
print('mean runtime:{}ms +- {}ms for MPC step'.format(mean, var))


# A função que descreve o gif:
if store_animation:
    x_arr = mpc.data['_x']
    def update(t_ind):
        linha1, linha2 = barras_pendulo(x_arr[t_ind])
        barra1[0].set_data(linha1[0],linha1[1])
        barra2[0].set_data(linha2[0],linha2[1])
        mpc_graphics.plot_results(t_ind)
        mpc_graphics.plot_predictions(t_ind)
        mpc_graphics.reset_axes()

    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
    gif_writer = ImageMagickWriter(fps=20)
    anim.save('anim_dip.gif', writer=gif_writer)


# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulador], 'dip_mpc')

input('Press any key to exit.')