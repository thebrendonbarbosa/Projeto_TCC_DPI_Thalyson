import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
#rel_do_mpc_path = os.path.join('..','..')
#sys.path.append(rel_do_mpc_path)
import do_mpc


def modelo_dpi(obstaculos, symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    #model_type = 'continuous' # either 'discrete' or 'continuous'
    tipo_de_modelo = 'discrete' # Definir se é 'continuos' ou 'discrete'
     
    #model = do_mpc.model.Model(model_type, symvar_type)
    modelo = do_mpc.model.Model(tipo_de_modelo, symvar_type)

    # Parâmetros

    m = 1.0 # kg, massa do carro
    m1 = 0.4  # kg, massa do pêndulo inferiror
    m2 = 0.4  # kg, massa do pêndulo superiror

    L1 = 0.5 # m, comprimento do pêndulo inferior 
    L2 = 0.5 # m, comprimento do pêndulo superior
    m1 = modelo.set_variable('_p', 'm1')
    m2 = modelo.set_variable('_p', 'm2')
    g = 9.81 # m/s^2 , aceleração da gravidade   

    l1 = L1/2 # m,
    l2 = L2/2 # m,
    J1 = (m1 * l1**2) / 3   # Inercia
    J2 = (m2 * l2**2) / 3   # Inercia
    f0 = 0.01
    f1 = 0.007
    f2 = 0.007

    a1 = m+m1+m2
    a2 = l1*(m1+2*m2)
    a3 = m2*l2

    b2 = m1*l1**2 +J1+2*m2*l1**2
    b3 = 2*m2*l1*l2

    c3 = 2*m2*l2*l2+J2


    d3 = m2*g*l2 

    # Setpoint x:
    pos_set = modelo.set_variable('_tvp', 'pos_set')


    # Estrutura de estados (variáveis ​​de otimização):
    pos = modelo.set_variable('_x',  'pos')
    phi = modelo.set_variable('_x',  'phi', (2,1))
    dpos = modelo.set_variable('_x',  'dpos')
    dphi = modelo.set_variable('_x',  'dphi', (2,1))
    
    # Estados algébricos:
    ddpos = modelo.set_variable('_z', 'ddpos')
    ddphi = modelo.set_variable('_z', 'ddphi', (2,1))

    # Estrutura de entrada (variáveis ​​de otimização):
    u = modelo.set_variable('_u',  'force')

    # Equações diferenciais
    modelo.set_rhs('pos', dpos)
    modelo.set_rhs('phi', dphi)
    modelo.set_rhs('dpos', ddpos)
    modelo.set_rhs('dphi', ddphi)

    # Equações de Euler Lagrange para o sistema DIP (na forma f(x,u,z) = 0)
    euler_lagrange = vertcat(
        # 1
        a1*ddpos - ddphi[0]*a2*cos(phi[0]) - a3*ddphi[1]*cos(phi[1]) + dphi[0]**2*a2*sin(phi[0]) + a3*phi[1]**2*sin(phi[1]) +f0*dpos - u,
        # 2
        -ddpos*cos([0])*a2 + ddphi[0]*b2 + a3*ddphi[1]*cos(phi[0]-phi[1]) + (f1+f2)*dphi[0] - dphi[1]*(b3*sin(phi[1]-phi[0])-f2) - g*a2*sin(phi[0]),
        # 3
        -ddpos*a3*cos(phi[1]) + b3*ddphi[0]*cos(phi[0]-phi[1]) + ddphi[1]*c3 - dphi[0]*(f2+b3*dphi[0]*sin(phi[1]-phi[0]))+f2*dphi[1] - d3*sin(phi[1])
    )

    modelo.set_alg('euler_lagrange', euler_lagrange)

    # Expressões para energia cinética e potencial
    E_kin_cart = 1 / 2 * m * dpos**2

    E_kin_p1 = 1 / 2 * m1 * (
        (dpos + l1 * dphi[0] * cos(phi[0]))**2 +
        (l1 * dphi[0] * sin(phi[0]))**2) + 1 / 2 * J1 * dphi[0]**2

    E_kin_p2 = 1 / 2 * m2 * (
        (dpos + L1 * dphi[0] * cos(phi[0]) + l2 * dphi[1] * cos(phi[1]))**2 +
        (L1 * dphi[0] * sin(phi[0]) + l2 * dphi[1] * sin(phi[1]))**
        2) + 1 / 2 * J2 * dphi[0]**2

    E_kin = E_kin_cart + E_kin_p1 + E_kin_p2

    E_pot = m1 * g * l1 * cos(
    phi[0]) + m2 * g * (L1 * cos(phi[0]) +
                                l2 * cos(phi[1]))

    modelo.set_expression('E_kin', E_kin)
    modelo.set_expression('E_pot', E_pot)

    # Cálculos para evitar obstáculos:

    # Coordenadas dos nós:
    node0_x = modelo.x['pos']
    node0_y = np.array([0])

    node1_x = node0_x+L1*sin(modelo.x['phi',0])
    node1_y = node0_y+L1*cos(modelo.x['phi',0])

    node2_x = node1_x+L2*sin(modelo.x['phi',1])
    node2_y = node1_y+L2*cos(modelo.x['phi',1])

    distancia_obstaculo = []

    for obs in obstaculos:
        d0 = sqrt((node0_x-obs['x'])**2+(node0_y-obs['y'])**2)-obs['r']*1.05
        d1 = sqrt((node1_x-obs['x'])**2+(node1_y-obs['y'])**2)-obs['r']*1.05
        d2 = sqrt((node2_x-obs['x'])**2+(node2_y-obs['y'])**2)-obs['r']*1.05
        distancia_obstaculo.extend([d0, d1, d2])


    modelo.set_expression('obstacle_distance',vertcat(*distancia_obstaculo))
    modelo.set_expression('tvp', pos_set)


    # Build the model
    modelo.setup()

    return modelo