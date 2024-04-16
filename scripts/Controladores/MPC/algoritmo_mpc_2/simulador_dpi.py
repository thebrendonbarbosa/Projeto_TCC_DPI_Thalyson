import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
#rel_do_mpc_path = os.path.join('..','..')
#sys.path.append(rel_do_mpc_path)
import do_mpc


def simulador_dpi(modelo):
    """
    --------------------------------------------------------------------------
    template_simulator: ajuste de parâmetros
    --------------------------------------------------------------------------
    """
    simulador = do_mpc.simulator.Simulator(modelo)

    params_simulador = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-8,
        'reltol': 1e-8,
        't_step': 0.04
    }

    simulador.set_param(**params_simulador)

    p_num = simulador.get_p_template()

    p_num['m1'] = 0.2
    p_num['m2'] = 0.2
    def p_fun(t_now):
        return p_num

    simulador.set_p_fun(p_fun)

    tvp_template = simulador.get_tvp_template()

    def tvp_fun(t_ind):
        return tvp_template

    simulador.set_tvp_fun(tvp_fun)


    simulador.setup()

    return simulador