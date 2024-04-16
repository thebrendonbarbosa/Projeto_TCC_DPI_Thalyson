import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
#rel_do_mpc_path = os.path.join('..','..')
#sys.path.append(rel_do_mpc_path)
import do_mpc


def mpc_dpi(modelo, silence_solver = False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(modelo)

    mpc.settings.n_horizon =  100
    mpc.settings.n_robust =  0
    mpc.settings.open_loop =  0
    mpc.settings.t_step =  0.04
    mpc.settings.state_discretization =  'collocation'
    mpc.settings.collocation_type =  'radau'
    mpc.settings.collocation_deg =  3
    mpc.settings.collocation_ni =  1
    mpc.settings.store_full_solution =  True

    if silence_solver:
        mpc.settings.supress_ipopt_output()


    mterm = modelo.aux['E_kin'] - modelo.aux['E_pot']
    lterm = -modelo.aux['E_pot']+10*(modelo.x['pos']-modelo.tvp['pos_set'])**2 # stage cost


    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(force=0.1)


    mpc.bounds['lower','_u','force'] = -4
    mpc.bounds['upper','_u','force'] = 4

    # Avoid the obstacles:
    mpc.set_nl_cons('obstacles', -modelo.aux['obstacle_distance'], 0)

    # Values for the masses (for robust MPC)
    m1_var = 0.2*np.array([1, 0.95, 1.05])
    m2_var = 0.2*np.array([1, 0.95, 1.05])
    mpc.set_uncertainty_values(m1=m1_var, m2=m2_var)


    tvp_template = mpc.get_tvp_template()

    # Quando mudar o ponto de ajuste:
    t_switch = 4    # seconds
    ind_switch = t_switch // mpc.settings.t_step

    def tvp_fun(t_ind):
        ind = t_ind // mpc.settings.t_step
        if ind <= ind_switch:
            tvp_template['_tvp',:, 'pos_set'] = -0.8
        else:
            tvp_template['_tvp',:, 'pos_set'] = 0.8
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc