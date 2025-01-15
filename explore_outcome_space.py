####################################################################################
# Explore the outcome space by evaluating (meaning NO optimisation) the bonding model
# on a large number of input points
####################################################################################
from ctpe_bonding_model_utils import *

run_wo_optimisation(material = 'ABS',
                    obj_fun = wrapper_obj_fun_fac('ABS'),
                    config_space = create_config_space('ABS'),
                    n_evaluations = 10000)

run_wo_optimisation(material = 'PPS',
                    obj_fun = wrapper_obj_fun_fac('PPS'),
                    config_space = create_config_space('PPS'),
                    n_evaluations = 10000)

run_wo_optimisation(material = 'GFRE',
                    obj_fun = wrapper_obj_fun_fac('GFRE'),
                    config_space = create_config_space('GFRE'),
                    n_evaluations = 10000)