import numpy as np
import os
import sys
# Constrained TPE code repo doesn't install anything into "...\envs\a_repo_name\Lib\site-packages
# so every module, function used here is accessed from this current project folder
# this behaviour is different than in plain vanilla TPE code repo
# nevertheless, to be on the safe side, I modify the search path by inserting this project folder in the very first position
project_path = os.path.dirname(os.path.abspath('something_meaningless'))
sys.path.insert(0, os.path.dirname(os.path.abspath(project_path)))
from ctpe_bonding_model_utils import *

n_macroreps = 30
# Set seed and draw some large integers which will be used as seeds for optimisation runs (macro reps)
np.random.seed(389092)
seed_list = np.random.randint(0, 10 ** 6, size=n_macroreps)

#####################################################################################
# Run the optimisation n_macroreps times, for each material separately
#####################################################################################

# Note1: from the matlab code of the bonding simulator, it follows that for each material only selected scenarios
# are available
# Note2: 'Aluminum' was excluded from the analysis because of very discrete nature of results (only ~10 distinct
# points compared to hundreds, thousands for other materials)

config_ABS = create_config_space('ABS', ['1', '2', '3'])
run_optimisation(material='ABS',
                 obj_fun=wrapper_obj_fun_fac('ABS'),
                 config_space=config_ABS,
                 seed_list=seed_list,
                 n_init=10 * len(config_ABS),
                 n_iterations=200,
                 n_macroreps=n_macroreps,
                 save_iteration_history=True)

config_PPS = create_config_space('PPS', ['1', '2', '3'])
run_optimisation(material='PPS',
                 obj_fun=wrapper_obj_fun_fac('PPS'),
                 config_space=config_PPS,
                 seed_list=seed_list,
                 n_init=10 * len(config_PPS),
                 n_iterations=200,
                 n_macroreps=n_macroreps,
                 save_iteration_history=True)

config_GFRE = create_config_space('GFRE', ['1', '2', '3', '4', '5', '6'])
run_optimisation(material='GFRE',
                 obj_fun=wrapper_obj_fun_fac('GFRE'),
                 config_space=config_GFRE,
                 seed_list=seed_list,
                 n_init=10 * len(config_GFRE),
                 n_iterations=200,
                 n_macroreps=n_macroreps,
                 save_iteration_history=True)