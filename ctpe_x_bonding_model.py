import matlab.engine
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition, InCondition
import numpy as np
import pandas as pd
import os
from util.utils import get_logger
from optimizer import TPEOptimizer
import json

# # any changes that I introduce to the TPE implementation are made in the files stored in the location below
# # the command below causes that the files from that location will be automatically used while calling TPE functions
# sys.path.extend(['./scripts/tpe-single-opt_original/tpe'])
import sys
sys.path.extend(['./scripts/'])

n_macroreps = 1
project_path = 'C:/git/constrained-tpe-main'

# set seed and draw some large integers which will be used as seeds for optimisation runs
np.random.seed(389092)
seed_list = np.random.randint(0, 10**6, size = n_macroreps)

# Start MATLAB engine
eng = matlab.engine.start_matlab()

eng.addpath("C:\\git\\TPE\\scripts\\bonding_model\\v2_0")

def get_misc_vars(eval_config):

    if eval_config['scenario'] in ['2', '3', '4', '5', '6']:

        if eval_config.get('pretreatment') == 'dry_tissue':
            dry_tissue = 1
        else:
            dry_tissue = 0

        if eval_config.get('pretreatment') == 'compressed_air':
            compressed_air = 1
        else:
            compressed_air = 0

        if eval_config.get('pretreatment') == 'US_bath':
            US_bath = 1
        else:
            US_bath = 0

        if eval_config.get('pretreatment') == 'degreasing':
            degreasing = 1
        else:
            degreasing = 0

        pretreatment = 1
    else:
        pretreatment = 0
        dry_tissue = 0
        compressed_air = 0
        US_bath = 0
        degreasing = 0

    if eval_config['scenario'] in ['3', '6']:
        plasma = 1
        plasma_power_value = float(eval_config['plasma_power'])
        plasma_speed_value = float(eval_config['plasma_speed'])
        plasma_distance_value = float(eval_config['plasma_distance'])
        plasma_passes_value = int(eval_config['plasma_passes'])
    else:
        plasma = 0
        # assigning arbitrary value below (e.g. the lower bound) as the value won't actually be used in outcome calculation
        plasma_power_value = 300.0
        plasma_speed_value = 5.0
        plasma_distance_value = 4.0
        plasma_passes_value = 1

    if eval_config['scenario'] in ['4', '6']:
        roughening = 1
    else:
        roughening = 0

    if eval_config['scenario'] in ['5', '6']:
        posttreatment = 1
    else:
        posttreatment = 0

    var_dict = {'pretreatment': pretreatment,
                'dry_tissue': dry_tissue,
                'compressed_air': compressed_air,
                'US_bath': US_bath,
                'degreasing': degreasing,
                'plasma': plasma,
                'plasma_power_value': plasma_power_value,
                'plasma_speed_value': plasma_speed_value,
                'plasma_distance_value': plasma_distance_value,
                'plasma_passes_value': plasma_passes_value,
                'roughening': roughening,
                'posttreatment': posttreatment}
    return var_dict

# function factory is a workaround for not knowing how to pass multiple arguments to the objective function.
# 'material' variable is required by the bonding simulator, but it's not an optimisation parameter and can't be found in eval_config arg

def wrapper_obj_fun_fac(input_material):
    def wrapper_obj_fun(eval_config):

        material = input_material

        tmp_dict = get_misc_vars(eval_config)

        pretreatment = tmp_dict.get('pretreatment')
        dry_tissue = tmp_dict.get('dry_tissue')
        compressed_air = tmp_dict.get('compressed_air')
        US_bath = tmp_dict.get('US_bath')
        degreasing = tmp_dict.get('degreasing')
        plasma = tmp_dict.get('plasma')
        plasma_power_value = tmp_dict.get('plasma_power_value')
        plasma_speed_value = tmp_dict.get('plasma_speed_value')
        plasma_distance_value = tmp_dict.get('plasma_distance_value')
        plasma_passes_value = tmp_dict.get('plasma_passes_value')
        roughening = tmp_dict.get('roughening')
        posttreatment = tmp_dict.get('posttreatment')

        scenario = int(eval_config['scenario'])
        glue_type = 'Araldite'
        sample_size = 25
        time_between_plasma_glue = 1
        curing_time = 30
        curing_temperature = 180
        batch_size = 30
        number_repetitions = 5
        Width_plasma = 2
        general_noise = 0
        noise_factor_plasma = 0
        noise_curing = 0.005
        noise_material = 0
        wt_particles = 30  # took the mid-point of the provided range
        curing_method = 'oven'
        ind_current_bonding = 17.5  # took the mid-point of the provided range
        ind_current_debonding = 0
        ind_time_debonding = 0

        tensileStrength, failureMode,VisualQ, cost, Feasibility, FinalcontactAngle = eng.bondingModel2(eng.double(scenario), eng.double(pretreatment),
                                                                                                       eng.double(posttreatment),
                                                                                                       material,
                                 eng.double(dry_tissue),
                                 eng.double(compressed_air),
                                 eng.double(US_bath),
                                 eng.double(degreasing),
                                 eng.double(roughening),
                                 glue_type,
                                 eng.double(sample_size),
                                 eng.double(plasma),
                                 eng.double(plasma_power_value),
                                 eng.double(plasma_speed_value),
                                 eng.double(plasma_distance_value),
                                 eng.double(plasma_passes_value),
                                 eng.double(time_between_plasma_glue),
                                 eng.double(curing_time),
                                 eng.double(curing_temperature),
                                 eng.double(batch_size),
                                 eng.double(number_repetitions),
                                 eng.double(Width_plasma),
                                 eng.double(general_noise),
                                 eng.double(noise_factor_plasma),
                                 eng.double(noise_curing),
                                 eng.double(noise_material),
                                 eng.double(wt_particles),
                                 curing_method,
                                 eng.double(ind_current_bonding),
                                 eng.double(ind_current_debonding),
                                 eng.double(ind_time_debonding),
                                 nargout=6)
        return dict(loss = -tensileStrength, c1 = VisualQ, failureMode = failureMode, VisualQ = VisualQ, cost = cost,
                    Feasibility = Feasibility, FinalcontactAngle = FinalcontactAngle)
    return wrapper_obj_fun

def run_bonding_model(obj_fun, obj_fun_name, cs, material, seed_list, n_init, n_iterations, n_macroreps, project_path, suffix, save_iteration_history=False):

    d = len(cs)

    # create objects for storing results
    outcomes = []
    saved_x = []

    for r in range(n_macroreps):

        nm = f'ctpe_bondingmodel_{suffix}_macrorep{r + 1}'
        logger = get_logger(file_name = nm, logger_name= nm)

        opt = TPEOptimizer(obj_func= obj_fun,
                           config_space=cs,
                           constraints={'c1': 1},
                           seed=int(seed_list[r]),
                           n_init = n_init,
                           max_evals = n_iterations,
                           resultfile = nm)

        #opt2 = opt.optimize(logger)

        best_config, best_primary, best_additional = opt.optimize(logger)

        # extract optimal x's
        temp = opt2[0]
        tmp_dict = get_misc_vars(temp)
        saved_x.append(temp | tmp_dict)

        opt.additional_metrics

        #outcomes.append([-opt2[1]] + [tensileStrength, failureMode, VisualQ, cost, Feasibility, FinalcontactAngle]) # TPE searches for minimum (but tensileStrength should be maximised), so the wrapper_obj_fun returns minus tensile strength, this is why minus sign here

        print(f'material={material}; macrorep={r+1}; VisualQ={VisualQ}; best_loss={-opt2[1]}')


    saved_x_df = pd.DataFrame(saved_x)
    saved_x_df = saved_x_df[sorted(saved_x_df.columns)]

    outcomes_df = pd.DataFrame(outcomes, columns=['loss', 'tensileStrength', 'failureMode', 'VisualQ', 'cost', 'Feasibility', 'FinalcontactAngle'])
    outcomes_df['seed'] = seed_list

    results_both = pd.concat([saved_x_df, outcomes_df], axis=1)
    results_both.to_excel(f'{project_path}/output/data/adhesive_bonding/{obj_fun_name}_{suffix}_solution.xlsx', index=False)

    if save_iteration_history:
        # Save intermediate optimization results (all macroreps in one file) into CSV
        iteration_history = pd.DataFrame()

        for r in range(1, n_macroreps + 1):
            nm = f'ctpe_bondingmodel_{suffix}_macrorep{r}'

            with open('./results/' + nm + '.json', 'r') as file:
                temp_intermediate_data = json.load(file)

            temp_intermediate_data['macrorep'] = [r for _ in range(1, n_iterations + 1)]
            temp_intermediate_data['iteration'] = [i for i in range(1, n_iterations + 1)]
            temp_intermediate_data['loss'] = list(np.array(-1) * temp_intermediate_data['loss'])

            temp_intermediate_data = pd.DataFrame(temp_intermediate_data)

            # Calculate cumulative maximum loss considering only rows with c1 == 1
            temp_intermediate_data['cumulative_max_loss'] = (
                temp_intermediate_data[temp_intermediate_data['c1'] == 1]['loss']
                .cummax()
                .reindex(temp_intermediate_data.index, method='ffill')  # Fill NaN for rows where c1 != 1
                .fillna(0)  # Fill initial rows with 0 if no c1 == 1 value is present yet
            )

            # Derive c1_cumulative_max_loss (c1 corresponding to cumulative_max_loss)
            temp_intermediate_data['c1_cumulative_max_loss'] = temp_intermediate_data.apply(
                lambda row: 1 if row['loss'] == row['cumulative_max_loss'] and row['c1'] == 1 else None,
                axis=1
            ).fillna(method='ffill').fillna(0).astype(int)

            # Derive best_row_number: row number corresponding to the current cumulative_max_loss
            temp_intermediate_data['best_row_number'] = temp_intermediate_data.apply(
                lambda row:
                temp_intermediate_data.loc[:row.name].query('loss == @row.cumulative_max_loss and c1 == 1').index[-1]
                if row.cumulative_max_loss > 0 else None,
                axis=1
            ).fillna(0).astype(int)
            temp_intermediate_data['best_row_number'] = temp_intermediate_data['best_row_number'] + 1

            # Concatenate the processed macrorep data to the final DataFrame
            iteration_history = pd.concat([iteration_history, temp_intermediate_data], ignore_index=True)

        # Save the final aggregated results to CSV
        output_file = os.path.join(f'{project_path}/output/data/adhesive_bonding/',
                                   f"bonding_model_material{material}_intermediate_results.csv")
        iteration_history.to_csv(output_file, index=False)

################################################

# material_choices = ['ABS', 'PPS', 'GFRE']
# scenario_choices = [
#     ['1', '2', '3'],
#     ['1', '2', '3'],
#     ['1', '2', '3', '4', '5', '6']]

material_choices = ['PPS']
scenario_choices = [[ '1', '2', '3' ]]

for material_idx in range(len(material_choices)):

    selected_material = material_choices[material_idx]

    cs_bonding_model = ConfigurationSpace(seed=1234)

    # specification of the decision variables
    scenario = CSH.CategoricalHyperparameter("scenario", scenario_choices[material_idx])

    plasma_distance = CSH.UniformFloatHyperparameter("plasma_distance", lower=4, upper=20)
    plasma_passes = CSH.UniformIntegerHyperparameter("plasma_passes", lower=1, upper=50)
    plasma_power = CSH.UniformFloatHyperparameter("plasma_power", lower=300, upper=500)
    plasma_speed = CSH.UniformFloatHyperparameter("plasma_speed", lower=5, upper=250)

    pretreatment = CSH.CategoricalHyperparameter("pretreatment", choices = ['dry_tissue', 'compressed_air', 'US_bath', 'degreasing'])
    cs_bonding_model.add_hyperparameters([scenario, plasma_distance, plasma_passes, plasma_power, plasma_speed, pretreatment])

    # the specification of conditions

    pretreatment_cond = InCondition(pretreatment, scenario, list(set(['2', '3', '4', '5', '6']) & set(scenario_choices[material_idx])))

    # plasma parameters allowed only for scenarios 3 and 6, which are forbidden for aluminum
    if (selected_material !=  'Aluminum'):

        plasma_distance_cond = InCondition(plasma_distance, scenario, list(set(['3', '6']) & set(scenario_choices[material_idx])))
        plasma_passes_cond = InCondition(plasma_passes, scenario, list(set(['3', '6']) & set(scenario_choices[material_idx])))
        plasma_power_cond = InCondition(plasma_power, scenario, list(set(['3', '6']) & set(scenario_choices[material_idx])))
        plasma_speed_cond = InCondition(plasma_speed, scenario, list(set(['3', '6']) & set(scenario_choices[material_idx])))

        cs_bonding_model.add_conditions([plasma_distance_cond,
                                         plasma_passes_cond,
                                         plasma_power_cond,
                                         plasma_speed_cond,
                                         pretreatment_cond])
    else:
        cs_bonding_model.add_conditions([pretreatment_cond])

    dim_bonding_model = len(cs_bonding_model)

    wrapper_obj_fun_instance = wrapper_obj_fun_fac(selected_material)

    run_bonding_model(obj_fun = wrapper_obj_fun_instance,
                      obj_fun_name = 'bonding_model',
                      cs = cs_bonding_model,
                      material = selected_material,
                      seed_list = seed_list,
                      n_init = 10 * dim_bonding_model,
                      n_iterations = 200,
                      n_macroreps = n_macroreps,
                      project_path = project_path,
                      suffix = f'material{selected_material}')

####### transform results into one CSV file per material type
#
# # Define materials and macrorep range
# macrorep_range = range(1, 31)
#
# materials = material_choices
# # Output directory for CSV files
# output_dir = './output/data/adhesive_bonding'
# os.makedirs(output_dir, exist_ok=True)
#
# # Process files for each material
# for material in materials:
#     all_data = []  # To collect data for each material
#
#     # Read each JSON file for the material
#     for macrorep in macrorep_range:
#         filename = f"./results/ctpe_bondingmodel_material{material}_macrorep{macrorep}.json"
#         if os.path.exists(filename):
#             with open(filename, 'r') as file:
#                 data = json.load(file)
#                 # Add macrorep and iteration columns, process negative loss
#                 for iteration, (loss, c1) in enumerate(zip(data['loss'], data['c1']), start=1):
#                     all_data.append({
#                         'macrorep': macrorep,
#                         'iteration': iteration,
#                         'curr_loss': -loss,  # Convert to positive
#                         'c1': c1
#                     })
#
#     # Convert to DataFrame
#     df = pd.DataFrame(all_data)
#
#     # Ensure DataFrame is not empty before processing further
#     if not df.empty:
#         # Calculate maximum loss up to the current iteration
#         df['best_loss'] = df.groupby('macrorep')['curr_loss'].cummax()
#
#         # Derive iter_max_objective (iteration with the maximum loss for each macrorep)
#         max_loss_iter = (
#             df.loc[df.groupby('macrorep')['curr_loss'].idxmax()]
#             [['macrorep', 'iteration']]
#             .rename(columns={'iteration': 'iter_max_objective'})
#         )
#         df = df.merge(max_loss_iter, on='macrorep', how='left')
#
#         # Save to CSV
#         output_file = os.path.join(output_dir, f"bonding_model_material{material}_intermediate_results.csv")
#         df.to_csv(output_file, index=False)
#
#     else:
#         print(f"No data for material {material}. Skipping...")

###########################
# explore outcomes' space by evaluating the bonding model on a large number of input points
###########################

# keep these parameters fixed
# glue_type = 'Araldite'
# sample_size = 25
# time_between_plasma_glue = 1
# curing_time = 30
# curing_temperature = 180
# batch_size = 30
# number_repetitions = 5
# Width_plasma = 2
# general_noise = 0
# noise_factor_plasma = 0
# noise_curing = 0.005
# noise_material = 0
# wt_particles = 30  # took the mid-point of the provided range
# curing_method = 'oven'
# ind_current_bonding = 17.5  # took the mid-point of the provided range
# ind_current_debonding = 0
# ind_time_debonding = 0
#
# n_evaluations = 20000
#
# material_choices = ['ABS', 'PPS', 'GFRE']
# scenario_choices = [
#     ['1', '2', '3'],
#     ['1', '2', '3'],
#     ['1', '2', '3', '4', '5', '6']]
#
# # Start the timer
# #start_time = time.time()
#
# for material_idx in range(len(material_choices)):
#
#     selected_material = material_choices[material_idx]
#     results = []
#     saved_x = []
#
#     selected_material = material_choices[material_idx]
#
#     cs_bonding_model = ConfigurationSpace(seed=1234)
#
#     # specification of the decision variables
#     scenario = CSH.CategoricalHyperparameter("scenario", scenario_choices[material_idx])
#
#     plasma_distance = CSH.UniformFloatHyperparameter("plasma_distance", lower=4, upper=20)
#     plasma_passes = CSH.UniformIntegerHyperparameter("plasma_passes", lower=1, upper=50)
#     plasma_power = CSH.UniformFloatHyperparameter("plasma_power", lower=300, upper=500)
#     plasma_speed = CSH.UniformFloatHyperparameter("plasma_speed", lower=5, upper=250)
#
#     pretreatment = CSH.CategoricalHyperparameter("pretreatment",
#                                                  choices=['dry_tissue', 'compressed_air', 'US_bath', 'degreasing'])
#     cs_bonding_model.add_hyperparameters(
#         [scenario, plasma_distance, plasma_passes, plasma_power, plasma_speed, pretreatment])
#
#     # the specification of conditions
#
#     pretreatment_cond = InCondition(pretreatment, scenario,
#                                     list(set(['2', '3', '4', '5', '6']) & set(scenario_choices[material_idx])))
#
#     # plasma parameters allowed only for scenarios 3 and 6, which are forbidden for aluminum
#     if (selected_material != 'Aluminum'):
#
#         plasma_distance_cond = InCondition(plasma_distance, scenario,
#                                            list(set(['3', '6']) & set(scenario_choices[material_idx])))
#         plasma_passes_cond = InCondition(plasma_passes, scenario,
#                                          list(set(['3', '6']) & set(scenario_choices[material_idx])))
#         plasma_power_cond = InCondition(plasma_power, scenario,
#                                         list(set(['3', '6']) & set(scenario_choices[material_idx])))
#         plasma_speed_cond = InCondition(plasma_speed, scenario,
#                                         list(set(['3', '6']) & set(scenario_choices[material_idx])))
#
#         cs_bonding_model.add_conditions([plasma_distance_cond,
#                                          plasma_passes_cond,
#                                          plasma_power_cond,
#                                          plasma_speed_cond,
#                                          pretreatment_cond])
#     else:
#         cs_bonding_model.add_conditions([pretreatment_cond])
#
#     for r in range(n_evaluations):
#
#         eval_config = cs_bonding_model.sample_configuration()
#         tmp_dict = get_misc_vars(eval_config)
#         scenario = int(eval_config['scenario'])
#         pretreatment = tmp_dict.get('pretreatment')
#         dry_tissue = tmp_dict.get('dry_tissue')
#         compressed_air = tmp_dict.get('compressed_air')
#         US_bath = tmp_dict.get('US_bath')
#         degreasing = tmp_dict.get('degreasing')
#         plasma = tmp_dict.get('plasma')
#         plasma_power_value = tmp_dict.get('plasma_power_value')
#         plasma_speed_value = tmp_dict.get('plasma_speed_value')
#         plasma_distance_value = tmp_dict.get('plasma_distance_value')
#         plasma_passes_value = tmp_dict.get('plasma_passes_value')
#         roughening = tmp_dict.get('roughening')
#         posttreatment = tmp_dict.get('posttreatment')
#
#         saved_x.append(eval_config.get_dictionary() | tmp_dict)
#
#         tensileStrength, failureMode, VisualQ, cost, Feasibility, FinalcontactAngle = eng.bondingModel2(
#             eng.double(scenario), eng.double(pretreatment),
#             eng.double(posttreatment),
#             selected_material,
#             eng.double(dry_tissue),
#             eng.double(compressed_air),
#             eng.double(US_bath),
#             eng.double(degreasing),
#             eng.double(roughening),
#             glue_type,
#             eng.double(sample_size),
#             eng.double(plasma),
#             eng.double(plasma_power_value),
#             eng.double(plasma_speed_value),
#             eng.double(plasma_distance_value),
#             eng.double(plasma_passes_value),
#             eng.double(time_between_plasma_glue),
#             eng.double(curing_time),
#             eng.double(curing_temperature),
#             eng.double(batch_size),
#             eng.double(number_repetitions),
#             eng.double(Width_plasma),
#             eng.double(general_noise),
#             eng.double(noise_factor_plasma),
#             eng.double(noise_curing),
#             eng.double(noise_material),
#             eng.double(wt_particles),
#             curing_method,
#             eng.double(ind_current_bonding),
#             eng.double(ind_current_debonding),
#             eng.double(ind_time_debonding),
#             nargout=6)
#
#         results.append([tensileStrength, failureMode, VisualQ, cost, Feasibility, FinalcontactAngle])
#
#     results_df=pd.DataFrame(results, columns = ['tensileStrength', 'failureMode', 'VisualQ', 'cost', 'Feasibility', 'FinalcontactAngle'], )
#
#     saved_x_df = pd.DataFrame(saved_x)
#     saved_x_df = saved_x_df[sorted(saved_x_df.columns)]
#
#     results_both = pd.concat([saved_x_df, results_df], axis = 1)
#     results_both.to_excel(f'{project_path}/output/data/adhesive_bonding/bonding_model_manyeval_{selected_material}.xlsx', index=False)
#
# end_time = time.time()
# execution_time = end_time - start_time
#
# # Close the MATLAB engine
# eng.quit()