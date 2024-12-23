import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, InCondition
import numpy as np
import pandas as pd
from util.utils import get_logger
from optimizer import TPEOptimizer
import json
import matlab.engine
import os
import sys

# start MATLAB engine
eng = matlab.engine.start_matlab()
eng.addpath('./adhesive_bonding_simulator/')

def create_config_space(material):
    """
    Specifies a configuration space appropriate for a given material.

    Args:
        material: char, one of 'ABS', 'PPS', or 'GFRE'.

    Returns:
        cs_bonding_model: a correctly created object of ConfigSpace type.
    """
    cs_bonding_model = ConfigurationSpace(seed=1234)

    material_scenario_dict = {
        'ABS': ['1', '2', '3'],
        'PPS': ['1', '2', '3'],
        'GFRE': ['1', '2', '3', '4', '5', '6']}
    selected_scenarios = material_scenario_dict.get(material)

    # specification of the decision variables
    scenario = CSH.CategoricalHyperparameter("scenario", selected_scenarios)

    plasma_distance = CSH.UniformFloatHyperparameter("plasma_distance", lower=4, upper=20)
    plasma_passes = CSH.UniformIntegerHyperparameter("plasma_passes", lower=1, upper=50)
    plasma_power = CSH.UniformFloatHyperparameter("plasma_power", lower=300, upper=500)
    plasma_speed = CSH.UniformFloatHyperparameter("plasma_speed", lower=5, upper=250)

    pretreatment = CSH.CategoricalHyperparameter("pretreatment",
                                                 choices=['dry_tissue', 'compressed_air', 'US_bath', 'degreasing'])
    cs_bonding_model.add_hyperparameters(
        [scenario, plasma_distance, plasma_passes, plasma_power, plasma_speed, pretreatment])

    # the specification of conditions
    pretreatment_cond = InCondition(pretreatment, scenario,
                                    list(set(['2', '3', '4', '5', '6']) & set(selected_scenarios)))

    # plasma parameters allowed only for scenarios 3 and 6, which are forbidden for aluminum
    if material != 'Aluminum':

        plasma_distance_cond = InCondition(plasma_distance, scenario,
                                           list(set(['3', '6']) & set(selected_scenarios)))
        plasma_passes_cond = InCondition(plasma_passes, scenario,
                                         list(set(['3', '6']) & set(selected_scenarios)))
        plasma_power_cond = InCondition(plasma_power, scenario,
                                        list(set(['3', '6']) & set(selected_scenarios)))
        plasma_speed_cond = InCondition(plasma_speed, scenario,
                                        list(set(['3', '6']) & set(selected_scenarios)))

        cs_bonding_model.add_conditions([plasma_distance_cond,
                                         plasma_passes_cond,
                                         plasma_power_cond,
                                         plasma_speed_cond,
                                         pretreatment_cond])
    else:
        cs_bonding_model.add_conditions([pretreatment_cond])

    return cs_bonding_model

def get_all_input_vars(eval_config):
    """
    Create a full set of input variables required by the adhesive bonding simulator (matlab function "bondingModel2").
    An incomplete set of variables can occur due to the conditionality of some input parameters, e.g., plasma_power,
    plasma_distance are conditional by plasma=1.

    Args:
        eval_config: dictionary with variable_name-value pairs that is used as input for cTPE optimisation.

    Returns:
        var_dict: dictionary with the fixed set of inputs.
    """
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
        plasma_power = float(eval_config['plasma_power'])
        plasma_speed = float(eval_config['plasma_speed'])
        plasma_distance = float(eval_config['plasma_distance'])
        plasma_passes = int(eval_config['plasma_passes'])
    else:
        plasma = 0
        # assigning arbitrary value below (e.g. the lower bound) as the value won't actually be used in outcome calculation
        plasma_power = 300.0
        plasma_speed = 5.0
        plasma_distance = 4.0
        plasma_passes = 1

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
                'plasma_power': plasma_power,
                'plasma_speed': plasma_speed,
                'plasma_distance': plasma_distance,
                'plasma_passes': plasma_passes,
                'roughening': roughening,
                'posttreatment': posttreatment}
    return var_dict

def wrapper_obj_fun_fac(input_material):
    def wrapper_obj_fun(eval_config):
        """
        Function factory (the '_fac' part) is a workaround for not knowing how to pass multiple arguments (i.e., eval_config and material) to the objective function.
        'material' variable is required by the bonding simulator, but it's not an optimisation parameter and can't be found in the eval_config argument.

        Args:
            eval_config: dictionary.

        Returns:
            A dictionary of the objective function (loss function, which is tensileStrength), constraints, and additional non-optimised outcomes.
        """
        material = input_material

        tmp_dict = get_all_input_vars(eval_config)

        pretreatment = tmp_dict.get('pretreatment')
        dry_tissue = tmp_dict.get('dry_tissue')
        compressed_air = tmp_dict.get('compressed_air')
        US_bath = tmp_dict.get('US_bath')
        degreasing = tmp_dict.get('degreasing')
        plasma = tmp_dict.get('plasma')
        plasma_power = tmp_dict.get('plasma_power')
        plasma_speed = tmp_dict.get('plasma_speed')
        plasma_distance = tmp_dict.get('plasma_distance')
        plasma_passes = tmp_dict.get('plasma_passes')
        roughening = tmp_dict.get('roughening')
        posttreatment = tmp_dict.get('posttreatment')

        # to limit the number of optimisation parameters (and make the optimisation problem easier), it was decided to fix the parameters below
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

        tensileStrength, failureMode, VisualQ, cost, Feasibility, FinalcontactAngle = eng.bondingModel2(
                eng.double(scenario), eng.double(pretreatment),
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
                eng.double(plasma_power),
                eng.double(plasma_speed),
                eng.double(plasma_distance),
                eng.double(plasma_passes),
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

        # cTPE minimises the objective function, while in our case we want to maximise the tensileStrength, hence the minus sign
        # c1 is the constraint, in our case Visual Quality which takes values 1 (quality is OK) or 2 (quality NOT OK)
        # failureMode, cost, Feasibility, FinalcontactAngle are additional non-optimised outcomes that also are returned
        return dict(loss=-tensileStrength, c1=VisualQ, failureMode=failureMode, cost=cost,
                    Feasibility=Feasibility, FinalcontactAngle=FinalcontactAngle)

    return wrapper_obj_fun

def run_optimisation(material, obj_fun, config_space, seed_list, n_init, n_iterations, n_macroreps, save_iteration_history=False):
    """
    Args:
        material: one of: 'ABS', 'PPS', 'GFRE'.
        obj_fun: python wrapper which calls the bonding model developed in matlab.
        config_space: the configuration of input variables that is used for cTPE-based sampling and optimisation.
        seed_list: a list with integers corresponding to seed to make each macro replication reproducible.
        n_init = int, number of initial points for cTPE.
        n_iterations = int, max number of internal optimisation evaluations.
        n_macroreps = int, number of independent (so initialised with different starting points) replications of the optimisation.
        save_iteration_history: boolean, whether to create a CSV with the values of input variables, objective function,
        and constraints across all internal optimisation iterations.

    Returns:
        NULL; the function's side effect is a CSV file with the optimal solutions. The file can be found at
        ./results/adhesive_bonding_optimal_parameters_materialXYZ.csv, where XYZ is one of the three material names. This file has one row per macroreplicate.
        If save_iteration_history=True, then ./results/adhesive_bonding_materialXYZ_intermediate_results.csv is created. This file contains a full history of results (multiple rows per macroreplicate)
        ./results/adhesive_bonding_optimal_parameters_materialXYZ.csv
    """
    # create objects for storing results
    outcomes = []
    saved_x = []

    for r in range(n_macroreps):
        nm = f'adhesive_bonding_material{material}_macrorep{r + 1}'
        logger = get_logger(file_name=nm, logger_name=nm)

        opt = TPEOptimizer(obj_func=obj_fun,
                           config_space=config_space,
                           constraints={'c1': 1},
                           seed=int(seed_list[r]),
                           n_init=n_init,
                           max_evals=n_iterations,
                           resultfile=nm)

        best_config, best_loss, additional_nonoptimised_outcomes, final_constraints = opt.optimize(logger)

        # extract optimal x's
        temp = best_config
        tmp_dict = get_all_input_vars(temp)
        saved_x.append(temp | tmp_dict)

        outcomes.append(
            [-best_loss] + list(additional_nonoptimised_outcomes.values()) + list(final_constraints.values()))

    saved_x_df = pd.DataFrame(saved_x)
    saved_x_df = saved_x_df[sorted(saved_x_df.columns)]

    outcomes_df = pd.DataFrame(outcomes, columns=['loss'] + list(additional_nonoptimised_outcomes.keys()) + list(
        final_constraints.keys()))
    outcomes_df['seed'] = seed_list

    results_both = pd.concat([saved_x_df, outcomes_df], axis=1)
    results_both.to_csv(f'./results/adhesive_bonding_optimal_parameters_material{material}.csv', index=False)

    if save_iteration_history:
        # Save intermediate optimization results (all macroreps in one file) into CSV
        iteration_history = pd.DataFrame()

        for r in range(1, n_macroreps + 1):
            nm = f'adhesive_bonding_material{material}_macrorep{r}'

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
        output_file = os.path.join(f'./results/adhesive_bonding_material{material}_intermediate_results.csv')
        iteration_history.to_csv(output_file, index=False)