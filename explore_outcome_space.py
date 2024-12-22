####################################################################################
# Explore the outcome space by evaluating (meaning NO optimisation) the bonding model
# on a large number of input points
####################################################################################

keep these parameters fixed
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

n_evaluations = 20000

material_choices = ['ABS', 'PPS', 'GFRE']
scenario_choices = [
    ['1', '2', '3'],
    ['1', '2', '3'],
    ['1', '2', '3', '4', '5', '6']]

# Start the timer
#start_time = time.time()

for material_idx in range(len(material_choices)):

    selected_material = material_choices[material_idx]
    results = []
    saved_x = []

    selected_material = material_choices[material_idx]

    cs_bonding_model = ConfigurationSpace(seed=1234)

    # specification of the decision variables
    scenario = CSH.CategoricalHyperparameter("scenario", scenarios)

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
                                    list(set(['2', '3', '4', '5', '6']) & set(scenarios)))

    # plasma parameters allowed only for scenarios 3 and 6, which are forbidden for aluminum
    if (selected_material != 'Aluminum'):

        plasma_distance_cond = InCondition(plasma_distance, scenario,
                                           list(set(['3', '6']) & set(scenarios)))
        plasma_passes_cond = InCondition(plasma_passes, scenario,
                                         list(set(['3', '6']) & set(scenarios)))
        plasma_power_cond = InCondition(plasma_power, scenario,
                                        list(set(['3', '6']) & set(scenarios)))
        plasma_speed_cond = InCondition(plasma_speed, scenario,
                                        list(set(['3', '6']) & set(scenarios)))

        cs_bonding_model.add_conditions([plasma_distance_cond,
                                         plasma_passes_cond,
                                         plasma_power_cond,
                                         plasma_speed_cond,
                                         pretreatment_cond])
    else:
        cs_bonding_model.add_conditions([pretreatment_cond])

    for r in range(n_evaluations):

        eval_config = cs_bonding_model.sample_configuration()
        tmp_dict = get_all_input_vars(eval_config)
        scenario = int(eval_config['scenario'])
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

        saved_x.append(eval_config.get_dictionary() | tmp_dict)

        tensileStrength, failureMode, VisualQ, cost, Feasibility, FinalcontactAngle = eng.bondingModel2(
            eng.double(scenario), eng.double(pretreatment),
            eng.double(posttreatment),
            selected_material,
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

        results.append([tensileStrength, failureMode, VisualQ, cost, Feasibility, FinalcontactAngle])

    results_df=pd.DataFrame(results, columns = ['tensileStrength', 'failureMode', 'VisualQ', 'cost', 'Feasibility', 'FinalcontactAngle'], )

    saved_x_df = pd.DataFrame(saved_x)
    saved_x_df = saved_x_df[sorted(saved_x_df.columns)]

    results_both = pd.concat([saved_x_df, results_df], axis = 1)
    results_both.to_excel(f'{project_path}/output/data/adhesive_bonding/bonding_model_manyeval_{selected_material}.xlsx', index=False)
