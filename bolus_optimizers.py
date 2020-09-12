import pandas as pd

from bayes_opt.blackbox_function import BlackBoxObjective
from bayes_opt.run_RGPE import run_RGPE
from bayes_opt.train_base_models import load_base_models, fit_base_models
from utils import compute_bounds, select_training_tasks

PATIENTS = pd.read_csv('subject_profiles.csv')


def bolus_optimizer(day: int):
    patient = PATIENTS.iloc[day]
    num_meals = int(patient['daily_meal_count'])
    bounds = compute_bounds(num_meals)
    training_tasks = select_training_tasks(num_meals, PATIENTS, num_tasks=5)
    objective = BlackBoxObjective(num_meals=num_meals).f

    base_model_list = load_base_models(objective, bounds, training_tasks, day)

    BM = run_RGPE(test_task=day,
                  objective=objective,
                  bounds=bounds,
                  base_model_list=base_model_list
                  )
    return day, BM


def bolus_optimizer_training_set(day: int):
    patient = PATIENTS.iloc[day]
    num_meals = int(patient['daily_meal_count'])
    bounds = compute_bounds(num_meals)
    training_tasks = select_training_tasks(num_meals, PATIENTS, num_tasks=5)
    objective = BlackBoxObjective(num_meals=num_meals).f
    base_model_list = fit_base_models(objective,
                                      bounds,
                                      training_tasks,
                                      training_size=20,
                                      save_models=True
                                      )
    BM = run_RGPE(test_task=day,
                  objective=objective,
                  bounds=bounds,
                  base_model_list=base_model_list
                  )
    return day, BM

print(bolus_optimizer(2))