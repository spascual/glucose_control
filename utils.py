import numpy as np
import torch

noise_std = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


def extract_patient_meal_times(patient_data):
    meal_times = [patient_data[r'meal_%i_time' % (k)] for k in range(1, 6)
                  if not np.isnan(patient_data[r'meal_%i_time' % (k)])
                  ]
    return meal_times


def extract_patient_meal_carbs(patient_data):
    meal_carbs = [patient_data[r'meal_%i_carb' % (k)] for k in range(1, 6)
                  if not np.isnan(patient_data[r'meal_%i_carb' % (k)])
                  ]
    return meal_carbs


def healthy_range(glucose: np.ndarray):
    return np.logical_and(glucose < 181, glucose > 69)


def hypoglucosis(glucose: np.ndarray):
    return glucose <= 69


def roll_col(X, shift):
    """
    Rotate columns to right by shift.
    """
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


def compute_bounds(num_meals: int):
    input_dim = num_meals + 1
    bounds = torch.cat([torch.zeros((1, input_dim), dtype=dtype),
                        torch.ones((1, input_dim), dtype=dtype)], dim=0)
    return bounds


def select_training_tasks(num_meals: int, patients, num_tasks=10):
    patient_ids = [idx for idx in range(40) if patients.iloc[idx]['daily_meal_count'] == num_meals]
    max_num_task = min(len(patient_ids), num_tasks)
    random_training_tasks = np.random.choice(patient_ids, size=max_num_task, replace=False)
    return random_training_tasks
