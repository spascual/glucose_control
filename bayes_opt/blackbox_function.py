import numpy as np
import pandas as pd
import torch

from patient_info import PatientInfo
from simulator_wrapper import run_simulator_day

from utils import dtype

torch.manual_seed(29)

PATIENTS = pd.read_csv('subject_profiles.csv')


class BlackBoxObjective:
    """
    Equivalent of the environment, but now accounting for multiple days with same amount of meals
    """

    def __init__(self, num_meals: int):
        self.num_meals = num_meals
        self.input_dim = num_meals + 1

    def f(self, X, shift=0, include_BMs=False):
        batch_size = X.shape[0]
        assert self.input_dim == X.shape[-1]
        env = PatientInfo(patient_data=PATIENTS.iloc[shift])
        time_bolus = self.time_insulin_heuristic(env)
        min_dose, max_dose = self.dose_bolus_range(env)
        scores = [
            self.simulate_day(day=shift,
                              BM=self.transform_to_BM(raw_dose_bolus=X[m, :],
                                                      time_bolus=time_bolus,
                                                      min_dose=min_dose,
                                                      max_dose=max_dose)
                              )
            for m in range(batch_size)
        ]
        f_X = torch.tensor(scores, dtype=dtype).view(-1, 1)
        if include_BMs:
            BMs = [
                self.transform_to_BM(raw_dose_bolus=X[m, :],
                                     time_bolus=time_bolus,
                                     min_dose=min_dose,
                                     max_dose=max_dose)
                for m in range(batch_size)
            ]
            return f_X, BMs
        return f_X

    def simulate_day(self, day, BM):
        _, glucose, score = run_simulator_day(day, BM)
        glucose = np.array(glucose, dtype=np.float32)
        return score - (glucose > 180.).sum() * (50. / 1440.)

    def transform_to_BM(self, raw_dose_bolus, time_bolus, min_dose, max_dose):
        dose_bolus = (
                max_dose * raw_dose_bolus
                + (1 - raw_dose_bolus) * min_dose
        )
        BM = torch.cat([time_bolus, dose_bolus], dim=0)
        return BM.numpy()

    def dose_bolus_range(self, env: PatientInfo):
        min_dose_day, max_dose_day = env.min_dose, env.max_dose
        torch_carb_meals = torch.tensor(env.carb_meals)
        norm_carb_meals = torch_carb_meals / torch_carb_meals.sum()
        weights_bolus = torch.zeros((1, self.input_dim))
        if env.initial_state > 180:
            # Bolus around breakfast
            weights_bolus[0, 0] = .5 * norm_carb_meals[0]
            weights_bolus[0, 1] = .5 * norm_carb_meals[0]
            # Before & After
            weights_bolus[0, 2:] = norm_carb_meals[2:]
        else:
            max_carb_idx = torch.argmax(norm_carb_meals, dim=0)
            # Bolus around main meal
            weights_bolus[0, max_carb_idx] = .5 * norm_carb_meals[max_carb_idx]
            weights_bolus[0, max_carb_idx + 1] = .5 * norm_carb_meals[max_carb_idx]
            # Before & After
            weights_bolus[0, :max_carb_idx] = norm_carb_meals[:max_carb_idx]
            weights_bolus[0, max_carb_idx + 2:] = norm_carb_meals[max_carb_idx + 1:]
        min_dose = 0. * weights_bolus
        max_dose = max_dose_day * weights_bolus
        return min_dose, max_dose

    def time_insulin_heuristic(self, env: PatientInfo):
        time_meals = torch.tensor(env.time_meals)
        carb_meals = torch.tensor(env.carb_meals)
        time_bolus = torch.zeros((1, self.input_dim))
        if env.initial_state > 180:
            # First two bolus around breakfast
            time_bolus[0, 0] = time_meals[0] - 60
            time_bolus[0, 1:] = torch.min(time_meals + 30, 1439 * torch.ones_like(time_meals))
        else:
            max_carb_idx = torch.argmax(carb_meals, dim=0)
            time_bolus[0, max_carb_idx] = time_meals[max_carb_idx] - 119
            time_bolus[0, :max_carb_idx] = time_meals[:max_carb_idx] + 20
            time_bolus[0, max_carb_idx + 1:] = torch.min(
                time_meals[max_carb_idx:] + 30, 1439. * torch.ones_like(time_meals[max_carb_idx:])
            )
        return time_bolus
