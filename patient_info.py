from typing import List

import numpy as np

from utils import extract_patient_meal_times, extract_patient_meal_carbs


class PatientInfo:
    def __init__(self, patient_data):
        # Extract patient (visible) stats that define MDP
        self.day = patient_data['day_id']
        self.time_meals = extract_patient_meal_times(patient_data)
        self.carb_meals = extract_patient_meal_carbs(patient_data)
        self.initial_state = patient_data['start_sg']
        self.num_meals = int(patient_data['daily_meal_count'])
        self.min_dose = patient_data['min_daily_bolus_allowed']
        self.max_dose = patient_data['max_daily_bolus_allowed']
        # Give 1hr between last meal and glucose reading to feed as state
        self.window_states = 60
        self.num_steps = self.num_meals + 1
        self.dose_range = np.array([self.min_dose, self.max_dose])
        self.time_stamps = np.array([0] + self.time_meals + [1439], dtype=np.int32)
