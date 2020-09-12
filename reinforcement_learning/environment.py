from typing import List

import numpy as np
from simulator_wrapper import run_simulator_day

from patient_info import PatientInfo
from utils import extract_patient_meal_times, extract_patient_meal_carbs, healthy_range, \
    hypoglucosis


class Environment(PatientInfo):
    """
    Mimicking a Reinforcement Learning Environment to represent a Glucose Control System for a given
    patient specified in `subject_profiles.csv`.

    Mayor assumption / simplification  here is that an episode consists of
    (num_meals + 1)-time steps:
    -> Agents take (num_meals + 1) actions
    -> Each insulin bolus (action) is located in an interval between two consecutive meals
    -> Where in that interval and the insulin dose is decided by the agent
    -> States are defined to be the glucose readings 2hrs after a meal ***
    -> States are hidden to the agents until the completion of the episode
    -> Agents act based on a prediction on the current state and extra information about the meals
        provided by the environment
    -> Rewards consists of a breakdown of the simulation score:
        r_t is the contribution to the score between states s_t and s_t+1

    In contrast, to having a 1440-step time horizon and taking actions each minute on the day.

    """

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
        self.time_states = self.compute_time_states()
        self.state_extra = self.compute_state_extra()

    def simulate_day(self, BM):
        time, glucose, score = run_simulator_day(self.day, BM)
        glucose = np.array(glucose, dtype=np.float32)
        states = glucose[self.time_states[:-1]]
        next_states = glucose[self.time_states[1:]]
        rewards = self.get_rewards_from_simulation(glucose)
        done = np.hstack([np.zeros((self.num_steps - 1,), dtype='uint8'),
                          np.ones((1,), dtype='uint8')])
        return states, next_states, self.state_extra, rewards, done, glucose, score

    def get_rewards_from_simulation(self, glucose):
        healthy = healthy_range(glucose)
        hypo = hypoglucosis(glucose)

        def rewards_interval(start, end):
            healthy_bonus = len(glucose[start: end][healthy[start: end]])
            hypo_penalty = len(glucose[start: end][hypo[start: end]])
            return healthy_bonus - hypo_penalty

        rewards = np.array([
            rewards_interval(self.time_states[i], self.time_states[i + 1])
            for i in range(self.num_steps)
        ], dtype=np.float32)
        return rewards

    def compute_state_extra(self):
        return dict(
            interval_size=self.time_stamps[1:] - self.time_stamps[:-1],
            carbs=np.hstack([self.carb_meals, 0.])
        )

    def compute_time_states(self):
        """
        Time of the day (in min) glucose levels are recorded in the buffer as part of the state:
        - First item is the time of initial glucose reading at time = 0
        - Last item is the time of final glucose reading of the day at time = 1439
        - The intermidate items are meal times plus a 1hr window
        """
        time_intermediate_states = np.minimum(
            np.array(self.time_meals, dtype=np.int32) + self.window_states, 1438
        )
        return np.hstack([self.time_stamps[0], time_intermediate_states, self.time_stamps[-1]])


def dumb_policy(env: Environment) -> List:
    # avg_dose_per_meal
    # single_dose = (.5 * env.min_dose + .5 * env.max_dose) / env.num_meals
    single_dose = (.0 * env.min_dose + 1. * env.max_dose) / env.num_meals
    actions = len(env.time_meals) * [single_dose]
    return actions
