import numpy as np


class ContinuousRandomPolicy:
    """
    Random Policy taking raw actions on a 2D discrete action space with
    - ax : 0 (lower) or 1 (upper) part of the time interval between current glucose state
        and next meal.
    - ay : 0 (no bolus), 1 (low-dose), 2 (medium-dose), 3 (high-dose)

    These raw actions are transformed by the environment (patient) into meaningful continuous
    values using `Environment.transform_to_BM()` method.

    Used as exploratory acting policy
    """

    def __init__(self):
        self.dim_action = 2

    def act(self):
        raw_action = np.random.uniform(0, 1, size=self.dim_action)
        return raw_action

    def act_episode(self, num_steps: int):
        raw_actions = np.random.uniform(0, 1,
                                        size=(self.dim_action, num_steps))
        return raw_actions


class DiscreteRandomPolicy:
    """
    Random Policy taking raw actions on a 2D discrete action space with
    - ax : 0 (lower) or 1 (upper) part of the time interval between current glucose state
        and next meal.
    - ay : 0 (no bolus), 1 (low-dose), 2 (medium-dose), 3 (high-dose)

    These raw actions are transformed by the environment (patient) into meaningful continuous
    values using `Environment.transform_to_BM()` method.

    Used as exploratory acting policy
    """

    def __init__(self):
        self.interval_bins = 2  # ax dim
        self.dose_levels = 4  # ay dim

    def act(self):
        raw_interval = np.random.randint(self.interval_bins)
        raw_dose = np.random.randint(self.dose_levels)
        raw_action = [raw_interval, raw_dose]
        return np.array(raw_action, dtype=np.int32)

    def act_episode(self, num_steps: int):
        raw_intervals = np.random.randint(self.interval_bins, size=num_steps)
        raw_doses = np.random.randint(self.dose_levels, size=num_steps)
        raw_actions = np.vstack([raw_intervals, raw_doses])
        return np.array(raw_actions, dtype=np.int32)


class TransformDiscreteActions:

    def forward(self, raw_actions, time_stamps, dose_range):
        raw_time_actions, raw_value_actions = raw_actions[0, :], raw_actions[1, :]

        time_actions = [
            self.sample_time_action(time_stamps[i], time_stamps[i + 1], raw_time)
            for i, raw_time in enumerate(raw_time_actions)
        ]
        # Compute dose range based on max/min total dose and number of intervals
        num_steps = len(raw_value_actions)
        dose_min, dose_max = dose_range[0] / num_steps, dose_range[1] / num_steps
        dose_levels = np.array([0., dose_min, (dose_min + dose_max) / 2, dose_max])
        value_actions = dose_levels[raw_value_actions]
        return np.vstack([time_actions, value_actions])

    @staticmethod
    def sample_time_action(min_time, max_time, raw_time: bool):
        middle_pt = (min_time + max_time) / 2
        if raw_time:
            return (middle_pt + max_time) / 2
        return (middle_pt + min_time) / 2


class TransformContinuousActions:

    def forward(self, raw_actions, time_stamps, dose_range):
        raw_time_actions, raw_value_actions = raw_actions[0, :], raw_actions[1, :]

        time_actions = [
            self.sample_time_action(time_stamps[i], time_stamps[i + 1], raw_time)
            for i, raw_time in enumerate(raw_time_actions)
        ]
        # Compute dose range based on max/min total dose and number of intervals
        num_steps = len(raw_value_actions)
        dose_min, dose_max = dose_range[0] / num_steps, dose_range[1] / num_steps
        value_actions = dose_max * raw_value_actions + (1 - raw_value_actions) * dose_min
        return np.vstack([time_actions, value_actions])

    @staticmethod
    def sample_time_action(min_time, max_time, raw_time_action):
        """
        Linear interpolation between start and end of the interval:
        -> For a first implementation it would be a middle point between
            current state time and next meal time
        """
        time_action = max_time * raw_time_action + (1 - raw_time_action) * min_time
        return time_action
