from copy import deepcopy
import numpy as np


class ExperienceBuffer:
    """
    Basic class to store transitions from a given environment
    """

    def __init__(self, max_memory_size, env_id_list: list, shape_dict: dict):
        self.max_memory_size = max_memory_size
        self.num_environments = len(env_id_list)
        self.statistics = shape_dict.keys()
        self.env_id_list = env_id_list

        step_dict = {}
        for key in self.statistics:
            step_dict[key] = np.zeros(shape=(max_memory_size, shape_dict[key][0]),
                                      dtype=shape_dict[key][1])
        self._replay_buffer = {env_id: deepcopy(step_dict) for env_id in env_id_list}
        self._top = {env_id: 0 for env_id in env_id_list}  # Trailing index with latest entry in env
        self._size = {env_id: 0 for env_id in env_id_list}  # Trailing index with num sam

    def register_trajectory(self, trajectory: dict, day: int):
        print(trajectory)
        for step_values in zip(*trajectory.values()):
            print(step_values)
            step = dict(zip(trajectory.keys(), step_values))
            step['day'] = day
            self.register_step(step, day)

    def register_step(self, step: dict, day: int):
        env_container = self._replay_buffer[day]
        for key in self.statistics:
            env_container[key][self._top[day]] = step[key]
        self._advance(day)

    def random_batch(self, batch_size):
        assert batch_size <= self.num_steps_can_sample, "Not enough samples in buffer"
        batch = {}
        for env_id in self.env_id_list:
            indices = np.random.randint(0, self._size[env_id], batch_size)
            env_container = self._replay_buffer[env_id]
            state_map = {}
            for key in self.statistics:
                state_map[key] = env_container[key][indices]
            batch[env_id] = state_map
        return batch

    def _advance(self, env_id):
        self._top[env_id] = (self._top[env_id] + 1) % self.max_memory_size
        if self._size[env_id] < self.max_memory_size:
            self._size[env_id] += 1

    @property
    def num_steps_can_sample(self):
        return min(self._size.values())
