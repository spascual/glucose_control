from typing import Union

import numpy as np
import pandas as pd

from reinforcement_learning.environment import Environment
from reinforcement_learning.experience_buffer import ExperienceBuffer
from reinforcement_learning.policy import DiscreteRandomPolicy, ContinuousRandomPolicy, \
    TransformDiscreteActions, TransformContinuousActions

df_patients = pd.read_csv('subject_profiles.csv')


class Agent:
    def __init__(self,
                 exploratory_policy: Union[DiscreteRandomPolicy, ContinuousRandomPolicy],
                 learnt_policy,
                 global_buffer: ExperienceBuffer,
                 fictional_buffer: ExperienceBuffer
                 ):
        self.exploratory_policy = exploratory_policy
        self.learnt_policy = learnt_policy
        self.global_buffer = global_buffer
        self.fictional_buffer = fictional_buffer
        self.select_transform()

    def sample_episode(self, env: Environment):
        """
        Collect episode experience with exploratory policy and store in global buffer:
        1. Sample raw actions from (random) policy
        2. Transform raw actions to action space, producing BM = [bolus_time, bolus_dose]
        3. Simulate day to Obtain glucose levels and score
        4. Compute state, next_state, state_extra, reward sequences from patient_data & simulation
        5. Store trajectory in (real experience) global buffer
        """
        num_steps = env.num_meals + 1
        raw_actions = self.exploratory_policy.act_episode(num_steps)
        BM = self.transform_to_BM(raw_actions, env.time_stamps, env.dose_range)
        states, next_states, state_extra, rewards, done, _, _ = env.simulate_day(BM)
        traj = dict(
            states=states.reshape(-1, 1),
            next_states=next_states.reshape(-1, 1),
            raw_actions=raw_actions.reshape(-1, 2),
            BM=BM.reshape(-1, 2),
            rewards=rewards.reshape(-1, 1),
            interval_size=state_extra['interval_size'].reshape(-1, 1),
            carbs=state_extra['carbs'].reshape(-1, 1),
            termination_masks=done.reshape(-1, 1)
        )
        self.global_buffer.register_trajectory(traj, env.day)
        return traj

    def simulate_episode(self, initial_state, state_extra, model):
        traj = []
        state = initial_state
        for extra in zip(*state_extra.values()):
            raw_action = self.learnt_policy.sample(state)
            next_state_f, reward_f = model.infer(state, raw_action, extra)
            done = np.ones((1, 1), dtype='uint8') if len(traj) == len(state_extra['carbs']) - 1 \
                else np.zeros((1, 1), dtype='uint8')
            step = dict(
                states=state.reshape(1, -1),
                next_states=next_state_f.reshape(1, -1),
                raw_actions=raw_action.reshape(1, -1),
                rewards=reward_f.reshape(1, -1),
                interval_size=extra['interval_size'].reshape(1, -1),
                carbs=extra['carbs'].reshape(1, -1),
                termination_masks=done.reshape(1, -1)
            )
            self.fictional_buffer.register_step(step)
            traj.append(step)
            state = next_state_f
        return traj

    def transform_to_BM(self, raw_actions, time_stamps, dose_range):
        return self.transform_action.forward(raw_actions, time_stamps, dose_range)

    def select_transform(self):
        if isinstance(self.exploratory_policy, DiscreteRandomPolicy):
            self.transform_action = TransformDiscreteActions()
        elif isinstance(self.exploratory_policy, ContinuousRandomPolicy):
            self.transform_action = TransformContinuousActions()
        else:
            raise NotImplementedError
