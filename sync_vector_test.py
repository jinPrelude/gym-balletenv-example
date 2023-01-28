
from gym_balletenv.envs import BalletEnvironment
from wrappers.gray_scale_observation import GrayScaleObservation
from wrappers.record_video import RecordVideo
from wrappers.newaxis_observation import NewAxisObservation

import gym
import numpy as np

def make_env(env_id, max_steps, seed, idx, capture_video, run_name):
    def thunk():
        env = BalletEnvironment(env_id, max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = RecordVideo(env, f"videos/{run_name}")
        env = GrayScaleObservation(env)
        env = NewAxisObservation(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

SEED = 0
NUM_DANCERS = 2
DANCE_DELAY = 2
MAX_STEPS = 200
NUM_ENVS = 2


envs = gym.vector.SyncVectorEnv(
        [make_env("2_delay2", MAX_STEPS, SEED + i, i, True, "test") for i in range(NUM_ENVS)]
    )

obs = envs.reset()
for i in range(1000):
    action = envs.action_space.sample()
    obs, reward, done, info = envs.step(action)