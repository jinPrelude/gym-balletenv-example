
from gym_balletenv.envs import BalletEnvironment
from gym_balletenv.wrappers import GrayScaleObservation, TransposeObservation

import time

import gymnasium as gym
import numpy as np

def make_env(env_id, max_steps, idx, capture_video, run_name):
    def thunk():
        env = BalletEnvironment(env_id, max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = GrayScaleObservation(env)
        env = TransposeObservation(env)
        return env

    return thunk

SEED = 0
NUM_DANCERS = 2
DANCE_DELAY = 2
MAX_STEPS = 200
NUM_ENVS = 64

if __name__=="__main__":
    envs = gym.vector.SyncVectorEnv(
            [make_env("2_delay2", MAX_STEPS, i, True, "test") for i in range(NUM_ENVS)]
        ) # AsyncVectorEnv also available.
    start = time.time()
    obs, infos = envs.reset(seed=SEED)
    for i in range(1000):
        action = envs.action_space.sample()
        obs, reward, done, _, info = envs.step(action)
    end = time.time()
    print(f"Time for syncvectorenv({NUM_ENVS} envs): {end - start}")