
from gym_balletenv.envs import BalletEnvironment
from gym_balletenv.wrappers import GrayScaleObservation, RecordVideo, TransposeObservation
from sb3_vec_env.subproc_vec_env import SubprocVecEnv

import time

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
        env = TransposeObservation(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

SEED = 0
NUM_DANCERS = 2
DANCE_DELAY = 2
MAX_STEPS = 200
NUM_ENVS = 16

if __name__=="__main__":
    envs = SubprocVecEnv(
            [make_env("2_delay2", MAX_STEPS, SEED + i, i, True, "test") for i in range(NUM_ENVS)]
        )
    start = time.time()
    obs = envs.reset()
    for i in range(1000):
        action = [envs.action_space.sample() for _ in range(NUM_ENVS)]
        obs, reward, done, info = envs.step(action)
    end = time.time()
    print(f"Time for subproc({NUM_ENVS} envs): {end - start}")