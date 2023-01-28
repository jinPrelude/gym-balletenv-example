
from gym_balletenv.envs import BalletEnvironment
from wrappers.gray_scale_observation import GrayScaleObservation
from wrappers.record_video import RecordVideo

import gym
import numpy as np

def make_env(idx, seed, num_dancers, dance_delay, max_steps, capture_video, run_name):
    def thunk():
        env = BalletEnvironment(num_dancers=2, dance_delay=16, max_steps=320)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = RecordVideo(env, f"videos/{run_name}")
        env = GrayScaleObservation(env)
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



envs = make_env(0, SEED, NUM_DANCERS, DANCE_DELAY, MAX_STEPS, True, "test")()
for i in range(3):
    obs = envs.reset()
    done = False
    while not done:
        action = envs.action_space.sample()
        obs, reward, done, info = envs.step(action)
    print(info)