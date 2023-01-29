
from gym_balletenv.envs import BalletEnvironment
from gym_balletenv.wrappers import GrayScaleObservation, RecordVideo, TransposeObservation, OnehotLanguage
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
        env = OnehotLanguage(env)
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



envs = make_env("2_delay2", MAX_STEPS, SEED, 0, True, "test")()
for i in range(3):
    obs = envs.reset()
    done = False
    while not done:
        action = envs.action_space.sample()
        obs, reward, done, info = envs.step(action)
    print(info)