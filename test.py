
from gym_balletenv.envs import BalletEnvironment
from wrappers.gray_scale_observation import GrayScaleObservation
import numpy as np
env = BalletEnvironment(num_dancers=2, dance_delay=16, max_steps=320)
env = GrayScaleObservation(env)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs[0])
