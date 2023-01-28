"""gray_scale_observation for balletenv.
original code
    : https://github.com/openai/gym/blob/master/gym/wrappers/gray_scale_observation.py
"""
import os

import numpy as np
import cv2
import imageio
import pygifsicle
import gym
from gym.spaces import Box, Tuple, Discrete


class RecordVideo(gym.Wrapper):

    def __init__(self, env: gym.Env, path: str, fps: int = 15):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)

        assert (
            isinstance(self.observation_space[0], Box)
            and len(self.observation_space[0].shape) == 3
        )
        assert self.observation_space[0].shape[-1] == 3, "observation space must be 3D. Did you use GrayScaleObservation first?"
        self.num_episode = 0
        self.path = path
        self.fps = fps
        self.video_lst = []
    
    def _render_frame(self, obs: np.array, lang: str=None):
        """create black canvas which the size is (99, 200, 3),
        draw obs on the left side and write lang on the right side for rendering cv2"""
        canvas = np.zeros((99, 200, 3), dtype=np.uint8)
        canvas[:, :99, :] = obs
        if lang:
            cv2.putText(canvas, lang, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas
        
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        canvas = self._render_frame(observation[0])
        self.video_lst.append(canvas)
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        canvas = self._render_frame(observation[0], info['instruction_string'])
        self.video_lst.append(canvas)

        if done:
            file_path = os.path.join(self.path, f"ep_{self.num_episode}.gif")
            imageio.mimsave(file_path, self.video_lst, fps=self.fps)
            pygifsicle.optimize(file_path)
            self.video_lst = []
            self.num_episode += 1

        return observation, reward, done, info