import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env,skip=4, use_max = True):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self.use_max = use_max 
        # most recent raw observations (for max pooling across time steps)
        if self.use_max:
            self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        else:
            self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.float32)
        self._skip       = skip
        

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            #try:
            obs, reward, done, info = self.env.step(action)
            #except TypeError:
            #    obs, reward, done, info = self.env.step(action)

            #print(obs,reward,done,info,flush=True)
            if self.use_max:
                if i == self._skip - 2: self._obs_buffer[0] = obs
                if i == self._skip - 1: self._obs_buffer[1] = obs
            else:
                self._obs_buffer[0] = obs

            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        if self.use_max:
            max_frame = self._obs_buffer.max(axis=0)
        else:
            max_frame = self._obs_buffer[0]

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, flat = False):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.flat = flat
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        if flat:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        #print(ob,reward,done,info,flush=True)
        self.frames.appendleft(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        if self.flat:
            return np.squeeze(self.frames).flatten()
        else:
            return np.squeeze(np.swapaxes(self.frames, 3, 0))

