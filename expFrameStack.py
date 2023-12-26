
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
import numpy as np
class FrameStack(VecEnvWrapper):

    def __init__(self, venv: VecEnv, frame):
        self.black_screen_envs = lambda: np.zeros((venv.num_envs,)+venv.observation_space.shape,dtype=np.float32)
        self.frame=frame
        self.counter = [frame-1] * venv.num_envs
        super().__init__(venv=venv, observation_space=venv.observation_space)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        blacks = [self.black_screen() for _ in range(self.frame-1)]
        blacks.append(obs)
        stacked_frames = np.concatenate(blacks,axis=0)
        for ind in len(self.counter):
            self.counter[ind] -= 1
        return stacked_frames

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info