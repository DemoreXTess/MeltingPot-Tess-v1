from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvWrapper
import numpy as np
from copy import deepcopy
class VecMonitor(VecEnvWrapper):

    def __init__(self, env):

        super().__init__(env)
        self.agent_count = self.venv.get_attr("num_players")[0]

    def reset(self):

        obs = self.venv.reset()
        self.episode_rewards = np.zeros((self.venv.num_envs))
        return obs

    def step_async(self, act):
        self.venv.step_async(act)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        self.episode_rewards += rew
        index = 0
        result_info = []
        for d in done:
            new_info = dict()
            if d:
                new_info["real_rewards"] = deepcopy(info[index]["real_rewards"])
                new_info["episode_rewards"] = deepcopy(self.episode_rewards[index])
                self.episode_rewards[index] = 0
                result_info.append(new_info)
            else:
                result_info.append([])
            index += 1
        
        return obs, rew, done, result_info