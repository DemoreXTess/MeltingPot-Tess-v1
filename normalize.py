from stable_baselines3.common.vec_env import VecEnvWrapper

class VecNormalize(VecEnvWrapper):

    def __init__(self, venv, gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        
        self.ret_rms = RunningMeanStd(shape=())
        self.agent_count = self.venv.get_attr("num_players")[0]
        self.ret = np.zeros((self.num_envs,self.agent_count))
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            self.ret_rms.update(self.ret.reshape(-1))
            rews = rews / np.sqrt(self.ret_rms.var + self.epsilon)

        done_info = np.array(dones, dtype = bool)
    
        self.ret[done_info] = np.zeros(self.agent_count)
        return obs, rews, dones, infos

    def reset(self):
        self.ret = np.zeros((self.num_envs,self.agent_count))
        obs = self.venv.reset()
        return obs
    
import numpy as np
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count