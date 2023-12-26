import Env
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import configs
import torch
import random
import numpy as np
from time import time
import cv2
from torch.optim import Adam

#Reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)

import importlib
def parse_args():

    parser = argparse.ArgumentParser(description="Train Tess")
    parser.add_argument("--substrate",type=str,default="clean_up")
    parser.add_argument("--model",type=str,default="impala")
    parser.add_argument("--train-config",type=str,default="ImpalaConfig")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    # Getting config information
    config = getattr(configs, args.train_config) 

    #Setting Up Environment
    env = SubprocVecEnv([lambda: gym.make("TessEnv-v1",\
                        render_mode="rgb_array",
                        name=args.substrate)\
                        for _ in range(config.num_envs)])
    obs_s = env.observation_space.shape
    act_s = env.action_space.shape
    num_act = env.get_attr("num_act")[0]
    
    #Setting Up Model Instance
    model = importlib.import_module(f"models.{args.model}").Model
    agent = model(obs_s[1:], num_act)
    agent = agent.to("cuda")

    # Setting hyperparameters
    num_steps = config.num_steps
    num_envs = config.num_envs
    batch_size = num_envs * num_steps * act_s[0]
    minibatch_size = batch_size // config.minibatch
    num_updates = config.total_timesteps // batch_size
    lr = config.lr
    gae = config.gae
    clip_coef = config.clip_coef
    gamma = config.gamma
    ent_coef = config.ent_coef
    epoch = config.epoch

    #Setting Up Optimizer
    optimizer = Adam(agent.parameters(), lr=lr)

    #Observation Wrapping
    obs = env.reset()
    done = torch.tensor([0]*num_envs,device="cuda").view(-1,1).expand(-1,act_s[0])
    shape = (obs_s[-1],)+obs_s[-3:-1]
    obs = torch.from_numpy(obs)\
        .permute((0,1,4,2,3))\
        .to("cuda")
    

    #Rollout transactions
    roll_o = torch.zeros((num_steps, num_envs, act_s[0],)+shape, device="cuda")
    roll_a = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")
    roll_lp = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")
    roll_rew = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")
    roll_dones = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")
    roll_val = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")

    for update in range(num_updates):

        start = time()

        for step in range(num_steps):

            #Render 1 env
            img = env.get_images()[0]

            if update % 3 == 0:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("image",cv2.resize(img, (700,500)))
                cv2.waitKey(50)
            else:
                cv2.destroyAllWindows()

            roll_dones[step] = done
            roll_o[step] = obs
            
            #Sample action and critic with logprob
            with torch.no_grad():

                batch_o = obs.view((-1,)+shape)
                act, log_prob, value = agent.sample_act_and_value(batch_o)
                
            act = act.view(num_envs,-1)
            log_prob = log_prob.view(num_envs,-1)
            value = value.view(num_envs,-1)

            roll_a[step] = act
            roll_lp[step] = log_prob
            roll_val[step] = value

            obs, rew, done, info = env.step(act)
            obs = torch.from_numpy(obs)\
                .permute((0,1,4,2,3))\
                .to("cuda")
            
            done = torch.from_numpy(done).to("cuda").view(-1,1).expand(-1,act_s[0])
            rew = torch.from_numpy(rew).view(num_envs,-1).to("cuda")
            roll_rew[step] = rew

        with torch.no_grad():

            batch_o = obs.view((-1,)+shape)
            val_plus1 = agent.get_value(batch_o).view(num_envs,-1)

            advantages = torch.zeros(roll_rew.shape[:-1], device="cuda")
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = (1.0 - done.max(dim=1).values)
                    nextvalues = val_plus1
                else:
                    nextnonterminal = 1.0 - roll_dones[t + 1].max(dim=1).values
                    nextvalues = roll_val[t + 1]
                delta = roll_rew[t].sum(dim=-1) + gamma * nextvalues.sum(dim=-1) * nextnonterminal - roll_val[t].sum(dim=-1)
                advantages[t] = lastgaelam = delta + gamma * gae * nextnonterminal * lastgaelam
            returns = advantages + roll_val.sum(dim=-1)

        print(returns.shape)
        print(advantages.shape)

        b_obs = roll_o.view((-1,)+shape)
        b_act = roll_a.view(-1)
        b_logprobs = roll_lp.view(-1)
        #b_returns = returns.view(-1)
        #Checking group success
        b_returns = returns.view(-1)
        #print(b_returns.shape)
        b_adv = advantages.view(-1)

        inds = np.arange(batch_size,)
        for ith_e in range(epoch):
            #np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]

                start_val = start//act_s[0]
                end_val = end//act_s[0]
                minibatch_ind_val = [i for i in range(start_val,end_val)]

                mb_advantages = b_adv[minibatch_ind_val]
                
                mb_obs = b_obs[minibatch_ind]
                mb_actions = b_act[minibatch_ind]

                new_logprob, entropy, value = agent.check_action_and_value(mb_obs, mb_actions)          

                mb_logprob = b_logprobs[minibatch_ind]
                mb_returns = b_returns[minibatch_ind_val]

                log = new_logprob - mb_logprob

                ratio = (log).exp()
                with torch.no_grad():
                    approx_kl = (ratio - 1 - log).mean()

                ratio_clip = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef).view(-1,act_s[0]).mean(dim=-1)
                ratio = ratio.view(-1,act_s[0]).mean(dim=-1)
                pg_loss = torch.max(-mb_advantages * ratio_clip, -mb_advantages * ratio).mean()

                value = value.view(-1,act_s[0]).sum(dim=-1)
                v_loss = (value - mb_returns).square().mean() * .5

                ent_loss = entropy.mean()
                loss =  pg_loss + v_loss - ent_loss * ent_coef

                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(agent.parameters(), .5)
                optimizer.step()


    
        

    
