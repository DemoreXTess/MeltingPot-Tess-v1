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
import os
from VecMonitor import VecMonitor

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
    env = VecMonitor(env)
    
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
    total_timesteps = config.total_timesteps
    num_updates = total_timesteps // batch_size
    lr = config.lr
    gae = config.gae
    clip_coef = config.clip_coef
    gamma = config.gamma
    ent_coef = config.ent_coef
    epoch = config.epoch
    clip_v_loss=config.clip_v_loss

    #Wandb
    import wandb
    wandb.login()
    run = wandb.init(project="Tess",name=f"{num_envs}-{num_steps}-{config.minibatch}-{lr}-{clip_coef}-{ent_coef}-{epoch}")

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
    
    #LSTM helper
    h_0 = torch.zeros((num_envs, act_s[0], agent.last_layer),device="cuda",dtype=torch.float32)
    c_0 = torch.zeros((num_envs, act_s[0], agent.last_layer),device="cuda",dtype=torch.float32)

    #Annealing Stuffs
    anneal_lr = lambda update: lr * (total_timesteps - update*batch_size) / total_timesteps

    #Metrics
    last_scores = [0] * num_envs
    last_act_scores = np.zeros((num_envs,act_s[0]))
    last_episode_scores = [0] * num_envs
    mean_act_score = 0

    #Metric Helpers
    done_envs = [False] * num_envs

    for update in range(num_updates):

        annealed_lr = anneal_lr(update)
        optimizer.param_groups[0]["lr"] = annealed_lr

        print((update +1) * batch_size)

        for step in range(num_steps):

            #Render 1 env
            img = env.get_images()[0]

            if config.visual:
                if update % 10 == 0:
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
                h_0 = h_0.view((-1,)+(agent.last_layer,))
                c_0 = h_0.view((-1,)+(agent.last_layer,))
                act, log_prob, value, (h_0, c_0) = agent.sample_act_and_value(batch_o, history=(h_0,c_0))
                
            act = act.view(num_envs,-1)
            log_prob = log_prob.view(num_envs,-1)
            value = value.view(num_envs,-1)

            roll_a[step] = act
            roll_lp[step] = log_prob
            roll_val[step] = value

            h_0 = h_0.view(num_envs,act_s[0],agent.last_layer)
            c_0 = c_0.view(num_envs,act_s[0],agent.last_layer)
            obs, rew, done, info = env.step(act)

            real_rewards = []
            episode_rewards = []
            log = False
            for ind,i in enumerate(done):
                if i:
                    log = True
                    real_ = info[ind]["real_rewards"]
                    epis_ = info[ind]["episode_rewards"]
                    last_act_scores[ind] = real_
                    last_scores[ind] = real_.sum()
                    last_episode_scores[ind] = epis_.sum()
                    h_0[ind] *= 0
                    c_0[ind] *= 0
                    done_envs[ind] = True

            save = False
            if not (False in done_envs):
                done_envs = [False] * num_envs
                std_epis = np.std(last_scores)
                std_acts = np.std(last_act_scores)
                mean_act = np.mean(last_act_scores)
                if mean_act > mean_act_score:
                    save = True
                    mean_act_score = mean_act
                mean_epis = np.mean(last_episode_scores)
                mean_real = np.mean(last_scores)
                run.log({"mean_real":mean_real,"std_scores":std_epis,\
                           "mean_epis":mean_epis,"std_acts":std_acts,\
                            "mean_act":mean_act})
                last_scores = [0] * num_envs
                last_act_scores = np.zeros((num_envs,act_s[0]))
                last_episode_scores = [0] * num_envs
                

            obs = torch.from_numpy(obs)\
                .permute((0,1,4,2,3))\
                .to("cuda")
            
            
            if save:
                try:
                    os.makedirs(f"./Tess/saved_models/{args.substrate}/",exist_ok=True)
                except:
                    pass
                torch.save(agent.state_dict(),f"./Tess/saved_models/{args.substrate}/{args.substrate}-v2.pt")

            done = torch.from_numpy(done).to("cuda").view(-1,1).expand(-1,act_s[0])
            #Unknown reason sometimes done comes as bool tensor ( probably related with SB3 )
            if done.dtype == torch.bool:
                done = torch.where(done, 1, 0)
            rew = torch.from_numpy(rew).view(num_envs,-1).to("cuda")
            roll_rew[step] = rew

        with torch.no_grad():

            batch_o = obs.view((-1,)+shape)
            h_0 = h_0.view(-1,agent.last_layer)
            c_0 = c_0.view(-1,agent.last_layer)
            val_plus1 = agent.get_value(batch_o, history=(h_0,c_0)).view(num_envs,-1)

            advantages = torch.zeros_like(roll_rew, device="cuda")
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = (1.0 - done)
                    nextvalues = val_plus1
                else:
                    nextnonterminal = 1.0 - roll_dones[t + 1]
                    nextvalues = roll_val[t + 1]
                delta = roll_rew[t] + gamma * nextvalues * nextnonterminal - roll_val[t]
                advantages[t] = lastgaelam = delta + gamma * gae * nextnonterminal * lastgaelam
            returns = advantages + roll_val

        b_obs = roll_o.view((-1,)+shape)
        b_act = roll_a.view(-1)
        b_logprobs = roll_lp.view(-1)
        b_returns = returns.view(-1)
        b_adv = advantages.view(-1)
        b_val = roll_val.view(-1)

        inds = np.arange(batch_size,)
        for ith_e in range(epoch):
            #np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]

                # Fill LSTM state batches
                mb_obs = b_obs[minibatch_ind].view((-1,num_envs, act_s[0],)+shape)
                h__0 = [torch.zeros((num_envs*act_s[0],agent.last_layer),dtype=torch.float32,device="cuda")]
                c__0 = [torch.zeros((num_envs*act_s[0],agent.last_layer),dtype=torch.float32,device="cuda")]
                with torch.no_grad():
                    for i in range(mb_obs.shape[0] - 1):
                        _, (h_n,c_n) = agent.forward(mb_obs[i].view((-1,)+shape),history=(h__0[-1],c__0[-1]))
                        for index,done_info in enumerate(roll_dones[start//act_s[0]//num_envs+i+1]):
                            if done_info[0] == 1:
                                h_n = h_n.view(num_envs,act_s[0],agent.last_layer)
                                c_n = c_n.view(num_envs,act_s[0],agent.last_layer)
                                h_n[index] *= 0
                                c_n[index] *= 0
                                h_n = h_n.view(-1,agent.last_layer)
                                c_n = c_n.view(-1,agent.last_layer)
                        h__0.append(h_n)
                        c__0.append(c_n)

                h__0 = torch.concatenate(h__0,dim=0)
                c__0 = torch.concatenate(c__0,dim=0)
                
                mb_obs = mb_obs.view((-1,)+shape)
                mb_advantages = b_adv[minibatch_ind]

                if config.use_advantage_norm:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std()+1e-8)
                
                mb_actions = b_act[minibatch_ind]

                new_logprob, entropy, value = agent.check_action_and_value(mb_obs,\
                                            mb_actions, history=(h__0,c__0))          

                mb_logprob = b_logprobs[minibatch_ind]
                mb_returns = b_returns[minibatch_ind]
                mb_values = b_val[minibatch_ind]

                log = new_logprob - mb_logprob

                ratio = (log).exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log).mean()
                    if ith_e == epoch-1:
                        pass
                        #print(approx_kl)
                ratio_clip = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(-mb_advantages * ratio_clip, -mb_advantages * ratio).mean()

                #Clipping Value Loss
                if clip_v_loss:
                    clip_v = config.clip_v
                    v_loss = (value - mb_returns).square()
                    v_clipped = mb_values + torch.clamp(
                        value - mb_values,
                        -clip_v,
                        clip_v,
                    )
                    v_loss_clipped = (v_clipped - mb_returns).square()
                    v_loss_max = torch.max(v_loss, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = (value - mb_returns).square().mean() * .5


                ent_loss = entropy.mean()
                loss =  pg_loss + v_loss - ent_loss * ent_coef
                run.log({"entropy":ent_loss,"policy_loss":pg_loss,\
                           "v_loss":v_loss,"kl":approx_kl})
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), .5)
                optimizer.step()


    
        

    
