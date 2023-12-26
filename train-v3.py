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
from torch.optim import Adam, RMSprop
import os
from VecMonitor import VecMonitor
from augmentation import cutout_color, noise

methods = {"noise":noise, "cutout_color":cutout_color}

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
    parser.add_argument("--model",type=str,default="impala_v4")
    parser.add_argument("--train-config",type=str,default="ImpalaConfig")
    parser.add_argument("--env-version",type=str,default="TessEnv-v3")
    parser.add_argument("--debug",type=bool,default=False)
    parser.add_argument("--save-stat",type=str,default="mean_real")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    # Getting config information
    config = getattr(configs, args.train_config) 

    #Setting Up Environment
    env = SubprocVecEnv([lambda: gym.make(args.env_version,\
                        render_mode="rgb_array",
                        name=args.substrate)\
                        for _ in range(config.num_envs)])
    env = VecMonitor(env)
    
    obs_s = env.observation_space.shape
    act_s = env.action_space.shape
    num_act = env.get_attr("num_act")[0]
    print(obs_s)
    print(act_s)
    print(env.observation_space)
    print(env.action_space)

    #Setting Up Model Instance
    
    model = importlib.import_module(f"models.{args.model}").Model
    agent = model(obs_s[1:], num_act, inv="prisoner" in args.substrate)
    agent = agent.to("cuda")
    print(sum([i.numel() for i in agent.parameters()]))

    #Data Augmentation
    aug_method = methods[config.aug]
    agent.set_augmentation_func(aug_method)

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
    if not args.debug:
        import wandb
        wandb.login()
        run = wandb.init(project="Tess",name=f"{num_envs}-{num_steps}-{config.minibatch}-{lr}-{clip_coef}-{ent_coef}-{epoch}")

    #Setting Up Optimizer
    if config.optimizer == "rmsprop":
        optimizer = RMSprop(agent.parameters(), lr=lr, eps=1e-5)
    elif config.optimizer == "adam":
        optimizer = Adam(agent.parameters(), lr=lr, eps=1e-5)

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
    roll_val_aug = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")
    roll_time = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")

    #Inventory_specific
    if "prisoner" in args.substrate:
        roll_inv = torch.zeros((num_steps, num_envs, act_s[0], 2), device="cuda")
        inv = torch.ones((num_envs, act_s[0], 2), device="cuda") / 2 
    
    #LSTM helper
    h_0 = torch.zeros((num_envs, act_s[0], agent.last_layer),device="cuda",dtype=torch.float32)
    c_0 = torch.zeros((num_envs, act_s[0], agent.last_layer),device="cuda",dtype=torch.float32)

    #Annealing Stuffs
    anneal_lr = lambda update: lr * (total_timesteps - update*batch_size) / total_timesteps
    anneal_ent = lambda update: ent_coef * (total_timesteps - update*batch_size) / total_timesteps

    #Metrics
    last_scores = [0] * num_envs
    last_act_scores = np.zeros((num_envs,act_s[0]))
    last_episode_scores = [0] * num_envs
    save_stat_score = 0

    #Metric Helpers
    done_envs = [False] * num_envs

    for update in range(num_updates):

        annealed_lr = anneal_lr(update)
        annealed_ent_coef = anneal_ent(update)

        optimizer.param_groups[0]["lr"] = annealed_lr

        print((update +1) * batch_size)

        #Reset augmentation data 
        agent.aug_data = None

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
            if config.check_partial_obs:
                img = cv2.cvtColor(env.get_attr("last_partial_obs")[0], cv2.COLOR_RGB2BGR)
                cv2.imshow("image",cv2.resize(img,(700,700),interpolation= cv2.INTER_NEAREST))
                cv2.waitKey(1000)

            time_d = torch.tensor(env.get_attr("time"),device="cuda").view(num_envs,1).expand(num_envs,act_s[0])
            roll_dones[step] = done
            roll_o[step] = obs
            roll_time[step] = time_d
            
            if "prisoner" in args.substrate:
                roll_inv[step] = inv
                inv = inv.view(-1,2)
                pri_kwargs = {"inv":inv}
            else:
                pri_kwargs = {"inv":False}

            #Sample action and critic with logprob
            with torch.no_grad():

                time_d = time_d.reshape(-1,1)
                batch_o = obs.view((-1,)+shape)
                h_0 = h_0.view((-1,)+(agent.last_layer,))
                c_0 = c_0.view((-1,)+(agent.last_layer,))
                act, log_prob, value, (h_0, c_0), value_aug = agent.sample_act_and_value(batch_o, history=(h_0,c_0), timestep=time_d, m=config.m, cut_size=config.cut_size, a=config.a, b=config.b, var=config.var, **pri_kwargs)

            act = act.view(num_envs,-1)
            log_prob = log_prob.view(num_envs,-1)
            value = value.view(num_envs,-1)
            value_aug = value_aug.view(num_envs,-1)

            roll_a[step] = act
            roll_lp[step] = log_prob
            roll_val[step] = value
            roll_val_aug[step] = value_aug

            h_0 = h_0.view(num_envs,act_s[0],agent.last_layer)
            c_0 = c_0.view(num_envs,act_s[0],agent.last_layer)
            obs, rew, done, info = env.step(act)

            if type(pri_kwargs["inv"]) != bool:
                inv = torch.tensor(env.get_attr("invs"), dtype=torch.float32,device="cuda")
            
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
                high_act = np.max(last_act_scores)
                mean_real = np.mean(last_scores)
                mean_epis = np.mean(last_episode_scores)
                if args.save_stat == "mean_act":
                    if mean_act > save_stat_score:
                        save = True
                        save_stat_score = mean_act
                if args.save_stat == "high_act":
                    if high_act > save_stat_score:
                        save = True
                        save_stat_score = high_act
                if args.save_stat == "mean_real":
                    if mean_real > save_stat_score:
                        save = True
                        save_stat_score = mean_real
                if args.save_stat == "mean_epis":
                    if mean_epis > save_stat_score:
                        save = True
                        save_stat_score = mean_epis
                if not args.debug:
                    run.log({"mean_real":mean_real,"std_scores":std_epis,\
                            "mean_epis":mean_epis,"std_acts":std_acts,\
                            "mean_act":mean_act,"highest_act":high_act})
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

            if type(pri_kwargs["inv"]) != bool:
                pri_kwargs = {"inv":inv.view(-1,2)}
            else:
                pri_kwargs = {"inv":False}

            time_d = torch.tensor(env.get_attr("time"),device="cuda").view(num_envs,1).expand(num_envs,act_s[0]).reshape(-1,1)
            batch_o = obs.view((-1,)+shape)
            h_0 = h_0.view(-1,agent.last_layer)
            c_0 = c_0.view(-1,agent.last_layer)
            val_plus1 = agent.get_value_with_augmentation(batch_o, history=(h_0,c_0), timestep=time_d, m=config.m, cut_size=config.cut_size, a=config.a, b=config.b, var=config.var, **pri_kwargs).view(num_envs,-1)

            advantages = torch.zeros_like(roll_rew, device="cuda")
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = (1.0 - done)
                    nextvalues = val_plus1
                else:
                    nextnonterminal = 1.0 - roll_dones[t + 1]
                    nextvalues = roll_val_aug[t + 1]
                delta = roll_rew[t] + gamma * nextvalues * nextnonterminal - roll_val_aug[t]
                advantages[t] = lastgaelam = delta + gamma * gae * nextnonterminal * lastgaelam
            returns = advantages + roll_val

        b_obs = roll_o.view((-1,)+shape)
        b_act = roll_a.view(-1)
        b_logprobs = roll_lp.view(-1)
        b_returns = returns.view(-1)
        b_adv = advantages.view(-1)
        b_val = roll_val.view(-1)
        b_time = roll_time.view(-1)

        if "prisoner" in args.substrate:
            b_invs = roll_inv.view(-1,2)

        inds = np.arange(batch_size,)
        for ith_e in range(epoch):
            #np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]

                if type(pri_kwargs["inv"]) != bool:
                    pri_kwargs = {"inv": True}
                    mb_invs = b_invs[minibatch_ind].view(-1,num_envs,act_s[0],2)
                else:
                    pri_kwargs = {"inv": False}

                # Fill LSTM state batches
                mb_obs = b_obs[minibatch_ind].view((-1,num_envs, act_s[0],)+shape)
                
                if start == 0:
                    h__0 = [torch.zeros((num_envs*act_s[0],agent.last_layer),dtype=torch.float32,device="cuda")]
                    c__0 = [torch.zeros((num_envs*act_s[0],agent.last_layer),dtype=torch.float32,device="cuda")]
                else:
                    with torch.no_grad():
                        if pri_kwargs["inv"]:
                            pre_inv = b_invs[start-config.burn_in*num_envs*act_s[0]:start].view(-1,num_envs,act_s[0],2)
                        else:
                            pre_inv = False
                        pre_obs = b_obs[start-config.burn_in*num_envs*act_s[0]:start].view((-1,num_envs,act_s[0],)+shape)
                        pre_h = torch.zeros((num_envs*act_s[0],agent.last_layer),dtype=torch.float32,device="cuda")
                        pre_c = torch.zeros((num_envs*act_s[0],agent.last_layer),dtype=torch.float32,device="cuda")
                        pre_dones = roll_dones[start//num_envs//act_s[0]-config.burn_in:start//num_envs//act_s[0]+1]
                        for i in range(pre_obs.shape[0]):
                            if config.burn_in == 0:
                                print("BUG")
                            if type(pre_inv) != bool:
                                _ , (pre_h,pre_c) = agent.forward(pre_obs[i].view((-1,)+shape),history=(pre_h,pre_c),inv=pre_inv[i].view(-1,2))
                            else:
                                _ , (pre_h,pre_c) = agent.forward(pre_obs[i].view((-1,)+shape),history=(pre_h,pre_c))
                            for index,done_info in enumerate(pre_dones[i+1]):
                                if done_info[0] == 1:
                                    pre_h = pre_h.view(num_envs,act_s[0],agent.last_layer)
                                    pre_c = pre_c.view(num_envs,act_s[0],agent.last_layer)
                                    pre_h[index] *= 0
                                    pre_c[index] *= 0
                                    pre_h = pre_h.view(-1,agent.last_layer)
                                    pre_c = pre_c.view(-1,agent.last_layer)
                    h__0 = [pre_h]
                    c__0 = [pre_c]
                with torch.no_grad():
                    for i in range(mb_obs.shape[0] - 1):
                        if pri_kwargs["inv"]:
                            _, (h_n,c_n) = agent.forward(mb_obs[i].view((-1,)+shape),history=(h__0[-1],c__0[-1]),inv=mb_invs[i].view(-1,2))
                        else: 
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
                mb_time = b_time[minibatch_ind].view(-1,1)

                if config.use_advantage_norm:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std()+1e-8)
                
                mb_actions = b_act[minibatch_ind]

                if pri_kwargs["inv"]:
                    pri_kwargs["inv"] = mb_invs.view(-1,2)

                new_logprob, entropy, value = agent.check_action_and_value(mb_obs,\
                                            mb_actions, history=(h__0,c__0), timestep=mb_time, **pri_kwargs)          

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
                loss =  pg_loss + v_loss * config.v_coef - ent_loss * annealed_ent_coef
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), .5)
                optimizer.step()

        if not args.debug:
            run.log({"entropy":ent_loss,"policy_loss":pg_loss,\
                    "v_loss":v_loss,"kl":approx_kl,\
                    "lr":annealed_lr,"ent_coef":annealed_ent_coef})


    
        

    
