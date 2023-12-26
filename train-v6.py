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
from augmentation import cutout_color, noise, change_color_channel
import time
from normalize import VecNormalize
from torch.nn import CrossEntropyLoss
cross_loss = CrossEntropyLoss()

torch.set_printoptions(profile="full")

methods = {"noise":noise, "cutout_color":cutout_color, "change_color_channel":change_color_channel}

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
    parser.add_argument("--model",type=str,default="impala_v4_cont")
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
    ids = [False] * (config.num_envs // 2) + [True] * (config.num_envs // 2)
    env = SubprocVecEnv([lambda: gym.make(args.env_version,\
                        render_mode="rgb_array",
                        name=args.substrate, use_event=i)\
                        for i in ids])
    
    env = VecMonitor(env)
    #env = VecNormalize(env)
    obs = env.reset()

    obs_s = env.observation_space.shape
    act_s = env.action_space.shape
    num_act = env.get_attr("num_act")[0]
    print(obs_s)
    print(act_s)
    print(env.observation_space)
    print(env.action_space)

    # Setting hyperparameters
    num_steps = config.num_steps
    num_envs = config.num_envs
    batch_size = num_envs * num_steps 
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
    aug_coef = config.aug_coef
    tasks = config.tasks
    beta = config.beta
    load = config.load
    load_loc = config.load_loc

    #Popart
    means = []
    deviations = []
    embedding = []
    for i in tasks:
        if i not in embedding:
            embedding.append(i)
            means.append(0)
            deviations.append(1)
    vec_means = torch.tensor(means,dtype=torch.float32)
    vec_dev = torch.tensor(deviations,dtype=torch.float32)
    vec_v = torch.ones_like(vec_dev,dtype=torch.float32)
    task_count = len(embedding)


    #Setting Up Model Instance
    print(task_count)
    model = importlib.import_module(f"models.{args.model}").Model
    agent = model(obs_s[1:], num_act, task_count ,inv="prisoner" in args.substrate)
    target_agent = model(obs_s[1:], num_act, task_count ,inv="prisoner" in args.substrate).to("cuda")
    agent = agent.to("cuda")
    if load:
        agent.load_state_dict(torch.load(load_loc))
        vec_means = torch.tensor(config.vec_means,device="cuda")
        vec_dev = torch.tensor(config.vec_dev,device="cuda")
        vec_v = torch.tensor(config.vec_v,device="cuda")
        print("\nLOADED\n")
    print(sum([i.numel() for i in agent.parameters()]))

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
    done = torch.tensor([0]*num_envs,device="cuda").view(-1)
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
    roll_time = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")
    #roll_correct_tasks = torch.tensor([[[1,0],[1,0],[0,1],[0,1]] for i in range(num_steps)], device="cuda", dtype=torch.float32) #Adjust for every other training
    roll_shoot = torch.zeros((num_steps, num_envs, act_s[0]), device="cuda")

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
    start_time = time.time()

    last_choosens = torch.randint(0,9,size=(4,))

    for update in range(num_updates):

        if "territory" in args.substrate:
            pass
            #last_choosens = torch.randint(0,9,size=(2,))

        #last_choosens = torch.randint(0,9,size=(2,))

        annealed_lr = anneal_lr(update)
        annealed_ent_coef = ent_coef

        optimizer.param_groups[0]["lr"] = annealed_lr

        print((update +1) * batch_size)

        #Reset augmentation data 
        agent.aug_data = None
        random_img = random.randint(0,3)

        for step in range(num_steps):

            #Render 1 env
            img = env.get_images()[random_img]
            #img = full_obs[0].permute(1,2,0).cpu().numpy()

            if config.visual:
                if update % 5 == 0:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("image",cv2.resize(img, (700,500)))
                    cv2.waitKey(50)
                else:
                    cv2.destroyAllWindows()
            if config.check_partial_obs:
                img = cv2.cvtColor(env.get_attr("last_partial_obs")[0], cv2.COLOR_RGB2BGR)
                cv2.imshow("image",cv2.resize(img,(700,700),interpolation= cv2.INTER_NEAREST))
                cv2.waitKey(1000)

            time_info = torch.tensor(env.get_attr("counter"),device="cuda").view(-1,1).expand(-1,act_s[0]) / 1000

            ready_shoot_info = torch.tensor(env.get_attr("shoot"),device="cuda")

            roll_dones[step] = done
            roll_o[step] = obs
            roll_time[step] = time_info
            roll_shoot[step] = ready_shoot_info

            if "prisoner" in args.substrate:
                roll_inv[step] = inv
                inv = inv.view(-1,2)
                pri_kwargs = {"inv":inv}
            else:
                pri_kwargs = {"inv":False}

            #Sample action and critic with logprob
            with torch.no_grad():

                batch_o = obs.view((-1,)+shape)
                batch_shoot = ready_shoot_info.view(-1)
                batch_time = time_info.view(-1,1)
                h_0 = h_0.view((-1,)+(agent.last_layer,))
                c_0 = c_0.view((-1,)+(agent.last_layer,))
                act, log_prob, value, (h_0, c_0) = agent.sample_act_and_value(batch_o, history=(h_0,c_0), ready_to_shoot=batch_shoot,\
                    m=config.m, cut_size=config.cut_size, a=config.a, b=config.b, var=config.var, **pri_kwargs)

            act = act.view(num_envs,act_s[0])
            log_prob = log_prob.view(num_envs,act_s[0])
            value = value.view(num_envs,act_s[0],task_count)
            value = torch.stack([value[ind,:,i] for ind,i in enumerate(tasks)],dim=0)

            roll_a[step] = act
            roll_lp[step] = log_prob
            roll_val[step] = value
       
            h_0 = h_0.view(num_envs,act_s[0],agent.last_layer)
            c_0 = c_0.view(num_envs,act_s[0],agent.last_layer)

            #Depends on the training situtations
            # Territory
            act2 = torch.zeros((3,act_s[0]))
            act2[0,:] = torch.randint(0,num_act+1,size=(act_s[0],))
            part = act2[0,:]
            part[part>=num_act] = 7
            act2[0,last_choosens[1]] = act[1,last_choosens[1]]
            act2[1,last_choosens[2]] = act[2,last_choosens[2]]
            act2[2,last_choosens[3]] = act[3,last_choosens[3]]
            #act2[3,last_choosens[3]] = act[3,last_choosens[3]]

            act[1:,:] = act2
            
            obs, rew, done, info = env.step(act)

            if type(pri_kwargs["inv"]) != bool:
                inv = torch.tensor(env.get_attr("invs"), dtype=torch.float32,device="cuda")
            
            real_rewards = []
            episode_rewards = []
            log = False
            save=False
            for ind,i in enumerate(done):
                if i:
                    last_choosens[ind] = random.randint(0,act_s[0]-1)
                    real_ = info[ind]["real_rewards"]
                    epis_ = info[ind]["episode_rewards"]
                    last_act_scores[ind] = real_
                    last_scores[ind] = real_.sum()
                    last_episode_scores[ind] = np.sum(epis_)
                    h_0[ind] *= 0
                    c_0[ind] *= 0
                    done_envs[ind] = True
            
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
                torch.save(agent.state_dict(),f"./Tess/saved_models/{args.substrate}/{args.substrate}-v11.pt")

            #done = torch.from_numpy(done).to("cuda")
            done = torch.from_numpy(done).to("cuda").view(-1,1).expand(-1,act_s[0])

            #Unknown reason sometimes done comes as bool tensor ( probably related with SB3 )
            if done.dtype == torch.bool:
                done = torch.where(done, 1, 0)
            rew = torch.from_numpy(rew).to("cuda")
            roll_rew[step] = rew

        #choosen_players = torch.argmax(roll_rew.sum(dim=0),dim=1).view(-1)
        choosen_players = last_choosens
        print(f"Last choosens: {last_choosens} \nChoosens: {choosen_players}")

        with torch.no_grad():

            if type(pri_kwargs["inv"]) != bool:
                pri_kwargs = {"inv":torch.stack([inv[ind,i,:] for ind, i in enumerate(choosen_players)],dim=0).view(-1,2)}
            else:
                pri_kwargs = {"inv":False}

            batch_time = torch.tensor(env.get_attr("counter"),device="cuda",dtype=torch.float32).view(-1,1) / 1000
            obs_tuple = [obs[ind,i,:,:,:] for ind,i in enumerate(choosen_players)]
            #time_tuple = [time_info[ind,i] for ind,i in enumerate(choosen_players)]
            batch_o = torch.stack(obs_tuple,dim=0)
            #batch_time = torch.stack(time_tuple,dim=0).view(-1,1)

            choosen_h = torch.stack([h_0[index,i,:] for index,i in enumerate(choosen_players)]).view(-1,agent.last_layer)
            choosen_c = torch.stack([c_0[index,i,:] for index,i in enumerate(choosen_players)]).view(-1,agent.last_layer)

            val_plus1 = agent.get_value(batch_o, history=(choosen_h,choosen_c), m=config.m, cut_size=config.cut_size, a=config.a, b=config.b, var=config.var, **pri_kwargs).view(num_envs,task_count)
            val_plus1 = torch.tensor([val_plus1[ind,i] for ind,i in enumerate(tasks)],device="cuda")
            
            choosen_dones = torch.stack([roll_dones[:,ind,i] for ind,i in enumerate(choosen_players)],dim=1)
            cho_done = torch.stack([done[ind,i] for ind,i in enumerate(choosen_players)],dim=0)
            choosen_val = torch.stack([roll_val[:,ind,i] for ind,i in enumerate(choosen_players)],dim=1)
            choosen_rew = torch.stack([roll_rew[:,ind,i] for ind,i in enumerate(choosen_players)],dim=1)
            lastgaelam = 0
            returns = torch.zeros((num_steps,num_envs), device="cuda")
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = (1.0 - cho_done)
                    nextreturn = val_plus1 * torch.tensor([vec_dev[i] for i in tasks],device="cuda") +\
                          torch.tensor([vec_means[i] for i in tasks],device="cuda")
                else:
                    nextnonterminal = 1.0 - choosen_dones[t + 1]
                    nextreturn = returns[t + 1]
                returns[t] = choosen_rew[t] + gamma * nextreturn * nextnonterminal 
            advantages = returns - (choosen_val * torch.tensor([vec_dev[i] for i in tasks],device="cuda") +\
                          torch.tensor([vec_means[i] for i in tasks],device="cuda"))

        b_obs = torch.stack([roll_o[:,ind,i,:,:,:] for ind,i in enumerate(choosen_players)],dim=1).reshape((-1,)+shape)
        b_act = torch.stack([roll_a[:,ind,i] for ind,i in enumerate(choosen_players)],dim=1).reshape(-1)
        b_logprobs = torch.stack([roll_lp[:,ind,i] for ind,i in enumerate(choosen_players)],dim=1).reshape(-1)
        b_time = torch.stack([roll_time[:,ind,i] for ind,i in enumerate(choosen_players)],dim=1).reshape(-1)
        b_returns = returns.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_val = choosen_val.reshape(-1)
        b_shoot = torch.stack([roll_shoot[:,ind,i] for ind,i in enumerate(choosen_players)],dim=1).reshape(-1)
        #b_corr_est = roll_correct_tasks.reshape(-1,task_count)

        if "prisoner" in args.substrate:
            b_invs = torch.stack([roll_inv[:,ind,i,:] for ind,i in enumerate(choosen_players)],dim=1).view(-1,2)

        #Rollback save
        target_agent.load_state_dict(agent.state_dict())
        early_stop = False

        inds = np.arange(batch_size,)
        for ith_e in range(epoch):
            if early_stop == True:
                break
            #np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]

                if type(pri_kwargs["inv"]) != bool:
                    pri_kwargs = {"inv": True}
                    mb_invs = b_invs[minibatch_ind].view(-1,num_envs,2)
                else:
                    pri_kwargs = {"inv": False}

                # Fill LSTM state batches
                mb_obs = b_obs[minibatch_ind].view((-1,num_envs,)+shape)
                mb_time = b_time[minibatch_ind].view(-1,num_envs)
                h__0 = [torch.zeros((num_envs,agent.last_layer),dtype=torch.float32,device="cuda")]
                c__0 = [torch.zeros((num_envs,agent.last_layer),dtype=torch.float32,device="cuda")]

                with torch.no_grad():
                    for i in range(mb_obs.shape[0] - 1):
                        if pri_kwargs["inv"]:
                            (h_n,c_n) = agent.lstm_layer(mb_obs[i].view((-1,)+shape), history=(h__0[-1],c__0[-1]),inv=mb_invs[i].view(-1,2))
                        else: 
                            (h_n,c_n) = agent.lstm_layer(mb_obs[i].view((-1,)+shape), history=(h__0[-1],c__0[-1]))
                        for index,done_info in enumerate(roll_dones[start//num_envs+i+1]):
                            if done_info[0] == 1:
                                h_n[index] *= 0
                                c_n[index] *= 0
                        h__0.append(h_n)
                        c__0.append(c_n)

                h__0 = torch.concatenate(h__0,dim=0)
                c__0 = torch.concatenate(c__0,dim=0)
                
                mb_obs = mb_obs.view((-1,)+shape)
                mb_time = mb_time.view(-1,1)
                mb_advantages = b_adv[minibatch_ind]

                if config.use_advantage_norm:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std()+1e-8)
                
                mb_actions = b_act[minibatch_ind]
                mb_shoots = b_shoot[minibatch_ind]

                if pri_kwargs["inv"]:
                    pri_kwargs["inv"] = mb_invs.view(-1,2)


                new_logprob, entropy, value, estimation = agent.check_action_and_value(mb_obs,\
                        mb_actions, history=(h__0,c__0), mb_shoots=mb_shoots, **pri_kwargs)
                
                #Estimation Loss
                #mb_est = b_corr_est[minibatch_ind]
                #est_loss = cross_loss(estimation, mb_est)
                
                mb_logprob = b_logprobs[minibatch_ind]
                mb_returns = b_returns[minibatch_ind].view(-1,num_envs)

                log = new_logprob - mb_logprob
                
                #Augmentation Loss
                """aug_ratio = (log1 - log2).exp()
                aug_log_loss = (torch.square(aug_ratio - 1) * .5).mean()
                aug_value_loss = (torch.square(value1 - value2) * .5).mean()
                aug_loss = aug_value_loss + aug_log_loss"""
            
                ratio = (log).exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log).mean()
                    if ith_e == epoch-1:
                        pass
                        #print(approx_kl)
                if approx_kl > config.max_kl:
                    #agent.load_state_dict(torch.load(f"./Tess/saved_models/{args.substrate}/{args.substrate}-v2.pt"))
                    agent.load_state_dict(target_agent.state_dict())
                    early_stop = True
                    break
                
                #Policy Loss
                ratio_clip = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(-mb_advantages * ratio_clip, -mb_advantages * ratio).mean()
                #pg_loss = (-mb_advantages * new_logprob).mean() #For ACB

                #Value Loss
                pred_return = torch.stack([(mb_returns[:,ind] - vec_means[i]) / vec_dev[i] for ind,i in enumerate(tasks)],dim=1).view(-1)
                value = value.view(-1,num_envs,task_count)
                value = torch.stack([value[:,ind,i] for ind,i in enumerate(tasks)],dim=1).view(-1) 
                v_loss = (value - pred_return).square().mean() * .5

                ent_loss = entropy.mean()
                loss =  pg_loss + v_loss * config.v_coef\
                    - ent_loss * annealed_ent_coef # + config.est_coef * est_loss # + aug_coef * aug_loss
                
                #Optimization
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), .5)
                optimizer.step()

                if not args.debug:
                    run.log({"entropy":ent_loss,"policy_loss":pg_loss,\
                            "v_loss":v_loss,"kl":approx_kl,\
                            "lr":annealed_lr,"ent_coef":annealed_ent_coef,
                            })

                #print(f"PG_LOSS: {pg_loss} \nV_loss: {v_loss} \nKL: {approx_kl}")
        
        #PopArt Calculations
        temp_means = vec_means.clone()
        temp_vs = vec_v.clone()
        temp_dev = vec_dev.clone()
        mean_returns = torch.mean(returns,dim=0) # Shape num_envs
        dev_returns = torch.std(returns,dim=0) # Shape num_envs

        for ind,i in enumerate(tasks):
            temp_means[i] = (1-beta) * temp_means[i] + beta * mean_returns[ind]
            temp_vs[i] = (1-beta) * temp_vs[i] + beta * torch.square(mean_returns[ind])
            #temp_dev[i] = (1-beta) * temp_dev[i] + beta * dev_returns[i]
        temp_dev = torch.sqrt(temp_vs - torch.square(temp_means))
        
        #PopArt Corrections
        with torch.no_grad():
            for i in range(task_count):
                agent.value.weight[i] *= vec_dev[i] / temp_dev[i]
                agent.value.bias[i] *= vec_dev[i]
                agent.value.bias[i] += vec_means[i] - temp_means[i]
                agent.value.bias[i] /= temp_dev[i]
        
        vec_means = temp_means
        vec_dev = temp_dev
        vec_v = temp_vs


        print(f"Vec Mean: {vec_means} \nVec Dev: {vec_dev} \nVec v: {vec_v}")

        if update==9:
            print(batch_size*10/(time.time()-start_time))


    try:
        os.makedirs(f"./Tess/saved_models/{args.substrate}/",exist_ok=True)
    except:
        pass
    torch.save(agent.state_dict(),f"./Tess/saved_models/{args.substrate}/{args.substrate}-last_output.pt")


