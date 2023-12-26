from meltingpot import substrate
import gym
from gym.envs.registration import register
import numpy as np
import dm_env
import cv2
import queue

class TessEnv(gym.Env):

    def __init__(self, render_mode="rgb_array", **kwargs):
        self.name = kwargs["name"]
        self.default_config = substrate.get_config(self.name)
        self.env = substrate.build(kwargs["name"], roles=self.default_config.default_player_roles)
        self.obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()
        self.num_players = len(self.default_config.default_player_roles)
        self.rewards = [0] * self.num_players
        self.observation_space = gym.spaces.Box(0,1,(self.num_players,)+(self.obs_spec[0]["RGB"].shape[0]// 2,self.obs_spec[0]["RGB"].shape[1]// 2,self.obs_spec[0]["RGB"].shape[2]))
        self.num_act = self.action_spec[0].num_values
        self.action_space = gym.spaces.MultiDiscrete([self.action_spec[0].num_values for _ in range(self.num_players)])
        self.env.observables().events.subscribe(on_next=self.on_next)
        self.real_rewards = [0] * self.num_players
        super(TessEnv, self).__init__()
    def reset(self):

        #Clean up specific
        if self.name == "clean_up":
            self.clean_up_histories = [queue.Queue(maxsize=10) for _ in range(self.num_players)]
            for i in self.clean_up_histories:
                for _ in range(10):
                    i.put(0)
            self.clean_rewards = [0] * self.num_players

        #Harvest Specific
        if "harvest" in self.name:
            self.replant_states = [0] * self.num_players

        #Territory Specific
        if "territory" in self.name:
            self.kill_list = [[]]*self.num_players

        res = self.env.reset()
        self.last_rgb = res.observation[0]["WORLD.RGB"]
        obs_list = [cv2.resize(i["RGB"], (i["RGB"].shape[0] // 2,i["RGB"].shape[1] // 2), cv2.INTER_AREA) for i in res.observation]
        obs = np.array(obs_list)
        self.last_partial_obs = obs_list[0]
        obs = np.array(obs / 255, dtype=np.float32)
        self.real_rewards = [0] * self.num_players
        return obs
    
    def convert_oar(self, rewards):

        tanh = np.tanh(rewards)
        max = np.where(tanh < 0, 0, tanh)
        min = np.where(tanh < 0, tanh, 0)
        return 5 * max + .3 * min
    
    def render(self, mode="rgb_array"):
        
        return self.last_rgb

    def step(self, action):
        self.rewards = [0] * self.num_players
        timestep = self.env.step(action)
        self.timestep_o = timestep

        #Clean up specific
        if self.name == "clean_up":
            for ind, clean_reward in enumerate(self.clean_rewards):
                self.clean_up_histories[ind].get()
                self.clean_up_histories[ind].put(clean_reward)
            self.clean_rewards = [0] * self.num_players

        #Inventory specific
        if "prisoner" in self.name:
            invs = [i["INVENTORY"] for i in timestep.observation]
            self.invs = []
            for i in invs:
                summa = sum(i)
                self.invs.append([i[0]/summa,i[1]/summa])
            for ind,i in enumerate(timestep.reward):
                self.real_rewards[ind] += i
                self.rewards[ind] += i

        #Territory specific
        if "territory" in self.name:
            for ind,i in enumerate(timestep.reward):
                self.rewards[ind] += i / 20 * ((len(self.kill_list[ind])+1)**2)
                self.real_rewards[ind] += i
                for u in self.kill_list[ind]:
                    self.rewards[u] += i / 20 * ((len(self.kill_list[ind])+1)**2)
        
        obs_list = [cv2.resize(i["RGB"], (i["RGB"].shape[0] // 2,i["RGB"].shape[1] // 2), cv2.INTER_AREA) for i in timestep.observation]
        obs = np.array(obs_list)
        self.last_partial_obs = obs_list[0]
        obs = np.array(obs / 255, dtype=np.float32)
        done = 1 if int(timestep.step_type) == 2 else 0
        info=dict()
        if done:
            info = {"real_rewards":self.real_rewards}
        self.last_rgb = timestep.observation[0]["WORLD.RGB"]
        #self.rewards = self.convert_oar(self.rewards)
        return obs, self.rewards, done , info
    
    def on_next(self, event):
        
        if "clean_up" in self.name:
            if event[0] == "player_cleaned":
                self.rewards[int(event[1][2])-1] += 1
                self.clean_rewards[int(event[1][2])-1] += 1
            elif event[0] == "edible_consumed":
                for ind,history in enumerate(self.clean_up_histories):
                    hist_list = sum(list(history.queue))
                    self.rewards[ind] += hist_list
                self.rewards[int(event[1][2])-1] += 10
                self.real_rewards[int(event[1][2])-1] += 1

        elif "territory" in self.name:
            if event[0] == "destroyed_resource":
                self.rewards[int(event[1][-1])-1] -= .01
            elif event[0] == "claimed_resource":
                pass
                """self.rewards[int(event[1][-1])-1] += .05
                for i in self.kill_list[int(event[1][-1])-1]:
                    self.rewards[i] += .05"""
            elif event[0] == "removal_due_to_sanctioning":
                source_id = int(event[1][-3])-1
                target_id = int(event[1][-1])-1
                self.kill_list[source_id].append(target_id)
        elif "harvest" in self.name:
            if event[0] == "eating":
                player_id = int(event[1][-3])
                berry_id = int(event[1][-1])
                self.replant_states[player_id-1] = 0
                if berry_id == 2:
                    self.real_rewards[player_id-1] += 1
                    self.rewards[player_id-1] += 2
                else:
                    self.rewards[player_id-1] += 1
            elif event[0] == "replanting":
                player_id = int(event[1][-5])
                berry_id = int(event[1][-1])
                if berry_id == 2:
                    self.rewards[player_id-1] += 1
                    self.replant_states[player_id-1] = 1
                else:
                    self.rewards[player_id-1] -= 1
                    self.replant_states[player_id-1] = 2
            elif event[0] == "zap":
                source = int(event[1][-3])
                target_player = int(event[1][-1])
                if self.replant_states[target_player-1] == 2:
                    self.rewards[source-1] += 1
                else:
                    self.rewards[source-1] -= 1
            elif event[0] == "removal_due_to_sanctioning":
                target = int(event[1][-1])
                self.replant_states[target-1] = 0
        elif "prisoner" in self.name:
            if event[0] == "collected_resource":
                player_id = int(event[1][-3])
                resource = int(event[1][-1])
                if resource == 1:
                    self.rewards[player_id-1] += .3
                    pass
                else:
                    self.rewards[player_id-1] -= .2
            

    


"""env = gym.make("TessEnv-v1", name="clean_up", render_mode="rgb_array")
obs = env.reset()

for i in range(10000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    frame = env.render() 
    frame = cv2.resize(frame, (frame.shape[1]*3,frame.shape[0]*3))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("image",frame)
    cv2.waitKey(200)
    
    if done:
        obs = env.reset()"""


"""env = TessEnv("clean_up")
print(env.action_space)
print(env.observation_space)
obs = env.reset()
action = env.action_space.sample()
timestep = env.step(action)"""

#print(obs)


