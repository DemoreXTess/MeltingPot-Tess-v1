import Env
import gym
import argparse
import cv2

import torch


args = argparse.ArgumentParser()
args.add_argument("--substrate",type=str,default="clean_up")
args = args.parse_args()
env = gym.make("TessEnv-v3",name=args.substrate)

obs = env.reset()
done = False
while not done:

    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    #print(env.timestep_o.observation[0]["INVENTORY"])
    rewards = env.timestep_o.reward
    rewards = [int(i) for i in rewards]

    print(env.counter)
    print(rewards)
    #print(rewards)
    #img = env.render()
    #print(img.shape)
    img = env.last_partial_obs
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("image",cv2.resize(img, (600,600), interpolation= cv2.INTER_NEAREST))
    cv2.waitKey(10000)




