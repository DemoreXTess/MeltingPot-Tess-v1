import torch
import cv2
from augmentation import cutout_color, noise, change_color_channel
import Env
import gym
import numpy as np
substrate = "territory__rooms"

env = gym.make("TessEnv-v2",name=substrate) 

obs = env.reset()
partial_obs = env.last_partial_obs
images = [torch.tensor(partial_obs).permute((2,0,1))]

for i in range(2000):

    action_space = env.action_space
    act = action_space.sample()
    env.step(act)
    img = torch.tensor(env.last_partial_obs).permute((2,0,1))
    images.append(img)

images = torch.stack(images, dim=0)

augmented, colors = change_color_channel(images / 255)
for i in range(0,4000,2):

    img1 = augmented[i].permute((1,2,0)).numpy()
    img2 = augmented[i+1].permute((1,2,0)).numpy()
    print(img1.shape)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    img = np.concatenate((img1,img2),axis=1)
    cv2.imshow("image",cv2.resize(img, (1000,500), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(60000)

