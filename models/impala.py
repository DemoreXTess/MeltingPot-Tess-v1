import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn.functional import relu, max_pool2d

class Model(nn.Module):

    def __init__(self, obs_space, act_space, last_layer=256):

        super().__init__()
        self.last_layer=last_layer
        self.obs_space = obs_space
        self.act_space = act_space
        output_size=obs_space[0]
        for _ in range(3):
            if output_size % 2 == 1:
                output_size = (output_size + 1) / 2
            else:
                output_size /= 2
        output_size = int(output_size ** 2 * 32)
        self.block1 = ImpalaBlock(obs_space[2], 16)
        self.block2 = ImpalaBlock(16,32)
        self.block3 = ImpalaBlock(32,32)
        self.value = nn.Linear(self.last_layer,1)
        self.actor = nn.Linear(self.last_layer,act_space)
        self.lstm = nn.LSTMCell(input_size=output_size,hidden_size=self.last_layer)
        self.output_size=output_size

    def forward(self, x, history=None):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        y = relu(x)
        x = nn.Flatten()(y)
        if history is not None:
            h_0, c_0 = self.lstm(x,history)
        else:
            h_0, c_0 = self.lstm(x)

        return h_0 , (h_0, c_0)
    
    def get_action(self, x, history):

        x, (h_0, c_0) = self.forward(x, history)
        new_logits = self.actor(x)
        cate_o = Categorical(logits=new_logits)
        #probs = cate_o.probs
        #print(probs)
        #new_logits = torch.where(probs < .10, -1e+8, probs)
        act, log_prob = self.sample(new_logits)
        return act, (h_0, c_0), cate_o.entropy()

    def sample_act_and_value(self, x, history):

        x, (h_0, c_0) = self.forward(x, history)
        value = self.value(x)
        logits = self.actor(x)
        act, log_prob = self.sample(logits)
        return act, log_prob, value, (h_0, c_0)
    
    def check_action_and_value(self, x, act, history):

        x, _ = self.forward(x, history)
        logits = self.actor(x)
        value = self.value(x)
        log_prob, entropy = self.log_prob(logits, act)
        return log_prob, entropy, value

    def sample(self, logits):

        cate_o = Categorical(logits=logits)
        act = cate_o.sample()
        log_prob = cate_o.log_prob(act)
        return act, log_prob
    
    def get_value(self, x, history):

        x, _ = self.forward(x, history)
        return self.value(x)
    
    def log_prob(self, logits, act):

        cate_o = Categorical(logits=logits)
        log_prob = cate_o.log_prob(act)
        entropy = cate_o.entropy()
        return log_prob, entropy


class ResidualBlock(nn.Module):

    def __init__(self, in_channel):

        super().__init__()
        self.first_cnn = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.second_cnn = nn.Conv2d(in_channel, in_channel, 3, padding=1)

    def forward(self, x):

        y = relu(x)
        y = self.first_cnn(y)
        y = relu(y)
        y = self.second_cnn(y)
        return y + x

class ImpalaBlock(nn.Module):

    def __init__(self, in_channel, out_channel):

        super().__init__()
        self.first_cnn = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.res_block1 = ResidualBlock(out_channel)
        self.res_block2 = ResidualBlock(out_channel)

    def forward(self, x):

        x = self.first_cnn(x)
        x = max_pool2d(x, 3, padding=1, stride=2)
        x = self.res_block1(x)
        return self.res_block2(x)


class LSTMBlock(nn.Module):

    def __init__(self, in_channel, frame, out_channel, obs_space):

        super().__init__()
        self.obs_space = obs_space
        self.lstm = nn.LSTM(input_size=in_channel,hidden_size=out_channel,batch_first=True)
        self.frame = frame
        self.in_channel = in_channel
        

    def forward(self, x, history):

        x = x.view((-1,1,)+self.obs_space)
        x = torch.concat((history,x))
        return self.lstm(input=x)
