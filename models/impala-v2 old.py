import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn.functional import relu, max_pool2d
from torch.nn.init import orthogonal_,constant_
class Model(nn.Module):

    def __init__(self, obs_space, act_space, last_layer=256, device="cuda",inv=False):

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
        if inv:
            self.inv_linear = nn.Linear(2,output_size)
        self.block1 = ImpalaBlock(3,16)
        self.block2 = ImpalaBlock(16,32)
        self.block3 = ImpalaBlock(32,32)
        self.value = nn.Linear(self.last_layer,1)
        self.actor = nn.Linear(self.last_layer,act_space)
        self.lstm = nn.LSTMCell(input_size=output_size,hidden_size=self.last_layer)
        self.output_size=output_size
        self.device=device

    def set_augmentation_func(self, function):
        self.function = function

    def forward(self, x, history=None, inv=False):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        y = relu(x)
        x = nn.Flatten()(y)
        if type(inv) != bool:
            x = self.inv_linear(inv) + x
        if history is not None:
            h_0, c_0 = self.lstm(x,history)
        else:
            h_0, c_0 = self.lstm(x)

        return h_0 , (h_0, c_0)
    
    def get_action(self, x, history, inv=False):

        x, (h_0, c_0) = self.forward(x, history, inv=inv)
        new_logits = self.actor(x)
        cate_o = Categorical(logits=new_logits)
        act, log_prob = self.sample(new_logits)
        return act, (h_0, c_0), cate_o.entropy()

    def sample_act_and_value(self, x, history, **kwargs):

        value_aug = self.get_value_with_augmentation(x, history, **kwargs)
        x, (h_0, c_0) = self.forward(x, history, kwargs["inv"])
        value = self.value(x)
        logits = self.actor(x)
        act, log_prob = self.sample(logits)
        return act, log_prob, value, (h_0, c_0), value_aug
    
    def check_action_and_value(self, x, act, history, inv=False):

        x, _ = self.forward(x, history=history, inv=inv)
        logits = self.actor(x)
        value = self.value(x)
        log_prob, entropy = self.log_prob(logits, act)
        return log_prob, entropy, value

    def sample(self, logits):

        cate_o = Categorical(logits=logits)
        act = cate_o.sample()
        log_prob = cate_o.log_prob(act)
        return act, log_prob
    
    def get_value_with_augmentation(self, x, history, **kwargs):

        shape = x.shape[1:]
        m = kwargs["m"]
        augmented_x = self.function(x,**kwargs).to(self.device)
        h_n, c_n = history
        h_n = h_n.view(-1,1,self.last_layer).expand(-1,m+1,self.last_layer).reshape(-1,self.last_layer)
        c_n = c_n.view(-1,1,self.last_layer).expand(-1,m+1,self.last_layer).reshape(-1,self.last_layer)
        history = (h_n, c_n)
        if type(kwargs["inv"]) != bool:
            inv = kwargs["inv"].view(-1,1,2).expand(-1,m+1,2).reshape(-1,2)
            x, _ = self.forward(augmented_x, history=history, inv=inv)
        else:
            x, _ = self.forward(augmented_x, history=history)
        value = self.value(x).view(-1,m+1)
        value = torch.mean(value, dim=1)
        return value.view(-1)
    
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

    def __init__(self, in_channel, out_channel, max_pool_kernel=3, stride=2):

        super().__init__()
        self.stride = stride
        self.max_pool_kernel = max_pool_kernel
        self.first_cnn = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.res_block1 = ResidualBlock(out_channel)
        self.res_block2 = ResidualBlock(out_channel)

    def forward(self, x):

        x = self.first_cnn(x)
        x = max_pool2d(x, self.max_pool_kernel, padding=1, stride=self.stride)
        x = self.res_block1(x)
        return self.res_block2(x)
    
class ShrinkLayer(nn.Module):

    def __init__(self, in_channel, out_channel):

        super().__init__()
        self.shrink1 = nn.Conv2d(in_channel,64,3,stride=2,padding=1)
        self.shrink2 = nn.Conv2d(64,out_channel,3,stride=2,padding=1)

    def forward(self, x):

        x = self.shrink1(x)
        x = relu(x)
        x = self.shrink2(x)
        x = relu(x)
        return x