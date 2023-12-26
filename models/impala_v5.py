import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn.functional import relu, max_pool2d
from torch.nn.init import orthogonal_,constant_
class Model(nn.Module):

    def __init__(self, obs_space, act_space, timestep_layer=32, last_layer=256, k=16, device="cuda",inv=False):

        super().__init__()
        self.k = k
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
        self.block1 = ImpalaBlock(3,16)
        self.block2 = ImpalaBlock(16,32)
        self.block3 = ImpalaBlock(32,32)
        if inv:
            last_layer=last_layer+16
            self.inv_linear = nn.Linear(2,16)
        self.fc = nn.Linear(output_size,last_layer)
        self.hierarchy = Hierarchy(last_layer,timestep_layer)
        self.info_to_embed = nn.Linear(last_layer, k, bias=False)
        self.value = nn.Linear(self.last_layer,1)
        self.actor = nn.Linear(self.last_layer,act_space*k)
        self.output_size=output_size
        # OLD self.timestep_linear = nn.Linear(1,timestep_layer)
        # OLD self.harmony = nn.Linear(self.last_layer+timestep_layer,last_layer)
        self.lstm = nn.LSTMCell(input_size=last_layer,hidden_size=self.last_layer)
        self.device=device
        self.aug_data = None

    def set_augmentation_func(self, function):
        self.function = function

    #OLD FUNC
    def harmony_layer(self, x, timestep):

        t = self.timestep_linear(timestep)
        x = torch.concatenate((x,t),dim=1)
        return self.harmony(x)

    def forward(self, x, timestep, history=None, hier_history=None, inv=False):

        history = None
        hier_history = None
        info, (h_hier, c_hier) = self.hierarchy(x, timestep, hier_history)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        y = relu(x)
        x = nn.Flatten()(y)
        x = self.fc(x)
        x = relu(x)
        if type(inv) != bool:
            inv = self.inv_linear(inv)
            x = torch.concatenate((x,inv),dim=1)
        if history is not None:
            h_0, c_0 = self.lstm(x,history)
        else:
            h_0, c_0 = self.lstm(x)

        return h_0 , (h_0, c_0), info, (h_hier, c_hier)
    
    def get_action(self, x, history, timestep, shoot, hiers, inv=False, reduce=True, **kwargs):

        x, (h_0, c_0), info, (h_hier, c_hier) = self.forward(x, timestep, history, hiers, kwargs["inv"])
        vary_logits = self.actor(x).view(-1,self.act_space,self.k) # act x k
        info_to_embed = self.info_to_embed(info).view(-1,self.k,1) # k x 1
        final_logits = vary_logits.matmul(info_to_embed).view(-1,self.act_space)
        cate_o = Categorical(logits=final_logits)
        if reduce:
            new_logits = torch.where(cate_o.probs < .1, -1e+8, new_logits)
            cate_o = Categorical(logits=new_logits)
        act, log_prob = self.sample(new_logits, shoot)
        return act, (h_0, c_0), (h_hier, c_hier), cate_o.entropy() 

    def sample_act_and_value(self, x, history, timestep, shoot, hiers, **kwargs):

        x, (h_0, c_0), info, (h_hier, c_hier) = self.forward(x, timestep, history, hiers, kwargs["inv"])
        value = self.value(info)
        vary_logits = self.actor(x).view(-1,self.act_space,self.k) # act x k
        info_to_embed = self.info_to_embed(info).view(-1,self.k,1) # k x 1
        final_logits = vary_logits.matmul(info_to_embed).view(-1,self.act_space)
        act, log_prob = self.sample(final_logits,shoot)
        return act, log_prob, value, (h_0, c_0), info, (h_hier, c_hier)
    
    def check_action_and_value(self, x, old_info, act, history, hiers, timestep, shoot, inv=False, m=2,**kwargs):
        """ x, _ = self.function(x, **kwargs)
        h_n, c_n = history
        h_n = h_n.view(-1,1,h_n.shape[1]).expand(-1,m,self.last_layer).reshape(-1,self.last_layer)
        c_n = c_n.view(-1,1,h_n.shape[1]).expand(-1,m,self.last_layer).reshape(-1,self.last_layer)
        timestep = timestep.view(-1,1,1).expand(-1,m,1).reshape(-1,1)
        history = (h_n, c_n)"""
        x, _, info, _ = self.forward(x, timestep=timestep, history=history, hier_history=hiers, inv=inv)
        vary_logits = self.actor(x).view(-1,self.act_space,self.k) # act x k
        info_to_embed = self.info_to_embed(old_info).view(-1,self.k,1) # k x 1
        final_logits = vary_logits.matmul(info_to_embed).view(-1,self.act_space)
        value = self.value(info)
        act = act.view(-1)
        shoot = shoot.view(-1)
        log_prob, entropy = self.log_prob(final_logits, act, shoot)
        return log_prob, entropy, value

    def sample(self, logits, shoot):
        logits[:,7] = torch.where(shoot==0, -1e+8, logits[:,7])
        cate_o = Categorical(logits=logits)
        act = cate_o.sample()
        log_prob = cate_o.log_prob(act)
        return act, log_prob
    
    def get_value_with_augmentation(self, x, history, timestep, **kwargs):

        shape = x.shape[1:]
        m = kwargs["m"]
        augmented_x, self.aug_data = self.function(x, aug_data = self.aug_data, **kwargs)
        augmented_x = augmented_x.to(self.device)
        h_n, c_n = history
        h_n = h_n.view(-1,1,self.last_layer).expand(-1,m+1,self.last_layer).reshape(-1,self.last_layer)
        c_n = c_n.view(-1,1,self.last_layer).expand(-1,m+1,self.last_layer).reshape(-1,self.last_layer)
        timestep = timestep.view(-1,1,1).expand(-1,m+1,1).reshape(-1,1)
        history = (h_n, c_n)
        if type(kwargs["inv"]) != bool:
            inv = kwargs["inv"].view(-1,1,2).expand(-1,m+1,2).reshape(-1,2)
            x, _ = self.forward(augmented_x, history=history, inv=inv)
        else:
            x, _ = self.forward(augmented_x, history=history)
        x = self.harmony_layer(x,timestep)
        value = self.value(x).view(-1,m+1)
        value = torch.mean(value, dim=1)
        return value.view(-1)
    
    def get_value(self, x, timestep, history, **kwargs):

        x , _ = self.hierarchy(x, timestep, history=history)
        return self.value(x)
    
    def log_prob(self, logits, act, shoot):

        shoot_part = torch.where(shoot==0, -1e+8, 0)
        logits[:,7] += shoot_part
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
    
class Hierarchy(nn.Module):

    def __init__(self, out_channel, timestep_layer):

        super().__init__()
        self.hier1 = ImpalaBlock(3,16)
        self.hier2 = ImpalaBlock(16,16)
        self.hier3 = ImpalaBlock(16,16)
        hidden_last = 16*6*6
        self.add_info = nn.Linear(hidden_last,out_channel)
        self.lstm = nn.LSTMCell(out_channel,out_channel)
        self.harmony = nn.Linear(out_channel+timestep_layer,out_channel)
        self.timestep_linear = nn.Linear(1,timestep_layer)

    def harmony_layer(self, x, timestep):

        t = self.timestep_linear(timestep)
        x = torch.concatenate((x,t),dim=1)
        return self.harmony(x)

    def forward(self, x, timestep, history=None):

        x = self.hier1(x)
        x = self.hier2(x)
        x = self.hier3(x)
        y = relu(x)
        x = nn.Flatten()(y)
        x = self.add_info(x)
        x = self.harmony_layer(x, timestep)
        if history == None:
            h_n, c_n = self.lstm(relu(x))
        else:
            h_n, c_n = self.lstm(relu(x), history)
        return h_n, (h_n, c_n)
    
