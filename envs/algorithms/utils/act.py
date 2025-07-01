import numpy as np

from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain,agent_num):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.continuous_action = False
        self.agent_num=agent_num
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
            # self.action_oil_out = Categorical(inputs_dim+ 1, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
            # self.action_oil_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                inputs_dim, discrete_dim, use_orthogonal, gain)])
    
    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        elif self.continuous_action:
            # actions = []
            # action_log_probs = []
            action_logit = self.action_out(x)
            actions = action_logit.mode() if deterministic else action_logit.sample()
            action_log_probs = action_logit.log_probs(actions)
            # actions.append(action.float())
            # action_log_probs.append(action_log_prob)
            # actions = torch.cat(actions, -1)
            # action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            # combined_input = torch.cat((x, actions.float()), dim=1)
            # action_oil_logits =self.action_oil_out(combined_input , available_actions)
            #
            # oil=action_oil_logits.mode() if deterministic else action_logits.sample()
            # if actions.is_cuda:
            #     actions_nup = actions.cpu().numpy()  # 转到 CPU 后转换
            # else:
            #     actions_nup = actions.numpy()  # 直接转换
            # if oil.is_cuda:
            #     oil_nup = oil.cpu().numpy()  # 将张量转为 NumPy 数组
            # else:
            #     oil_nup = oil.numpy()  # 直接转为 NumPy 数组

            # muti_actions=[]
            #
            # for i in range(self.agent_num):
            #     if actions_nup[i][0] == 0:
            #         if oil_nup[i][0] == 0:
            #             muti_actions.append(0)
            #         elif oil_nup[i][0] == 1:
            #             muti_actions.append(1)
            #         elif oil_nup[i][0] == 2:
            #             muti_actions.append(2)
            #
            #     elif actions_nup[i][0] == 1:
            #         if oil_nup[i][0] == 0:
            #             muti_actions.append(3)
            #         elif oil_nup[i][0] == 1:
            #             muti_actions.append(4)
            #         elif oil_nup[i][0] == 2:
            #             muti_actions.append(5)
            #     elif actions_nup[i][0] == 2:
            #         if oil_nup[i][0] == 0:
            #             muti_actions.append(6)
            #         elif oil_nup[i][0] == 1:
            #             muti_actions.append(7)
            #         elif oil_nup[i][0] == 2:
            #             muti_actions.append(8)
            #
            # muti_actions=torch.tensor(muti_actions)
            action_log_probs = action_logits.log_probs(actions)
            # oil_log_probs = action_oil_logits.log_probs(oil)

            # for i in range(self.agent_num):
            #         muti_actions_log_probs.append(action_log_probs[i]+oil_log_probs[i])
            #
            # muti_actions_log_probs = torch.stack(muti_actions_log_probs)
        return actions, action_log_probs,action_logits

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action,available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b] 
            action_log_probs = [] 
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                    else:
                        dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())
                
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98 #! dosen't make sense

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()

        elif self.continuous_action:
            # a, b = action.split((2, 1), -1)
            # b = b.long()
            # action = [a, b]
            action_log_probs = []
            dist_entropy = []
            # for action_out, act in zip(self.action_outs, action):
            action_logit = self.action_out(x)
            action_log_probs.append(action_logit.log_probs(action))
            if active_masks is not None:
                if len(action_logit.entropy().shape) == len(active_masks.shape):
                    dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                else:
                    dist_entropy.append(
                        (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
            else:
                dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] # / 2.0 + dist_entropy[1] / 0.98  # ! dosen't make sense

        else:
            # 假设 action 是一个 Tensor，直接使用它而不是转换为 NumPy 数组
            # ac = np.array(action)  # 这行不再需要



            # 计算 logits
            action_logits = self.action_out(x, available_actions)


            # 计算 log_probs
            action_log_probs = action_logits.log_probs(action)

            a=action_logits.entropy()
            # 计算 entropy
            if active_masks is not None:
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy


