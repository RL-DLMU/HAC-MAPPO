"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_actor_critic.py
"""
import numpy as np

from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.popart import PopArt
from utils.util import get_shape_from_obs_space
import torch
import torch.nn as nn
import torch.distributions as distributions

class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space,agent_num,device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.rate = 1
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.agent_num=agent_num
        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain,self.agent_num)
        # self.act_continous = DiscreteActor(self.hidden_size)
        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=True):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions,  action_log_probs,action_logits = self.act(actor_features)
        return  actions,  action_log_probs, rnn_states,action_logits

    def evaluate_actions(self, obs, rnn_states, action, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        assert not torch.isnan(obs).any(), "Input observations contain NaN values."
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        actor_features = self.base(obs)


        action_log_probs,dist_entropy = self.act.evaluate_actions(actor_features ,
                                                                   action ,
                                                                  active_masks=
                                                                  active_masks if self._use_policy_active_masks
                                                                  else None)



        return action_log_probs,dist_entropy

class DiscreteActor(nn.Module):
    def __init__(self, hidden_dim):
        super(DiscreteActor, self).__init__()
        # 一个网络用于仓类型选择 (单仓/双仓)
        self.warehouse_fc = nn.Linear(hidden_dim, 2)  # 2类：单仓or双仓
        # 另一个网络用于油品类型选择 (92, 95, 柴油)
        self.oil_fc = nn.Linear(hidden_dim, 3)  # 3类：92号、95号、柴油

    def forward(self, encoded_state, discrete_action):
        # 获取两个网络的logits
        # discrete_action=np.array(discrete_action)
        # warehouse_logits = self.warehouse_fc(encoded_state)  # [6, 2]，6个智能体，2类仓类型
        # oil_logits = self.oil_fc(encoded_state)  # [6, 3]，6个智能体，3类油品类型

        # 创建分类分布
        warehouse_logit = distributions.Categorical(2, 2, self._use_orthogonal)
        oil_logit = distributions.Categorical(3, 3, self._use_orthogonal)
        warehouse_type = warehouse_logit.mode()  # [6, 1]
        oil_type = oil_logit.mode()
        warehouse_log_probs = warehouse_logit.log_probs(warehouse_type)
        oil_log_probs = oil_logit.log_probs(oil_type)
        warehouse_types=[]
        oil_types=[]
        warehouse_log_probs=[]
        oil_log_probs=[]
        # type= warehouse_dist.mode()
        # 如果出车，采样离散动作
        for i in range(6):
            if discrete_action[i][0] == 1:
                warehouse_type = a[i].unsqueeze(0)  # [6, 1]
                oil_type = oil_dist.mode().unsqueeze(-1)  # [6, 1]

                # 计算 log 概率
                warehouse_log_prob = warehouse_dist.log_prob(warehouse_type.squeeze()).unsqueeze(-1)  # [6, 1]
                oil_log_prob = oil_dist.log_prob(oil_type.squeeze()).unsqueeze(-1)  # [6, 1]
                warehouse_types.append(warehouse_type)
                oil_types.append(oil_type)
                warehouse_log_probs.append(warehouse_log_prob)
                oil_log_probs.append(oil_log_prob)
            else:
                # 不出车时，返回无效动作值（-1，-1），并设置log_prob为0
                warehouse_types.append(-1)
                oil_types.append(-1)
                warehouse_log_probs.append(0)
                oil_log_probs.append(0)

        return torch.tensor(warehouse_types), torch.tensor(oil_types),torch.tensor( warehouse_log_probs), torch.tensor(oil_log_probs)


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.dropout = nn.Dropout(p=0.1)  # 随机丢弃10%的神经元

        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.v_out = nn.utils.spectral_norm(self.v_out)  # 防止梯度爆炸

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        critic_features = self.dropout(critic_features)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)
        # costs_values  = self.v_out(critic_features)

        return values, rnn_states
