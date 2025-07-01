"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : rMAPPOPolicy.py
"""
import numpy as np
import torch
from gym.spaces import Box

from algorithms.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule


class RMAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,act_space2,num_agents,device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.oil_lr=args.oil_lr
        self.critic_lr = args.critic_lr
        self.cost_lr = args.cost_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_agents=num_agents
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        # self.act_space1 = act_space1
        self.act_space2 = act_space2
        self.actor = R_Actor(args, self.obs_space, self.act_space,self.num_agents,self.device,)
        obs_shape_oil = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.actor_oil = R_Actor(args, obs_shape_oil, self.act_space2, self.num_agents, self.device, )
        self.actor_oil_optimizer = torch.optim.Adam(self.actor_oil.parameters(),
                                                    lr=self.oil_lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.cost_critic = R_Critic(args, self.share_obs_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(),
                                               lr=self.cost_lr,
                                               eps=self.opti_eps,
                                               weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.actor_oil_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.cost_optimizer, episode, episodes, self.cost_lr)
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, rnn_states_cost=None,available_actions=None,
                    deterministic=False):
        actions, action_log_probs,rnn_states_actor,action_logits= self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 deterministic)
        action_logits_tensor = action_logits.logits
        obs = torch.tensor(obs, dtype=torch.float32)  # 如果 obs 是 ndarray，可以转换为 Tensor
        actions_copy = actions
        combined_input = torch.cat((obs,actions_copy ), dim=1)

        oil_type,oil_log_probs, rnn_states_actor,a= self.actor_oil(
            combined_input,
            rnn_states_actor,
            masks,
            deterministic)


        actions_nup = actions.numpy()  # 直接转换
        oil_nup=oil_type.numpy()
        muti_actions=[]

        for i in range(self.num_agents):
            if actions_nup[i][0] == 0:
                if oil_nup[i][0] == 0:
                    muti_actions.append(0)
                elif oil_nup[i][0] == 1:
                    muti_actions.append(1)
                elif oil_nup[i][0] == 2:
                    muti_actions.append(2)

            elif actions_nup[i][0] == 1:
                if oil_nup[i][0] == 0:
                    muti_actions.append(3)
                elif oil_nup[i][0] == 1:
                    muti_actions.append(4)
                elif oil_nup[i][0] == 2:
                    muti_actions.append(5)
            elif actions_nup[i][0] == 2:
                if oil_nup[i][0] == 0:
                    muti_actions.append(6)
                elif oil_nup[i][0] == 1:
                    muti_actions.append(7)
                elif oil_nup[i][0] == 2:
                    muti_actions.append(8)

        muti_actions=torch.tensor(muti_actions)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        cost_preds, rnn_states_cost = self.cost_critic(cent_obs, rnn_states_cost, masks)  # 执行此行
        return values,muti_actions,action_log_probs,oil_log_probs, rnn_states_actor, rnn_states_critic, cost_preds, rnn_states_cost,action_logits_tensor


    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def get_cost_values(self, cent_obs, rnn_states_cost, masks):
        """
        Get constraint cost predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        cost_preds, _ = self.cost_critic(cent_obs, rnn_states_cost, masks)
        return cost_preds
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,action_logits,
                          rnn_states_cost_batch=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_i = []
        oil = []
        # 直接使用 Tensor 的索引来处理
        for i in range(len(action)):
            if action[i][0] == 0:
                oil.append(0)
                action_i.append(0)
            elif action[i][0] == 1:
                action_i.append(0)
                oil.append(1)
            elif action[i][0] == 2:
                action_i.append(0)
                oil.append(2)
            elif action[i][0] == 3:
                action_i.append(1)
                oil.append(0)
            elif action[i][0] == 4:
                action_i.append(1)
                oil.append(1)
            elif action[i][0] == 5:
                action_i.append(1)
                oil.append(2)
            elif action[i][0] == 6:
                action_i.append(2)
                oil.append(0)
            elif action[i][0] == 7:
                action_i.append(2)
                oil.append(1)
            elif action[i][0] == 8:
                action_i.append(2)
                oil.append(2)

        # 将 action_i 和 oil 转换为 Tensor，确保使用同样的设备
        action_i = torch.tensor(action_i).unsqueeze(1)
        oil = torch.tensor(oil).unsqueeze(1)
        active_mask = (action_i == 1) | (action_i == 2)
        active_mask = active_mask * masks
        obs = torch.from_numpy(obs)
        combined_input = torch.cat((obs,  action_i), dim=1)

        action_log_probs,dist_entropy= self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action_i,
                                                                    None,
                                                                     )

        oil_log_probs, dist_entropy_oil= self.actor_oil.evaluate_actions(combined_input,rnn_states_actor,oil,
                                                                         active_mask,)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        cost_values, _ = self.cost_critic(cent_obs, rnn_states_cost_batch, masks)
        return values, action_log_probs,oil_log_probs,dist_entropy,dist_entropy_oil,cost_values,active_mask

    def act(self, obs, rnn_states_actor, masks, deterministic=True):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs,rnn_states_actor,action_logits = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 deterministic)
        # 确保 obs 和 actions 都是 tensor 类型
        # 检查 obs 是否为 list 类型，如果是，转换为 torch.Tensor
        action_logits_tensor = action_logits.logits
        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float32)

        # 检查 actions 是否为 list 类型，如果是，转换为 torch.Tensor
        if isinstance(actions, list):
            actions = torch.tensor(actions, dtype=torch.float32)

        # 如果 obs 或 actions 是 numpy.ndarray 类型，转换为 torch.Tensor
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()

        combined_input = torch.cat((obs, actions), dim=1)

        oil, oil_log_probs, rnn_states_actor,a = self.actor_oil(combined_input ,
                                                          rnn_states_actor,
                                                          masks,
                                                         deterministic)
        muti_actions = []
        for i in range(self.num_agents):
            if actions[i][0] == 0:
                if oil[i][0] == 0:
                    muti_actions.append(0)
                elif oil[i][0] == 1:
                    muti_actions.append(1)
                elif oil[i][0] == 2:
                    muti_actions.append(2)

            elif actions[i][0] == 1:
                if oil[i][0] == 0:
                    muti_actions.append(3)
                elif oil[i][0] == 1:
                    muti_actions.append(4)
                elif oil[i][0] == 2:
                    muti_actions.append(5)
            elif actions[i][0] == 2:
                if oil[i][0] == 0:
                    muti_actions.append(6)
                elif oil[i][0] == 1:
                    muti_actions.append(7)
                elif oil[i][0] == 2:
                    muti_actions.append(8)

        muti_actions = torch.tensor(muti_actions)
        return muti_actions, rnn_states_actor