"""
# @Time    : 2021/7/1 6:52 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_mappo.py
"""
import math

import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check


class RMAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.entropy_coef_oil=  args.entropy_coef_oil
        self.epoch=0
        self.max_grad_norm = args.max_grad_norm
        self.max_grad_norm_oil=args.max_grad_norm_oil
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.gamma = args.gamma
        self.lagrangian_coef = args.lagrangian_coef_rate  # lagrangian_coef
        self.lamda_lagr = args.lamda_lagr

        self.fair_threshold = 0.05
        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            self.cost_normalizer  = self.policy.cost_critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
            self.cost_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
            self.cost_normalizer = None
    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks:
        #     value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        # else:
        value_loss = value_loss.mean()

        return value_loss
    def cal_cost_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.cost_normalizer.update(return_batch)
            error_clipped = self.cost_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.cost_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks:
        #     value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        # else:
        value_loss = value_loss.mean()

        return value_loss
    def ppo_update(self, sample,t,update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """

        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, old_oil_log_probs_batch,\
        adv_targ,factor_batch,cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ,action_logits\
             = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        old_oil_log_probs_batch = check(old_oil_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        cost_returns_batch = check(cost_return_batch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        action_logits = check(action_logits).to(**self.tpdv)
        #

        values, action_log_probs,oil_log_probs, dist_entropy,dist_entropy_oil, cost_values,active_masks= self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,action_logits,
                                                                              rnn_states_cost_batch)


        adv_targ_hybrid = adv_targ - self.lamda_lagr*cost_adv_targ
        a=adv_targ - self.lamda_lagr*cost_adv_targ
        a=np.array(a)
        b=a.mean()
        imp_weights_oil = torch.exp(oil_log_probs - old_oil_log_probs_batch)

        imp_weights_oil_new =  imp_weights_oil *active_masks + (1 -  active_masks)
        surr1_oil = imp_weights_oil * adv_targ_hybrid
        surr2_oil = torch.clamp(imp_weights_oil, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid
        if self._use_policy_active_masks:
            policy_action_loss_oil = (-torch.sum(torch.min(surr1_oil, surr2_oil),
                                                 dim=-1,
                                                 keepdim=True) * active_masks).sum() / active_masks.sum()
        else:
            policy_action_loss_oil = -torch.sum(torch.min(surr1_oil, surr2_oil), dim=-1, keepdim=True).mean()

            # actor update

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid
        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        imp_weights_combined = imp_weights * imp_weights_oil_new

        self.policy.actor_optimizer.zero_grad()
        # if self.epoch < 10000:
        self.policy.actor_oil_optimizer.zero_grad()
        if update_actor:
            (policy_action_loss-dist_entropy*self.entropy_coef).backward()
            policy_action_loss_new = policy_action_loss-dist_entropy*self.entropy_coef
            # if self.epoch < 10000:
            (policy_action_loss_oil-dist_entropy_oil*self.entropy_coef_oil).backward()
            policy_action_loss_oil_new = policy_action_loss_oil-dist_entropy_oil*self.entropy_coef_oil
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)

            actor_grad_norm_oil = nn.utils.clip_grad_norm_(self.policy.actor_oil.parameters(), self.max_grad_norm_oil)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

            actor_grad_norm_oil = get_gard_norm(self.policy.actor_oil.parameters())
        self.policy.actor_optimizer.step()
        # if self.epoch < 10000:
        self.policy.actor_oil_optimizer.step()
        # critic update
        # 确保在转换为 numpy 之前，将张量移动到 CPU
        # 假设每批数据更新 5 次
        constants = np.full_like(cost_values.detach().cpu().numpy(), self.fair_threshold)
        R_Relu = torch.nn.ReLU()
        constants_tensor = torch.tensor(constants).to(cost_returns_batch.device)
        delta_lamda_lagr = -((cost_returns_batch - constants_tensor)).mean().detach()
        # delta_lamda_lagr = -((cost_returns_batch - constants_tensor) * (1 - self.gamma) + (
        #         imp_weights_combined * cost_adv_targ)).mean().detach()  # 19  ,value_preds_batch in sample , cost_values policy evaluate
        # delta_lamda_lagr = torch.clamp(delta_lamda_lagr, min=-0.1, max=0.1)
        self.lamda_lagr = torch.clamp(R_Relu(self.lamda_lagr - (delta_lamda_lagr * self.lagrangian_coef)),min=10,max=200)
        mean_cost = (cost_returns_batch - constants_tensor) .mean()

        # new_lamda_lagr = R_Relu(
        #     self.lamda_lagr - (delta_lamda_lagr * self.lagrangian_coef))  # self.lagrangian_coef 0.0005

        # self.lamda_lagr = new_lamda_lagr
        lamda_lagr_value = self.lamda_lagr.item()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        cost_loss = self.cal_cost_loss(cost_values, cost_preds_batch, cost_returns_batch)
        self.policy.cost_optimizer.zero_grad()
        (cost_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm)
        else:
            cost_grad_norm = get_gard_norm(self.policy.cost_critic.parameters())
        self.policy.cost_optimizer.step()
        return value_loss, critic_grad_norm, policy_action_loss_new,dist_entropy, actor_grad_norm,\
                cost_loss, cost_grad_norm,\
          policy_action_loss_oil_new,dist_entropy_oil,actor_grad_norm_oil,\
          lamda_lagr_value,delta_lamda_lagr,mean_cost\



    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # min_entropy_oil_coeff = 0.001  # 最小熵系数
        # min_entropy_coeff = 0.005  # 最小熵系数
        # decay_rate = 3e-7  # 衰减速率
        # decay_rate_oil = 2e-7
        self.epoch +=1
        # if self.epoch == 1000:
        #     self.lagrangian_coef = 1e-3
        # if self.epoch<=1500:
        #     self.max_grad_norm = 1
        # if self.epoch==3000:
        #    self.entropy_coef =0
        #    self.entropy_coef_oil =0
        #    self.lagrangian_coef=1e-3#
        #    self._use_max_grad_norm=True
        # self.entropy_coef=self.entropy_coef- (0.99*self.entropy_coef * (self.epoch/ float(30000)))
        # self.entropy_coef_oil=self.entropy_coef_oil- (0.99*self.entropy_coef_oil* (self.epoch/ float(30000)))
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self._use_popart:
            cost_adv = buffer.cost_returns[:-1] - self.cost_normalizer.denormalize(buffer.cost_preds[:-1])
        else:
            cost_adv = buffer.cost_returns[:-1] - buffer.cost_preds[:-1]
        cost_adv_copy = cost_adv.copy()
        cost_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_adv = np.nanmean(cost_adv_copy)
        std_cost_adv = np.nanstd(cost_adv_copy)
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-5)

        train_info = {}
        train_info['cost_loss'] = 0
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        # train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        # train_info['combined_params'] = 0
        train_info['policy_loss_oil'] = 0
        # train_info['dist_entropy_oil'] = 0
        train_info['actor_grad_norm_oil'] = 0
        train_info['mean_cost'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['cost_grad_norm'] = 0
        train_info['cost_grad_norm'] = 0

        train_info['lamda_lagr_value'] = 0
        train_info['delta_lamda_lagrv'] = 0

        self.lamda_lagr =1
        # self.entropy_coef0 -= 0.0001
        for t in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch,cost_adv=cost_adv)

            for sample in data_generator:
                # for k in range(5):
                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, \
                        cost_loss, cost_grad_norm, \
                        policy_oil_loss, dist_entropy_oil, actor_grad_norm_oil, \
                        lamda_lagr_value, delta_lamda_lagr, mean_cost=self.ppo_update(sample,t, update_actor)
                    train_info['value_loss'] += value_loss.item()
                    train_info['policy_loss'] += policy_loss.item()
                    # train_info['dist_entropy'] += dist_entropy.item()
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                    train_info['policy_loss_oil'] +=policy_oil_loss.item()
                    # train_info['dist_entropy_oil'] +=  dist_entropy_oil.item()
                    train_info['actor_grad_norm_oil'] += actor_grad_norm_oil
                    train_info['cost_loss']+=cost_loss
                    train_info['mean_cost'] += mean_cost
                    train_info['cost_grad_norm']+=cost_grad_norm
                    # train_info['combined_params']+=combined_params
                    train_info['delta_lamda_lagrv'] +=delta_lamda_lagr
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        train_info['lamda_lagr_value'] = self.lamda_lagr
        train_info['lagrangian_coef'] = self.lagrangian_coef
        mean_cost = train_info['mean_cost']
        # if mean_cost < 0.2:
        #     self.lagrangian_coef = 0.001
        # if mean_cost < 0.1:
        #     self.lagrangian_coef = 0.0001
        # if mean_cost < 0.1:
        #     self.lagrangian_coef = max(self.lagrangian_coef * 0.5, 1e-6)  # 衰减
        # else:
        #     self.lagrangian_coef = min(self.lagrangian_coef * 1.2, 0.1)
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

