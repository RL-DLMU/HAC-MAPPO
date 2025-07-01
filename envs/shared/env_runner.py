import math
import os
import time
import numpy as np
import torch
from shared.base_runner import Runner
import  json
# import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        with open("HAC-MAPPO.json", "w") as f:
            json.dump([], f)
        print(f"Log file  has been cleared.")
        save_path = 'D:\\Q-realse\\save_path'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # agent_num = 6  # 智能体数量
        episode_rewards = []
        episode_costs=[]
        c_total_92_empty = []
        c_total_95_empty = []
        c_total_derv_empty = []
        c_total_92_full = []
        c_total_95_full = []
        c_total_derv_full = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ----------------------------------------- #
        # 环境设置--onpolicy
        # ----------------------------------------- #
        num_episodes =10001

        epsilon = 0.1
        epsilon_decay = 0.995
        min_epsilon = 0.01

        station_num=self.num_agents
        for episode in range(num_episodes):
            if episode % 10 == 0 :
                save_path = f'model_checkpoint_{episode}.pth'
                torch.save({
                    'actor_state_dict': self.policy.actor.state_dict(),
                    'actor_oil_state_dict': self.policy.actor_oil.state_dict(),
                }, save_path)
                print(f"Checkpoint saved at step {episode}: {save_path}")

            print(episode)
            # if episode==15000:
            #     self.entropy_coef =0.0005
            #     self.entropy_coef_oil =0.0005


            if self.use_linear_lr_decay :

            # #     # self.lagrangian_coef_rate=self.lagrangian_coef_rate-(0.9*self.entropy_coef * (episode / float(num_episodes)))

                self.trainer.policy.lr_decay(episode, num_episodes)
            state =self.envs.reset()
            # replay buffer
            state = np.array(state)
            share_obs = state.reshape(self.n_rollout_threads, -1) # shape = [env_num, agent_num * obs_dim]
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  # shape = shape = [env_num, agent_num， agent_num * obs_dim]

            self.buffer.share_obs[0] = share_obs.copy()
            self.buffer.obs[0] =  state.copy()
            done = []


            for step in range(self.episode_length):

                self.trainer.prep_rollout()
                actions_92 = []
                actions_95 = []
                actions_derv = []
                action_total = []
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,oil_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    cost_preds,
                    rnn_states_cost,action_logits_tensor
                ) = self.collect(step)
                # joint_act = []

                for station_index in range(station_num):
                    # ε-贪心策略选择动作
                    if actions[0][station_index] == 0 or actions[0][station_index] == 1 or actions[0][station_index] == 2:
                        actions_92.append(0)
                        actions_95.append(0)
                        actions_derv.append(0)
                    elif actions[0][station_index]== 3:
                                actions_92.append(1)
                                actions_95.append(0)
                                actions_derv.append(0)
                    elif actions[0][station_index] == 4:
                                actions_92.append(0)
                                actions_95.append(1)
                                actions_derv.append(0)
                    elif actions[0][station_index] == 5:
                                actions_92.append(0)
                                actions_95.append(0)
                                actions_derv.append(1)
                    elif actions[0][station_index] == 6:
                            actions_92.append(2)
                            actions_95.append(0)
                            actions_derv.append(0)
                    elif actions[0][station_index] == 7:
                            actions_92.append(0)
                            actions_95.append(2)
                            actions_derv.append(0)
                    elif actions[0][station_index] == 8:
                            actions_92.append(0)
                            actions_95.append(0)
                            actions_derv.append(2)
                action_total.append(actions_92)
                action_total.append(actions_95)
                action_total.append(actions_derv)
                next_state, reward, dones,cost = self.envs.step(action_total)

                state = next_state
                rewards = np.reshape(reward, (self.n_rollout_threads, self.num_agents, 1))
                costs = np.reshape(cost, (self.n_rollout_threads, self.num_agents, 1))
                state = np.reshape(state, (self.n_rollout_threads, self.num_agents, self.envs.obs_dim))
                actions = np.reshape(actions, (self.n_rollout_threads, self.num_agents, 1))
                done.append(dones)
                data = (
                    state,
                    rewards,
                    costs,
                    np.array(done),
                    values,
                    actions,
                    action_log_probs,oil_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    cost_preds, rnn_states_cost,action_logits_tensor
                )
                self.insert(data)

                # share_state = next_state_share
            # print('92空舱数',self.envs.empty92)
            # print('95空舱数',self.envs.empty95)
            # print('柴油空舱数',self.envs.emptyderv)

            self.compute()
            train_infos = self.train()
            # print(episode_reward)
            if episode%1==0:
                train_infos['episode_reward'] = 0
                train_infos['episode_cost'] = 0
                train_infos['eval_over_order'] = 0
                train_infos['eval_dis_time'] = 0
                train_infos['safe_time'] = 0
                train_infos['default_time'] = 0
                episode_reward,episode_cost,eval_over_order,eval_dis_time,safe_time,default_time=self.eval(1)
                train_infos['episode_reward'] += episode_reward
                train_infos['episode_cost'] += episode_cost
                train_infos['eval_over_order'] += eval_over_order
                train_infos['eval_dis_time'] += eval_dis_time
                train_infos['safe_time'] += safe_time
                train_infos['default_time'] += default_time
                self.log_train(train_infos, episode)
                # if episode % 200 == 0:
                #     episode_rewards.append(episode_reward)
                #     episode_costs.append(episode_cost)
                #     c_total_92_empty.append(eval_92_empty)
                #     c_total_95_empty.append(eval_95_empty)
                #     c_total_derv_empty.append(eval_derv_empty)
                #     c_total_92_full.append(eval_92_full)
                #     c_total_95_full.append(eval_95_full)
                #     c_total_derv_full.append(eval_derv_full)
                # c_total_gas_single.append(eval_gas_single)
                # c_total_gas_double.append(eval_gas_double)
                # c_total_derv_single.append(eval_derv_single)
                # c_total_derv_double.append(eval_derv_double)
                # c_total_92_dis.append(eval_92_dis)
                # c_total_95_dis.append(eval_95_dis)
                # c_total_derv_dis.append(eval_derv_dis)

            # if episode % test_interval == 0:
            #     episode_rewards.append(rewards)

            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # np.save(save_path + '/episode_rewards{}', episode_rewards)
        # np.save(save_path + '/episode_costs{}', episode_costs)
        # # np.save(save_path + '/c_total_92_dis{}', c_total_92_dis)
        # # np.save(save_path + '/c_total_95_dis{}', c_total_95_dis)
        # # np.save(save_path + '/c_total_derv_dis{}', c_total_derv_dis)
        # np.save(save_path + '/c_total_92_empty{}',c_total_92_empty)
        # np.save(save_path + '/c_total_95_empty{}', c_total_95_empty)
        # np.save(save_path + '/c_total_derv_empty{}', c_total_derv_empty)
        # np.save(save_path + '/c_total_92_full{}', c_total_92_full)
        # np.save(save_path + '/c_total_95_full{}', c_total_95_full)
        # np.save(save_path + '/c_total_derv_full{}', c_total_derv_full)
        # np.save(save_path + '/c_total_gas_single {}', c_total_gas_single )
        # np.save(save_path + '/c_total_gas_double {}', c_total_gas_double)
        # np.save(save_path + '/c_total_derv_single {}', c_total_derv_single)
        # np.save(save_path + '/c_total_derv_double {}', c_total_derv_double)





    def collect(self, step):
        self.trainer.prep_rollout()

        (
            value,
            muti_actions,
            action_log_prob,oil_log_probs,
            rnn_states,
            rnn_states_critic,
            cost_pred, rnn_states_cost,action_logits_tensor
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            rnn_states_cost=np.concatenate(self.buffer.rnn_states_cost[step])
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(muti_actions), self.n_rollout_threads))  # [env_num, agent_num, action_dim]


         # [env_num, agent_num, 1]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        action_logits_tensor= np.array(np.split(_t2n(action_logits_tensor), self.n_rollout_threads))

        oil_log_probs = np.array(
            np.split(_t2n(oil_log_probs), self.n_rollout_threads)
        )
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        cost_preds = np.array(np.split(_t2n(cost_pred), self.n_rollout_threads))
        rnn_states_cost = np.array(np.split(_t2n(rnn_states_cost), self.n_rollout_threads))
        # available_actions_oil=np.array(
        #     np.split(_t2n(available_actions_oil), self.n_rollout_threads)
        # )
        # available_actions_warehouse = np.array(
        #     np.split(_t2n(available_actions_warehouse), self.n_rollout_threads)
        # )
        # rearrange action
        # actions_env = np.squeeze(np.eye(7)[actions], 2)
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]



        return (
            values,
            actions,
            action_log_probs,oil_log_probs,
            rnn_states,
            rnn_states_critic,
            cost_preds,
            rnn_states_cost,action_logits_tensor
        )


    @torch.no_grad()

    def insert(self, data):
        (
            obs,
            rewards,
            costs,
            dones,
            values,
            actions,
            action_log_probs,oil_log_probs,
            rnn_states,
            rnn_states_critic,
            cost_preds, rnn_states_cost,action_logits_tensor
        ) = data

        if dones.any():  # 检查是否有 `True` 值
            rnn_states[0] = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32
            )

            rnn_states_critic[0] = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32
            )
            rnn_states_cost[0] = np.zeros(
                (self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32
            )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        if dones.any():
            masks[0] = np.zeros(
                (1, self.n_rollout_threads, self.num_agents, 1),
                dtype=np.float32
            )

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,oil_log_probs,
            values,
            rewards,
            masks,
            costs, cost_preds, rnn_states_cost,action_logits_tensor
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_episode_costs = []
        c_over_order = []
        c_dis_time = []
        c_safe_time = []
        eval_default = []
        for i in range(total_num_steps):
            episode_rewards = 0
            episode_costs = 0
            eval_obs = self.envs.reset()
            eval_obs=np.array(eval_obs)
            eval_rnn_states = np.zeros(
                (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            for eval_step in range(self.episode_length):
                actions_92 = []
                actions_95 = []
                actions_derv = []
                action_total = []

                self.trainer.prep_rollout()
                eval_action, eval_rnn_states= self.trainer.policy.act(
                    eval_obs,
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True,
                )
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))

                for station_index in range(self.num_agents):
                    # ε-贪心策略选择动作
                    if eval_actions[0][station_index] == 0 or eval_actions[0][station_index] == 1 or eval_actions[0][station_index] == 2:
                        actions_92.append(0)
                        actions_95.append(0)
                        actions_derv.append(0)
                    elif eval_actions[0][station_index]== 3:
                                actions_92.append(1)
                                actions_95.append(0)
                                actions_derv.append(0)
                    elif eval_actions[0][station_index] == 4:
                                actions_92.append(0)
                                actions_95.append(1)
                                actions_derv.append(0)
                    elif eval_actions[0][station_index] == 5:
                                actions_92.append(0)
                                actions_95.append(0)
                                actions_derv.append(1)
                    elif eval_actions[0][station_index] == 6:
                            actions_92.append(2)
                            actions_95.append(0)
                            actions_derv.append(0)
                    elif eval_actions[0][station_index] == 7:
                            actions_92.append(0)
                            actions_95.append(2)
                            actions_derv.append(0)
                    elif eval_actions[0][station_index] == 8:
                            actions_92.append(0)
                            actions_95.append(0)
                            actions_derv.append(2)
                action_total.append(actions_92)
                action_total.append(actions_95)
                action_total.append(actions_derv)


                next_eval_obs, eval_rewards, eval_dones, eval_cost = self.envs.step(action_total)
                reward = np.array(eval_rewards)
                rewards_avg = np.mean(reward)
                cost = np.array(eval_cost)
                cost_sum = np.sum(cost)
                episode_rewards+=rewards_avg
                episode_costs+=cost_sum
                eval_obs=next_eval_obs

                # eval_rnn_states[eval_dones == True] = np.zeros(
                #     ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                #     dtype=np.float32,
                # )
                eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                # eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            c_over_order.append(self.envs.over_order)
            c_dis_time.append(self.envs.dis_time)
            c_safe_time.append(self.envs.safe_time)
            eval_episode_rewards.append(episode_rewards)
            eval_episode_costs.append(episode_costs)
            eval_default.append(self.envs.dc)
        eval_avg_reward = np.mean(eval_episode_rewards)
        eval_avg_cost = np.mean(eval_episode_costs)
        eval_over_order = np.mean(c_over_order) / 40
        eval_dis_time = np.mean(c_dis_time) / 40
        safe_time = np.mean(c_safe_time) / (40 * 288)
        eval_default= np.mean(eval_default)
        # eval_92_dis = np.mean(c_total_92_dis)
        # eval_95_dis = np.mean(c_total_95_dis)
        # eval_derv_dis = np.mean(c_total_derv_dis)
        # eval_gas_single = np.mean(c_total_gas_single)
        # eval_gas_double = np.mean(c_total_gas_double)
        # eval_derv_single = np.mean(c_total_derv_single)
        # eval_derv_double = np.mean(c_total_derv_double)
        print(eval_avg_reward)
        print(eval_avg_cost)
        return eval_avg_reward ,eval_avg_cost,eval_over_order,eval_dis_time,safe_time,eval_default
        # eval_env_infos = {}
        # eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        # eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        # print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        # self.log_env(eval_env_infos, total_num_steps)
    def eval_1(self, total_num_steps, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.policy.actor_oil.load_state_dict(checkpoint['actor_oil_state_dict'])
        new_rewards=[]
        new_cost=[]
        episode_rewards = 0
        episode_costs = 0
        eval_obs = self.envs.reset()
        eval_obs = np.array(eval_obs)
        eval_rnn_states = np.zeros(
            (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        for eval_step in range(self.episode_length):
            actions_92 = []
            actions_95 = []
            actions_derv = []
            action_total = []

            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                eval_obs,
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))

            for station_index in range(self.num_agents):
                # ε-贪心策略选择动作
                if eval_actions[0][station_index] == 0 or eval_actions[0][station_index] == 1 or eval_actions[0][
                    station_index] == 2:
                    actions_92.append(0)
                    actions_95.append(0)
                    actions_derv.append(0)
                elif eval_actions[0][station_index] == 3:
                    actions_92.append(1)
                    actions_95.append(0)
                    actions_derv.append(0)
                elif eval_actions[0][station_index] == 4:
                    actions_92.append(0)
                    actions_95.append(1)
                    actions_derv.append(0)
                elif eval_actions[0][station_index] == 5:
                    actions_92.append(0)
                    actions_95.append(0)
                    actions_derv.append(1)
                elif eval_actions[0][station_index] == 6:
                    actions_92.append(2)
                    actions_95.append(0)
                    actions_derv.append(0)
                elif eval_actions[0][station_index] == 7:
                    actions_92.append(0)
                    actions_95.append(2)
                    actions_derv.append(0)
                elif eval_actions[0][station_index] == 8:
                    actions_92.append(0)
                    actions_95.append(0)
                    actions_derv.append(2)
            action_total.append(actions_92)
            action_total.append(actions_95)
            action_total.append(actions_derv)

            next_eval_obs, eval_rewards, eval_dones, eval_cost = self.envs.step(action_total)
            reward = np.array(eval_rewards)
            rewards_avg = np.mean(reward)
            cost = np.array(eval_cost)
            cost_sum = np.sum(cost)

            episode_rewards += rewards_avg
            episode_costs += cost_sum
            new_rewards.append(episode_rewards)
            new_cost.append(episode_costs)
            eval_obs = next_eval_obs

            eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        print(episode_costs)
        return new_cost
    def eval_news(self, total_num_steps, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.policy.actor_oil.load_state_dict(checkpoint['actor_oil_state_dict'])
        eval_episode_rewards = []
        eval_episode_costs = []
        c_empty = []
        c_full = []
        c_wait = []
        c_over_order=[]
        c_dis_time=[]
        c_fillment = []
        c_safe_time=[]
        c_travel= []
        consum = 0
        for i in range(total_num_steps):

            oil_inital=0
            oil_final = 0
            episode_rewards = 0
            episode_costs = 0
            eval_obs = self.envs.reset()
            eval_obs = np.array(eval_obs)
            eval_rnn_states = np.zeros(
                (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            for eval_step in range(self.episode_length):
                actions_92 = []
                actions_95 = []
                actions_derv = []
                action_total = []

                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(
                    eval_obs,
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True,
                )
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))

                for station_index in range(self.num_agents):
                    # ε-贪心策略选择动作
                    if eval_actions[0][station_index] == 0 or eval_actions[0][station_index] == 1 or eval_actions[0][
                        station_index] == 2:
                        actions_92.append(0)
                        actions_95.append(0)
                        actions_derv.append(0)
                    elif eval_actions[0][station_index] == 3:
                        actions_92.append(1)
                        actions_95.append(0)
                        actions_derv.append(0)
                    elif eval_actions[0][station_index] == 4:
                        actions_92.append(0)
                        actions_95.append(1)
                        actions_derv.append(0)
                    elif eval_actions[0][station_index] == 5:
                        actions_92.append(0)
                        actions_95.append(0)
                        actions_derv.append(1)
                    elif eval_actions[0][station_index] == 6:
                        actions_92.append(2)
                        actions_95.append(0)
                        actions_derv.append(0)
                    elif eval_actions[0][station_index] == 7:
                        actions_92.append(0)
                        actions_95.append(2)
                        actions_derv.append(0)
                    elif eval_actions[0][station_index] == 8:
                        actions_92.append(0)
                        actions_95.append(0)
                        actions_derv.append(2)
                action_total.append(actions_92)
                action_total.append(actions_95)
                action_total.append(actions_derv)

                next_eval_obs, eval_rewards, eval_dones, eval_cost = self.envs.step(action_total)
                reward = np.array(eval_rewards)
                rewards_avg = np.mean(reward)
                cost = np.array(eval_cost)
                cost_sum = np.sum(cost)
                episode_rewards += rewards_avg
                episode_costs += cost_sum
                eval_obs = next_eval_obs

                # eval_rnn_states[eval_dones == True] = np.zeros(
                #     ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                #     dtype=np.float32,
                # )
                eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                # eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            c_empty.append(self.envs.empty_time)
            c_full.append(self.envs.full_time)
            c_over_order.append(self.envs.over_order)
            c_dis_time.append(self.envs.dis_time)
            c_safe_time.append(self.envs.safe_time)
            c_wait.append(self.envs.wait_time)
            eval_episode_rewards.append(episode_rewards)
            eval_episode_costs.append(episode_costs)

            if i == 0:
                for i in range(self.num_agents):
                    consum += self.envs.allstations[i].cosm_92 + self.envs.allstations[i].cosm_95 + \
                              self.envs.allstations[
                                  i].cosm_derv
                    oil_inital += (self.envs.allstations[i].initial_oil_92 + self.envs.allstations[i].initial_oil_95 +
                                   self.envs.allstations[i].initial_oil_derv)
                    oil_final += (self.envs.allstations[i].oil_92 + self.envs.allstations[i].oil_95 +
                                  self.envs.allstations[
                                      i].oil_derv)
            oil=(oil_inital+oil_final)/2
            a=self.envs.singe_dis * 2000 + self.envs.double_dis * 4000
            c_fillment.append((self.envs.singe_dis * 1000 + self.envs.double_dis * 2000) / (consum))

        eval_avg_reward = np.mean(eval_episode_rewards)
        eval_avg_cost = np.mean(eval_episode_costs)
        eval_empty = np.mean(c_empty)
        eval_full = np.mean(c_full)
        eval_over_order= np.mean(c_over_order)
        eval_dis_time = np.mean(c_dis_time)
        eval_fillment = np.mean(c_fillment)
        safe_time=np.mean(c_safe_time)
        eval_wait = np.mean(c_wait)
        travel=np.mean(c_travel)

        # print('平均等待时间：',eval_wait_time)
        print('周转率：', eval_fillment)
        #print(safe_time/(6*288))
        print(eval_wait/eval_dis_time)
        # print( travel)
        
        # eval_92_dis = np.mean(c_total_92_dis)
        # eval_95_dis = np.mean(c_total_95_dis)
        # eval_derv_dis = np.mean(c_total_derv_dis)
        # eval_gas_single = np.mean(c_total_gas_single)
        # eval_gas_double = np.mean(c_total_gas_double)
        # eval_derv_single = np.mean(c_total_derv_single)
        # eval_derv_double = np.mean(c_total_derv_double)
        print(eval_avg_reward)
        print(eval_avg_cost)
        # print(eval_empty)
        # print(eval_full)
        print(eval_over_order/40)
        print(eval_dis_time/40)
        return eval_avg_reward, eval_avg_cost,eval_empty,eval_full
        # eval_env_infos = {}
        # eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        # eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        # print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        # self.log_env(eval_env_infos, total_num_steps)
    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
