import os
import numpy as np
import torch
from utils.shared_buffer import SharedReplayBuffer
import json
def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters

        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.recurrent_N = self.all_args.recurrent_N
        self.entropy_coef=self.all_args.entropy_coef
        self.entropy_coef_oil=self.all_args.entropy_coef_oil
        self.lagrangian_coef_rate=self.all_args.lagrangian_coef_rate
        # interval
        # self.save_interval = self.all_args.save_interval
        # self.use_eval = self.all_args.use_eval
        # self.eval_interval = self.all_args.eval_interval
        # self.log_interval = self.all_args.log_interval

        # dir
        # self.model_dir = self.all_args.model_dir
        #
        # self.run_dir = config["run_dir"]
        # self.log_dir = str(self.run_dir / 'logs')
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)

        # self.save_dir = str(self.run_dir / 'models')
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
        share_observation_space = self.envs.share_observation_space[0]


        # policy network
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                           share_observation_space,
                            self.envs.action_space[0],
                             self.envs.action_space2[0],
                             self.num_agents,
                            device = self.device)

        # if self.model_dir is not None:
        #     self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])


    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

        next_costs = self.trainer.policy.get_cost_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_cost[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        next_costs = np.array(np.split(_t2n(next_costs), self.n_rollout_threads))
        self.buffer.compute_cost_returns(next_costs, self.trainer.cost_normalizer)


    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()


        action_dim = 3


        num_agents = self.envs.agent_num
        factor = np.ones((self.episode_length, self.n_rollout_threads, num_agents, action_dim), dtype=np.float32)
        # factor_warehouse = np.ones((self.episode_length, self.n_rollout_threads, num_agents, action_warehouse_dim), dtype=np.float32)
        self.buffer.update_factor(factor)

        # self.buffer.update_factor_warehouse(factor_warehouse)
        # old_actions_logprob, _ = self.trainer.policy.actor.evaluate_actions(
        #     self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
        #     self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
        #     self.buffer.actions.reshape(-1, *self.buffer.actions.shape[3:]),
        #     self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
        #     self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:]))
        #
        # old_oil_logprob, _ = self.trainer.policy.actor_oil.evaluate_actions(
        #     self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
        #     self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
        #     self.buffer.oil_type.reshape(-1, *self.buffer.oil_type.shape[3:]),
        #     self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
        #     self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:])
        #     )
        #
        # old_warehouse_type_logprob, _ = self.trainer.policy.actor_warehouse.evaluate_actions(
        #     self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
        #     self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
        #     self.buffer.warehouse_type.reshape(-1, *self.buffer.warehouse_type.shape[3:]),
        #     self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
        # self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:])
        #     )


        train_infos = self.trainer.train(self.buffer)
        # fwq
        # random update order

        # new_actions_logprob, _ = self.trainer.policy.actor.evaluate_actions(
        #     self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
        #     self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
        #     self.buffer.actions.reshape(-1, *self.buffer.actions.shape[3:]),
        #     self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
        #     self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:]))
        # new_oil_logprob, _ = self.trainer.policy.actor_oil.evaluate_actions(
        #     self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
        #     self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
        #     self.buffer.oil_type.reshape(-1, *self.buffer.oil_type.shape[3:]),
        #     self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
        #     self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:]))
        # new_warehouse_logprob, _ = self.trainer.policy.actor_warehouse.evaluate_actions(
        #     self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
        #     self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
        #     self.buffer.warehouse_type.reshape(-1, *self.buffer.warehouse_type.shape[3:]),
        #     self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
        #     self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:]))
        # factor = factor * _t2n(torch.exp(new_actions_logprob - old_actions_logprob).reshape(-1,
        #                                                                                     self.n_rollout_threads,
        #                                                                                     action_dim))
        # factor_oil = factor_oil * _t2n(torch.exp(new_oil_logprob - old_oil_logprob).reshape(-1,
        #                                                                                     self.n_rollout_threads,
        #                                                                                     action_oil_dim))
        # factor_warehouse = factor_warehouse * _t2n(torch.exp(new_warehouse_logprob -old_warehouse_type_logprob).reshape(-1,
        #                                                                                     self.n_rollout_threads,
        #                                                                                     action_warehouse_dim))

        self.buffer.after_update()


        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps,log_file="HAC-MAPPO.json"):
        # 初始化文件为空列表（如果不存在）
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                json.dump([], f)

        # 安全读取现有日志数据
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []  # 文件损坏或空文件时，初始化为空

        # 将 train_infos 中的 Tensor 转换为 Python 类型
        train_infos_serializable = {k: (v.item() if hasattr(v, "item") else v) for k, v in train_infos.items()}

        # 新增一条日志
        log_entry = {"total_num_steps": total_num_steps}
        log_entry.update(train_infos_serializable)  # 合并 total_num_steps 和 train_infos
        logs.append(log_entry)

        # 将更新后的日志写回文件
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)
    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
