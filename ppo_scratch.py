from collections import deque
import os
from typing import Union, Optional
import gymnasium as gym
from gymnasium import register, spaces
from stable_baselines3.common.vec_env import (
    VecEnv,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.monitor import Monitor

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs.observer import EmbeddedObserveHistory
from envs.dummy_vec_env import DummyVecEnv
from policy import Transformer_Policy, RolloutBuffer

ENV_NAME = 'AOPEnv-{}-v0'

def register_neg_env(issue, observer, scale):
    env_name = ENV_NAME.format(issue)
    register(
        id=env_name,
        entry_point='envs.env:AOPEnv',
        kwargs={'domain': issue, 'is_first': True, 'observer': observer, 'scale': scale},
    )
    return env_name

def make_vec_env(
    env_id,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class = None,
    env_kwargs = None,
    vec_env_cls = None,
    vec_env_kwargs = None,
    monitor_kwargs = None,
    wrapper_kwargs = None,
) -> VecEnv:
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int):
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

class PPO():
    def __init__(
        self,
        issue: Optional[Union[str, list[str]]] = None,
        agents: Optional[Union[str, list[str]]] = 'Linear',
        n_envs: int = 4,
        learning_rate: float = 3e-4,
        n_rollout_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        obs_space: Optional[tuple[int,...]] = None,
        action_space: Optional[tuple[int,...]] = None,
        device: torch.device = "cuda:1",
        model: Optional[nn.Module] = None,
        random_train: bool = False, 
        is_acceptable: bool = False,
        decoder_only: bool = False,
        decoder_num: int = 1, 
        scale: str = 'small', 
    ) -> None:
        
        self.issue = issue
        self.agents = agents
        self.n_envs = n_envs
        # 複数環境リスト作成
        self.env_list: list[VecEnv] = self.make_env_list(scale)
        self.env = self.env_list[0]

        self.n_timesteps = 0
        self.learning_rate = learning_rate
        self.n_rollout_steps = n_rollout_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.obs_space = obs_space if obs_space is not None else self.env.observation_space
        self.action_space = action_space if action_space is not None else self.env.action_space


        self.device = self.get_device(device)
        # モデル定義
        self.model = model if model is not None else Transformer_Policy(obs_space=self.obs_space, action_space=self.action_space, domain=self.env.envs[0].unwrapped.domain, issues=self.env.envs[0].unwrapped.issues, device=self.device, is_acceptable=is_acceptable, decoder_only=decoder_only, scale=scale, decoder_num=decoder_num)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

        self.episode_frame_numbers = None
        self.episode_rewards = None
        self.vec_env_reward = None
        self.global_step = 0
        self.rollout_buffer_list = [RolloutBuffer(
            buffer_size=self.n_rollout_steps,
            n_envs=self.n_envs,
            obs_space=e.observation_space,
            action_space=e.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        ) for e in self.env_list]
        self.ep_info_buffer = deque(maxlen=100)

        self._last_obs = None
        self._last_episode_starts = None
        self._logger = None
        self.save_log = True
        # self.save_log = False
        self.random_train = random_train
    
    def make_env_list(self, scale):
        if isinstance(self.issue, str):
            self.issue = [self.issue]        
        observer = EmbeddedObserveHistory
        env_list = []
        
        for i in self.issue:
            env_name = register_neg_env(i, observer, scale)
            env = make_vec_env(env_name, n_envs=self.n_envs, wrapper_class=Monitor,wrapper_kwargs={"info_keywords":tuple(["state",])})
            env_list.append(env)
        return env_list
    
    def on_rollout_start(self):
        self.episode_frame_numbers.clear()
        self.episode_rewards.clear()
    
    def collect_rollouts(
        self, 
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        agent_id: int = None, 
    ):
        self.model.eval()
        n_steps = 0
        rollout_buffer.reset()
        self.on_rollout_start()
        is_first_offer = [False] * self.env.num_envs
        agent_idx = [agent_id] * self.env.num_envs

        with tqdm(total=n_rollout_steps) as pbar:
            while n_steps < n_rollout_steps:
                with torch.no_grad():
                    for i in range(self.env.num_envs):
                        if self.env.envs[i].state is None and self.env.envs[i].is_first_turn:
                            is_first_offer[i] = True
                    # 行動選択
                    actions, values, log_probs = self.model.sample(self._last_obs, is_first_offer)
                    is_first_offer = [False] * self.env.num_envs
                actions = actions.cpu().numpy()
                if isinstance(self.action_space, spaces.Box):
                    actions = np.clip(actions, self.action_space.low, self.action_space.high)
                # 1ステップ進める
                new_obs, rewards, dones, infos = self.env.step(actions)

                self.n_timesteps += self.n_envs
                for idx, info in enumerate(infos):
                    maybe_ep_info = info.get("episode")
                    if maybe_ep_info is not None:
                        self.ep_info_buffer.extend([maybe_ep_info])
                n_steps += 1

                # 1エピソード終了
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        if self.random_train:
                            agent_idx[idx] += np.random.randint(len(self.agents))
                        else:
                            agent_idx[idx] += 1
                        self.env._options[idx] = self.agents[agent_idx[idx]%len(self.agents)]
                        terminal_obs = torch.as_tensor(infos[idx]["terminal_observation"])
                        with torch.no_grad():
                            terminal_value = self.model.predict_values(terminal_obs)  # type: ignore[arg-type]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                )
                self._last_obs = new_obs
                self._last_episode_starts = dones
                pbar.update(1)
        
        with torch.no_grad():
            values = self.model.predict_values(new_obs).to(self.device)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # Logger


    def train(
        self,
        total_timesteps:int = 1_000_000,
        save_path = None,
    ):
        # torch.autograd.set_detect_anomaly(True)
        self.episode_frame_numbers = []
        self.episode_rewards = []
        self.vec_env_reward = [0 for _ in range(self.n_envs)]
        # self._last_obs = self.env.reset()
        self._last_episode_starts = torch.ones((self.n_envs,), dtype=bool)
        self._logger = SummaryWriter(log_dir=save_path)
        self.save_path = save_path

        self.n_timesteps = 0
        self.global_step = 0
        self.stop_training = False

        iteration = 0
        n_iteration = total_timesteps // (self.n_envs*self.n_rollout_steps) + 1
        idxes_i = np.array([_ for _ in range(len(self.issue))]*n_iteration)
        np.random.shuffle(idxes_i)
        idxes_a = np.array([_ for _ in range(len(self.agents))]*n_iteration)
        np.random.shuffle(idxes_a)
        with tqdm(total=total_timesteps) as pbar:
            while self.n_timesteps< total_timesteps:
                # 環境の更新
                if self.stop_training:
                    return self
                if self.random_train:
                    agent_id = idxes_a[iteration]
                    self.env = self.env_list[idxes_i[iteration]]
                    self.rollout_buffer = self.rollout_buffer_list[idxes_i[iteration]]
                    self.env.set_options({"opponent":self.agents[agent_id]})
                    self._last_obs = self.env.reset()
                else:
                    agent_id = iteration%len(self.agents)
                    self.env = self.env_list[iteration%len(self.issue)]
                    self.rollout_buffer = self.rollout_buffer_list[iteration%len(self.issue)]
                    self.env.set_options({"opponent":self.agents[agent_id]})
                    self._last_obs = self.env.reset()
                self.model.action_dim = int(self.env.action_space.n)
                self.model.embedding_model.make_offer_embedding(save_path='./embeddings/openai/small/'+self.env.envs[0].unwrapped.domain+'.json')
                self.model.feature_extractor.memory = torch.as_tensor(self.model.embedding_model.outcomes_embedding(self.env.envs[0].unwrapped.issues, self.batch_size, './embeddings/openai/i_embs/small/'+self.env.envs[0].unwrapped.domain+'.json')).to(self.device)

                # ロールアウト（シミュレーション）実行
                self.collect_rollouts(self.rollout_buffer, n_rollout_steps=self.n_rollout_steps, agent_id=agent_id)

                pbar.update(self.n_envs*self.n_rollout_steps)
                iteration += 1

                # 収集したデータから勾配更新
                self.train_loop()
                self.rollout_buffer.empty_cache()

                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(checkpoint, save_path+'/checkpoint.pt'.format())
        return self
    
    def train_loop(self):
        self.model.train()
        for epoch in tqdm(range(self.n_epochs),total=self.n_epochs):
            rollouts = self.rollout_buffer.get(self.batch_size)
            for rollout_data in rollouts:
                actions = rollout_data.actions
                values, log_prob, entropy, probs = self.model.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ratio between old and new policy, should be one at the 1st iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # Clipped Surrogate Objective
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)

                # Entropy loss favor exploration
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Logs
                if self.save_log:
                    self._logger.add_scalar("train/policy_loss", policy_loss.item(), self.global_step)
                    self._logger.add_scalar("train/value_loss", value_loss.item(), self.global_step)
                    self._logger.add_scalar("train/entropy_loss", entropy_loss.item(), self.global_step)
                    self._logger.add_scalar("train/loss", loss.item(), self.global_step)
                    self._logger.add_scalar("train/values", torch.mean(values).item(), self.global_step)
                    self._logger.add_scalar("train/log_prob", torch.mean(log_prob).item(), self.global_step)
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        self._logger.add_scalar("rollout/ep_rew_mean", float(np.mean([ep_info["r"] for ep_info in self.ep_info_buffer])), self.global_step)
                        self._logger.add_scalar("rollout/ep_len_mean", float(np.mean([ep_info["l"] for ep_info in self.ep_info_buffer])), self.global_step)

                self.global_step += 1
    
    
    @staticmethod
    def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
        """
        Retrieve PyTorch device.
        It checks that the requested device is available first.
        For now, it supports only cpu and cuda.
        By default, it tries to use the gpu.

        :param device: One for 'auto', 'cuda', 'cpu'
        :return: Supported Pytorch device
        """
        # Cuda by default
        if device == "auto":
            device = "cuda"
        # Force conversion to torch.device
        device = torch.device(device)

        # Cuda not available
        if device.type == torch.device("cuda").type and not torch.cuda.is_available():
            return torch.device("cpu")

        return device
    
    def predict(self, observation, state, episode_start=None, deterministic=False):
        if np.all(observation == np.zeros(observation.shape)):
            is_first_offer = [True]
        else:
            is_first_offer = [False]
        return self.model.predict(observation, state, deterministic, is_first_offer=is_first_offer)

