import copy
from typing import NamedTuple
import gymnasium as gym
from gymnasium import spaces

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from NegTransformer import NegTransformer, Config
from embedding_model import MyEmbedding
torch.set_printoptions(edgeitems=1000)

def get_action_dim(action_space) -> int:
    if isinstance(action_space, tuple):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return sum(list(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")

def get_obs_shape(observation_space,):
    if isinstance(observation_space, tuple):
        return observation_space
    elif isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_space = None,
        action_space = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.observation_space = obs_space
        self.action_space = action_space
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, device=self.device)
        x = self.feature_extractor(x)
        policy = self.fc_p(self.actor_extract_features(x))
        value = self.fc_v(self.value_extract_features(x))
        return policy, value
    
    def predict_values(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, device=self.device)
        x = self.feature_extractor(x)
        value = self.fc_v(self.value_extract_features(x))
        return value
    
    # @staticmethod
    def log_prob(self, value, logits):
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        log_prob = log_pmf.gather(-1, value).squeeze(-1)
        return log_prob

    # @staticmethod
    def entropy(self, logits):
        min_real = torch.finfo(logits.dtype).min
        l = torch.clamp(logits, min=min_real)
        probs = F.softmax(l, dim=-1)
        p_log_p = l * probs
        return -p_log_p.sum(-1)
    
    def mode(self, obs, is_first_offer):
        logits, values = self.forward(obs)
        if isinstance(self.action_space, gym.spaces.Discrete):
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            probs = F.softmax(logits, dim=-1)
            actions = torch.argmax(probs, dim=1)
            return actions
        splits = torch.split(logits, list(self.action_space.nvec), dim=1)
        actions = []
        for split in splits:
            logit = split - split.logsumexp(dim=-1, keepdim=True)
            probs = F.softmax(logit, dim=-1)
            action = torch.argmax(probs, dim=1)
            actions.append(action)
        return torch.stack(actions, dim=1)
    
    def sample(self, obs, is_first_offer):
        logits, values = self.forward(obs)
        if isinstance(self.action_space, gym.spaces.Discrete):
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1, True)
            return actions, values, self.log_prob(actions, logits)
        splits = torch.split(logits, list(self.action_space.nvec), dim=1)
        actions = []
        log_probs = []
        for split in splits:
            logit = split - split.logsumexp(dim=-1, keepdim=True)
            probs = F.softmax(logit, dim=-1)
            probs_2d = probs.reshape(-1, logit.size()[-1])
            sample_2d = torch.multinomial(probs_2d, torch.Size().numel(), True).T
            action = sample_2d.reshape(torch.Size()+logit.size()[:-1]+torch.Size())

            value = action.long().unsqueeze(-1)
            log_prob = self.log_prob(value, logit)

            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.stack(actions, dim=1).reshape((-1, *self.action_space.shape))
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        if isinstance(self.action_space, gym.spaces.Discrete):
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            log_prob = self.log_prob(actions, logits)
            entropy = self.entropy(logits)
            return values, log_prob, entropy
        splits = torch.split(logits, list(self.action_space.nvec), dim=1)
        log_probs = []
        entropy = []
        for split, action in zip(splits, torch.unbind(actions, dim=1)):
            logit = split - split.logsumexp(dim=-1, keepdim=True)
            value = action.long().unsqueeze(-1)
            log_prob = self.log_prob(value, logit)
            log_probs.append(log_prob)

            e = self.entropy(logit)
            entropy.append(e)

        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        entropy = torch.stack(entropy, dim=1).sum(dim=1)
        return values, log_probs, entropy
    
    def predict(self, observation, state, deterministic, vectorized=False, is_first_offer=[False]):
        vectorized_env = vectorized
        obs_tensor  = torch.as_tensor(np.array(observation).reshape(-1, *self.observation_space.shape))

        with torch.no_grad():
            actions, values = self._predict(obs_tensor, deterministic=deterministic, is_first_offer=is_first_offer)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        # Remove batch dimension if needed
        # if not vectorized_env:
        #     assert isinstance(actions, np.ndarray)
        #     actions = actions.squeeze(axis=0)

        return actions, state, values.cpu().numpy().reshape((-1,))
    
    def _predict(self, observation, deterministic, is_first_offer):
        if deterministic:
            return self.mode(observation, is_first_offer)
        actions, values, _ = self.sample(observation, is_first_offer)
        return actions, values

class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class RolloutBuffer:
    def __init__(
        self, 
        buffer_size, 
        n_envs, 
        obs_space, 
        action_space, 
        device,
        gamma = 0.99,
        gae_lambda = 1
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.observation_space = obs_space
        self.obs_dim = get_obs_shape(obs_space)
        if isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = len(action_space.nvec)
        else:
            self.action_dim = get_action_dim(action_space)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.generator_ready = False
    
    def empty_cache(self):
        del self.observations
        del self.actions
        del self.rewards
        del self.returns
        del self.episode_starts
        del self.values
        del self.log_probs
        del self.advantages
        torch.cuda.empty_cache()

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_dim))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.cpu().flatten()
        self.log_probs[self.pos] = log_prob.cpu()
        self.pos += 1

    # GAEを計算する
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray):
        last_values = last_values.cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            '''delta = r_t + gamma * V(s_{t+1}) - V(s_t)'''
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            '''A = delta(t) + gamma * lamda * delta(t+1)'''
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    @staticmethod
    def swap_and_flatten(arr):
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size):
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            self.observations = self.swap_and_flatten(self.observations)
            self.actions = self.swap_and_flatten(self.actions)
            self.values = self.swap_and_flatten(self.values)
            self.log_probs = self.swap_and_flatten(self.log_probs)
            self.advantages = self.swap_and_flatten(self.advantages)
            self.returns = self.swap_and_flatten(self.returns)
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def to_torch(self, array):
        return torch.as_tensor(array, device=self.device)

    def _get_samples(
        self,
        batch_inds
    ):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

class Transformer_Policy(PolicyNet):
    def __init__(self, obs_space=None, action_space=None, device: torch.device = None, config: Config = None, domain=None, issues=None, batch_size=4, is_acceptable=True, decoder_only=False, scale='small', decoder_num=1) -> None:
        super().__init__(obs_space, action_space, device)
        self.action_dim = int(action_space.n)
        print(self.action_dim)
        self.n_embd = get_obs_shape(obs_space)[-1]
        self.embedding_model = MyEmbedding(scale=scale)
        # 埋め込みベクトルロード
        self.embedding_model.make_offer_embedding(save_path='./embeddings/openai/'+scale+'/'+domain+'.json')
        all_bids_embd = self.embedding_model.outcomes_embedding(issues, batch_size, save_path='./embeddings/openai/i_embs/'+scale+'/'+domain+'.json')
        if config is None:
            config = Config(action_dim=self.action_dim, block_size=256, n_embd=self.n_embd, decoder_num=decoder_num)
        self.feature_extractor = NegTransformer(config=config, memory=torch.as_tensor(all_bids_embd, device=self.device, dtype=torch.float32), device=device, decoder_only=decoder_only)
        self.fc_p = nn.Linear(config.n_embd, 16384)
        self.fc_v = nn.Linear(config.n_embd, 1)
        self.is_acceptable = is_acceptable
    
    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        x = self.feature_extractor(x)
        policy = self.fc_p(x[:, 0, 0, :])[:,:self.action_dim+1]
        value = self.fc_v(x[:, 0, 0, :])
        return policy, value
    
    def mode(self, obs, is_first_offer):
        logits, values = self.forward(obs)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        for i in range(len(is_first_offer)):
            if is_first_offer[i]:
                logits[i,0] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=1)
        return actions, values
    
    def sample(self, obs, is_first_offer):
        logits, values = self.forward(obs)
        a = copy.deepcopy(logits)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        if self.is_acceptable:
            for i in range(len(is_first_offer)):
                if is_first_offer[i]:
                    logits[i,0] = float('-inf')
        else:
            logits[:,0] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1, True)
        log_prob = self.log_prob(actions, logits)
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logits[:,0] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        log_prob = self.log_prob(actions, logits)
        entropy = self.entropy(logits)
        return values, log_prob, entropy, probs
    
    def predict_values(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        x = self.feature_extractor(x)
        value: torch.Tensor = self.fc_v(x[:, 0, 0, :])
        return value