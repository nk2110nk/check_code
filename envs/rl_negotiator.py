import torch

from negmas.sao import SAONegotiator
from typing import Optional
from negmas.common import *
from negmas.outcomes import Outcome
from negmas.sao import ResponseType
from sao.opponent_model import *
from envs.observer import *
from ppo_scratch import PPO


class RLNegotiator(SAONegotiator):
    def __init__(self, name='RLAgent', accept_offer=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outcomes = None
        self.next_bid = None
        self.last_bid = None
        self.accept_offer = accept_offer

    def on_ufun_changed(self):
        super().on_ufun_changed()
        self.next_bid = None
        self.last_bid = None
        self.n_outcomes = len(self._nmi.discrete_outcomes())

    @property
    def all_bids(self):
        return self._nmi.discrete_outcomes()

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        if self.accept_offer:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        self.last_bid = self.next_bid
        return self.next_bid

    def set_next_bid(self, next_bid) -> None:
        self.next_bid = next_bid


class TestRLNegotiator(RLNegotiator):
    def __init__(self, domain, issue, opponent, path, deterministic=False, mode='issue', accept_offer=False, decoder_only=False, n_turns=80, scale='small', decoder_num=1, action_space=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode    # issue, venas, dqn
        self.deterministic = deterministic
        self.observer = EmbeddedObserveHistory(domain=issue, issues=domain, scale=scale)
        self.domain = domain
        self.actions = None
        self.states = None
        self.values = None
        self.observation = None
        self.accept_offer = accept_offer

        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.model = PPO(n_envs=1, issue=issue, agents=opponent, device="cpu", is_acceptable=accept_offer, scale=scale, decoder_num=decoder_num, action_space=action_space, obs_space=self.observer.observation_space)
        self.model.model.load_state_dict(checkpoint['model'])
        self.model.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.model.feature_extractor.decoder_only = decoder_only

    def respond(self, state: MechanismState, offer: "Outcome", source=None) -> "ResponseType":
        # 行動選択
        self.actions, self.states, self.values = self.model.predict(self.observation, state=self.states, deterministic=self.deterministic)
        if isinstance(self.model.env.action_space, gym.spaces.Discrete):
            if self.actions[0] == 0:
                return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.actions is None:
            self.observation = self.observer(None)
            self.actions, self.states, self.values = self.model.predict(self.observation, state=self.states, deterministic=self.deterministic)
        if isinstance(self.model.env.action_space, gym.spaces.Discrete):
            action = self.all_bids[self.actions[0]-1]
        else:
            # インデックスをbidに変換
            return {i.name: i.values[v] for i, v in zip(self.domain, self.actions[0])}
        return {i.name: v for i, v in zip(self.domain, action)}


