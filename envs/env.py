import io
import sys
from typing import Optional

import gymnasium as gym

from .inout import load_genius_domain_from_folder
from .rl_negotiator import *
from .observer import *
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *

PENALTY = -1.


class NaiveEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array']}

    def __init__(self, domain: str, is_first: bool = False, test: bool = False, scale='small'):
        super().__init__()
        self.test = test
        self.session_start = False
        # ドメイン読み込み
        scenario = load_genius_domain_from_folder('domain/' + domain)
        self.domain = domain
        self.issues = scenario.issues
        self.util1 = scenario.ufuns[0].scale_max(1.0)
        self.util2 = scenario.ufuns[1].scale_max(1.0)

        self.my_agent: Optional[RLNegotiator] = None
        self.session: Optional[MySAOMechanism] = None

        # 設定読み込み
        self.is_first = is_first
        self.is_first_turn = is_first
        if self.is_first:
            self.my_util = self.util1
            self.opp_util = self.util2
        else:
            self.opp_util = self.util1
            self.my_util = self.util2

        # 強化学習関連
        self.state = None
        self.action = None
        self.observation = None

        self.observer = EmbeddedObserveHistory(domain, issues=scenario.issues, n_turn=80, scale=scale)
        self.all_bids = self.get_all_bids()
        self.observation_space = self.observer.observation_space
        self.action_space = gym.spaces.Discrete(len(self.all_bids))
        self.reward_range = [PENALTY, 1.0]
        self.seed()
        self.reset(options={"opponent":"Boulware"})

    def reset(self, seed=None, options=None):
        self.opponent = options["opponent"]
        self.is_first_turn = False
        # セッション，エージェントの作成
        if self.session is not None:
            self.session.reset()
            self.session_start = False
        self.session = MySAOMechanism(issues=self.issues, n_steps=80, avoid_ultimatum=False)
        self.my_agent = RLNegotiator()
        opponent = self.get_opponent(add_noise=True)

        # セッションにエージェントの追加
        if self.is_first_turn:
            self.session.add(self.my_agent, ufun=self.my_util)
            self.session.add(opponent, ufun=self.opp_util)
            self.state = None
        else:
            self.session.add(opponent, ufun=self.opp_util)
            self.session.add(self.my_agent, ufun=self.my_util)
            # 後攻だったら相手に1回提案させる
            self.state = self.session.step().asdict()

        self.observer.reset()
        self.observation = self.observer(self.state)
        return self.observation, {}

    def step(self, action: int):
        self.action = self.all_bids[action]
        self.my_agent.set_next_bid(self.action)
        self.state = self.session.step().asdict()
        # 状態を更新
        self.observation = self.observer(self.state)
        if self.state['agreement'] is not None:  # 合意していたら
            return self.observation, self.get_reward(), True, {}
        if self.state['timedout'] or self.state['broken']:
            return self.observation, self.get_reward(), True, {}
        return self.observation, self.get_reward(), False, False, {}

    def render(self, mode='human', close=False):
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        if not self.state['running']:
            self.session.plot()
        return outfile

    def get_opponent(self, add_noise=False):
        if self.opponent == 'Boulware':
            opponent = TimeBasedNegotiator(name='Boulware', aspiration_type=4.0, add_noise=add_noise)
        elif self.opponent == 'Linear':
            opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
        elif self.opponent == 'Conceder':
            opponent = TimeBasedNegotiator(name='Conceder', aspiration_type=0.2, add_noise=add_noise)
        elif self.opponent == 'TitForTat1':
            opponent = AverageTitForTatNegotiator(name='TitForTat1', gamma=1, add_noise=add_noise)
        elif self.opponent == 'TitForTat2':
            opponent = AverageTitForTatNegotiator(name='TitForTat2', gamma=2, add_noise=add_noise)
        elif self.opponent == 'AgentK':
            opponent = AgentK(add_noise=add_noise)
        elif self.opponent == 'HardHeaded':
            opponent = HardHeaded(add_noise=add_noise)
        elif self.opponent == 'Atlas3':
            opponent = Atlas3(add_noise=add_noise)
        elif self.opponent == 'AgentGG':
            opponent = AgentGG(add_noise=add_noise)
        else:
            opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
        return opponent

    def close(self):
        del self.issues
        del self.util1
        del self.util2
        del self.my_util
        del self.opp_util
        del self.my_agent
        del self.all_bids
        del self.session

    def seed(self, seed=None):
        pass

    def get_reward(self):
        if self.state['timedout'] or self.state['broken']:
            assert not self.state['broken'], "session broken"
            if not self.test:
                return PENALTY
            else:
                return 0
        elif self.state['agreement'] is not None:
            offer = tuple(v for _, v in self.state['agreement'].items())
            return self.my_util(offer)
        else:
            return 0

    def get_all_bids(self):
        session = MySAOMechanism(issues=self.issues, n_steps=80, avoid_ultimatum=False)
        agent = RLNegotiator()
        session.add(agent, ufun=self.util1 if self.is_first else self.util2)
        return agent.all_bids


class AOPEnv(NaiveEnv):
    def __init__(self, domain='party', is_first=False, observer=None, test=False, render_mode=None, scale='small'):
        super().__init__(domain, is_first, test, scale)
        self.action_space = gym.spaces.Discrete(len(self.all_bids))

    def step(self, action: int):
        # インデックス0はAcceptに対応
        idx = action[0]
        if idx == 0:
            self.my_agent.accept_offer = True
        self.action = self.all_bids[idx-1]
        # インデックスからbidに変換
        self.action = {i.name: v for i, v in zip(self.issues, self.action)}
        self.my_agent.set_next_bid(self.action)
        #print(self.state['step'])
        # 自分と相手の1ターン更新
        for _ in range(2):
            self.state = self.session.step().asdict()
            # 状態を更新
            #print(self.session._current_proposer.name)
            self.observation = self.observer(self.state)
            #print(self.session._current_proposer)
            if self.state['agreement'] is not None:  # 合意していたら
                return self.observation, self.get_reward(), True, False, {"state":{"step":self.state['step'],"agreement":self.state["agreement"],"my_util":self.get_reward(),"opp_util":self.opp_util(tuple(v for _, v in self.state['agreement'].items())),"last_neg":self.state['last_negotiator']}}
            if self.state['timedout'] or self.state['broken']:
                return self.observation, self.get_reward(), True, False, {"state":[]}
        return self.observation, self.get_reward(), False, False, {"state":[]}

