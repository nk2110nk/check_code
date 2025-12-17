import gymnasium as gym
import numpy as np
from abc import abstractmethod, ABCMeta
from negmas import SAOState

from embedding_model import MyEmbedding

import os
# tokenizer の dead lock warning を回避
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class AbstractObserve(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.observation = None
        self.init_observation = None
        self.observation_space = None

    def reset(self):
        self.observation[...] = self.init_observation

    def __call__(self, state):
        if state is None:
            return self.init_observation
        else:
            return self.observe(state)

    @abstractmethod
    def observe(self, state):
        pass


class EmbeddedObserveHistory(AbstractObserve):
    def __init__(self, domain, issues=None, n_turn=80, scale='small'):
        super().__init__()
        self.domain = domain
        self.issues = issues

        # 埋め込み設定
        self.emb_dim = 256
        self.embedding_model = MyEmbedding(scale=scale)
        # 論点名
        self.i_names = [i.name for i in self.issues]
        # 論点名の埋込ベクトルロード
        self.i_emb = self.embedding_model.list_embedding(self.i_names, save_path='./embeddings/openai/i_embs/'+scale+'/'+self.domain+'.json')
        # 各論点における選択肢の埋め込みベクトルをロード
        self.embedding_model.make_offer_embedding(save_path='./embeddings/openai/'+scale+'/'+self.domain+'.json')

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=((n_turn+1) * 2, len(self.i_emb), self.emb_dim), dtype=np.float32)
        self.init_observation = np.zeros(((n_turn+1) * 2, len(self.i_emb), self.emb_dim), dtype=np.float32)
        self.observation = self.init_observation
        self.padded = False
    
    def observe(self, state: SAOState):
        offers = state['new_offers']
        for neg, offer in offers:
            embed_offer = self.embedding_model.offer_embedding(offer.values())
            # 埋め込みベクトル作成
            embed_offer = embed_offer + self.i_emb
            if 'RLAgent' in neg:
                self.observation[2*(state['step']-1),:,:] = embed_offer
            else:
                self.observation[2*state['step']-1,:,:] = embed_offer
        if self.padded == False:
            self.observation[2*state['step']:,:,:] = 1e-6
            self.padded = True
        return self.observation
