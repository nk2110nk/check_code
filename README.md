# Transformer-based Negotiation AI Agent
## Overview
implementation code for transformer-based negotiation AI agent architecture
## Instructions

### Requirements

```
dill==0.3.5.1
gym==0.21.0
gymnasium==0.29.1
matplotlib==3.5.3
negmas==0.9.7
numpy==1.23.2
numpy==1.23.1
pandas==1.4.3
stable_baselines3==2.2.1
sympy==1.12
torch==2.1.2
tqdm==4.64.0
transformers==4.35.2
```


### Running experiments

- Example command for training:
```
python3 ./train.py -a Boulware Conceder Linear TitForTat1 TitForTat2 -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
```


- Example command for testing with a pretrained model:
```
python3 ./test_negotiator.py -a Boulware Conceder Linear TitForTat1 TitForTat2 -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A -m ./results/pretrained_model
```

- `-a` and `-i` arguments specify agents and issues for training or testing

## 各ファイル・クラスの説明
### ./train.py
- 学習実行
### ./test_negotiator.py
- テスト実行
### ./ppo_scratch.py
- 学習アルゴリズム実装部
- `PPO` : 環境を切り替えながら学習ループを回す．ロールアウトを実行し集めたデータで勾配更新の流れ
- stable-baselines3の実装をベースにAI agent用に改良（複数環境の切り替えやTransformer用の拡張）
### ./policy.py
- 各コンポーネントをまとめて全体のモデルにしている部分．ロールアウトバッファに関する定義もここにある
- `Transformer_Policy` : モデル本体．Transformer，方策・価値ネットワークなどを備え順伝搬等の処理を記述してある
- `RolloutBuffer` : シミュレーション時のデータを格納するためのバッファの定義&GAEの計算処理もここにある
### ./NegTransformer.py
- AIエージェントに用いたTransformer部分の実装．
### ./envs/env.py
- gymnasium環境を継承した交渉シミュレーション用環境を定義．
- `NaiveEnv` : RL用の各種変数やドメイン読み込み，交渉セッション，報酬等の定義
- `AOPEnv` : `step`を改良しAOP準拠の挙動を実装
### ./envs/rl_negotiator.py
- 学習時・テスト時のAIエージェント本体であるNegotiatorを実装
- `RLNegotiator` : 学習時のエージェント本体．`env.py`の`step`時に選択された`self.next_bid`がそのまま相手に提案される
- `TestRLNegotiator.py` : テスト時のエージェント本体．チェックポイントからモデルをロードしそれを用いて推論を行う．
### ./envs/observer.py
- bid履歴を観測するためのobserverを定義
- `EmbeddedObserveHistroy` : OpenAIのテキスト埋め込みによる埋込ベクトルをjsonファイルからロードし，実際のbidを埋め込みバッファに格納する
### ./embedding_model.py
- 新規に埋め込みベクトルを作成したい場合はこのファイルを実行．
- `MyEmbedding` : 埋め込みベクトルを新規作成 or 作成済みの埋め込みベクトルを`embeddings`に保存してあるjsonファイルからロードする
- `self.client = openai.OpenAI(api_key='')`にapi_keyを入力

