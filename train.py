import dill
import torch

dill.extend(False)
import argparse
import os
from datetime import datetime

from ppo_scratch import PPO
from envs.observer import *

from datetime import datetime

dill.extend(True)

import faulthandler
faulthandler.enable()

ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A'
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    'TitForTat1',
    'TitForTat2',
    "AgentK",
    "HardHeaded",
    "Atlas3",
    "AgentGG",
]
global SAVE_PATH


def run_rl(issue, agents, model_path, device, lr, total_timesteps, batch_size, scale, random_train, decoder_only, horizon, entropy_coef, clip_range, decoder_num):
    if model_path is not None:
        checkpoint = torch.load(f'{model_path}/checkpoint.pt',map_location='cpu')
        model = PPO(issue=issue, agents=agents, learning_rate=lr, batch_size=batch_size, device=device, random_train=random_train, decoder_only=decoder_only, scale=scale, n_rollout_steps=horizon, ent_coef=entropy_coef, clip_range=clip_range, decoder_num=decoder_num)
        model.model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        model = PPO(issue=issue, agents=agents, learning_rate=lr, device=device, batch_size=batch_size, random_train=random_train, decoder_only=decoder_only, scale=scale, n_rollout_steps=horizon, ent_coef=entropy_coef, clip_range=clip_range, decoder_num=decoder_num)
    model.train(total_timesteps=total_timesteps, save_path=SAVE_PATH)

    del model


def main():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # IssueとAgentを指定して実行 -> --agents, --issue
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', '-a', required=True, nargs='*', type=str)
    parser.add_argument('--issue', '-i', required=True, nargs='*', type=str)
    # 学習済モデルをfine-tuningする場合はパス
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--device', '-d', type=str, default="auto")
    # ハイパラ
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--total_timesteps', '-ts', type=int, default=300_000)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--horizon', '-hr', type=int, default=2048)
    parser.add_argument('--entropy_coef', '-ec', type=float, default=0.0)
    parser.add_argument('--clip_range', '-cr', type=float, default=0.2)
    
    parser.add_argument('--scale', '-s', type=str, default='small') # 埋め込みベクトルのサイズ変えるなら．現状なにか指定する必要なし
    parser.add_argument('--random_train', '-r', action='store_true') # 学習順ランダム化
    parser.add_argument('--decoder_only', '-do', action='store_true') # decoderのみのモデル学習
    parser.add_argument('--decoder_num', '-dn', type=int, default=1) # decoderのブロック数指定
    parser.add_argument('--save_path', '-sp', type=str, default="./results/")
    args = parser.parse_args()
    print(args)

    agents = args.agents
    issue = args.issue
    model_path = args.model
    device = args.device
    lr = args.learning_rate
    total_timesteps = args.total_timesteps
    batch_size = args.batch_size
    scale = args.scale
    random_train = args.random_train
    decoder_only = args.decoder_only
    decoder_num = args.decoder_num
    entropy_coef = args.entropy_coef
    clip_range = args.clip_range
    horizon = args.horizon
    save_path = args.save_path

    global SAVE_PATH
    SAVE_PATH = "./results/{}_{}/{}-TA/".format('-'.join(issue), '-'.join(agents), current_time) if save_path == './results/' else save_path

    os.makedirs(SAVE_PATH, exist_ok=True)
    run_rl(issue, agents, model_path, device, lr, total_timesteps, batch_size, scale, random_train, decoder_only, horizon, entropy_coef, clip_range, decoder_num)


if __name__ == '__main__':
    main()
