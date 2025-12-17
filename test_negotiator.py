import argparse

import gc
import os
import csv
from itertools import product

from tqdm import tqdm

from negmas import load_genius_domain_from_folder
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *
from envs.rl_negotiator import TestRLNegotiator
from matplotlib import pyplot as plt

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
global LOAD_PATH
global PLOT


def a(x):
    return 'T' if x else 'F'


def run_session_trained(path, save_path, opponent, issue, domain, util1, util2, det, noise, decoder_only, scale, decoder_num, is_first_turn=False):
    session = MySAOMechanism(issues=domain, n_steps=80, avoid_ultimatum=False)
    my_agent = TestRLNegotiator(domain, issue, opponent, path, deterministic=det, accept_offer=False, decoder_only=decoder_only, scale=scale, decoder_num=decoder_num)
    opponent = get_opponent(opponent, add_noise=noise)

    if is_first_turn:
        session.add(my_agent, ufun=util1)
        session.add(opponent, ufun=util2)
    else:
        session.add(opponent, ufun=util2)
        session.add(my_agent, ufun=util1)

    values = []
    for _ in session:
        my_agent.observation = my_agent.observer(session.state.asdict())
        if session.state['last_negotiator'] == 'RLAgent':
            values.append(my_agent.values)
    result = session.state
    print(result)


    if result['agreement'] is not None:
        agreement_offer = tuple(v for k, v in result['agreement'].items())
        my_util, opp_util = util1(agreement_offer), util2(agreement_offer)
    else:
        my_util, opp_util = 0, 0

    results = [
        my_util,
        opp_util,
        my_util + opp_util,
        my_util * opp_util,
        result['agreement'],
        result['step'], 
        result['last_negotiator'], 
        values[-1],
    ]
    print(results)

    # 結果を描画
    if PLOT:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(values)),values,linestyle='-')
        ax.set_xlim(0,80)
        ax.set_ylim(0,1.0)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        plt.savefig(save_path + f'values.png')

        my_agent.name = "Our Agent"
        session.plot(path=save_path + '/plot.png')
        plt.clf()
        plt.close()
        with open(save_path + f'values.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(values)

    session.reset()
    del my_agent, session, opponent
    gc.collect()
    return [
        my_util,
        opp_util,
        my_util + opp_util,
        my_util * opp_util,
        result['agreement'],
        result['step'], 
        result['last_negotiator'], 
    ]


def test_trained(config):
    issue, agent, det, noise, save_path, decoder_only, scale, decoder_num, is_first = config
    results = [['my_util', 'opp_util', 'social', 'nash', 'agreement', 'step', 'last_neg', 'value']]
    scenario = load_genius_domain_from_folder('domain/' + issue)
    domain = scenario.issues
    util1 = scenario.ufuns[0].scale_max(1.0)
    util2 = scenario.ufuns[1].scale_max(1.0)
    for _ in tqdm(range(1 if PLOT else 100)):
        result = run_session_trained(f'{LOAD_PATH}/checkpoint.pt', save_path, agent, issue, domain, util1, util2, det, noise, decoder_only, scale, decoder_num, is_first)
        results.append(result)

    if not PLOT:
        with open(f'{save_path}{issue}-{agent}-d{a(det)}-n{a(noise)}.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(results)
            print(results)
            print(result)
    else:
        print(results)
        print(result)


def get_opponent(opponent, add_noise=False):
    if opponent == 'Boulware':
        opponent = TimeBasedNegotiator(name='Boulware', aspiration_type=4.0, add_noise=add_noise)
    elif opponent == 'Linear':
        opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
    elif opponent == 'Conceder':
        opponent = TimeBasedNegotiator(name='Conceder', aspiration_type=0.2, add_noise=add_noise)
    elif opponent == 'TitForTat1':
        opponent = AverageTitForTatNegotiator(name='TitForTat1', gamma=1, add_noise=add_noise)
    elif opponent == 'TitForTat2':
        opponent = AverageTitForTatNegotiator(name='TitForTat2', gamma=2, add_noise=add_noise)
    elif opponent == 'AgentK':
        opponent = AgentK(add_noise=add_noise)
    elif opponent == 'HardHeaded':
        opponent = HardHeaded(add_noise=add_noise)
    elif opponent == 'Atlas3':
        opponent = Atlas3(add_noise=add_noise)
    elif opponent == 'AgentGG':
        opponent = AgentGG(add_noise=add_noise)
    else:
        opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
    return opponent


def main_trained():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', '-a', nargs='*', required=True, type=str)
    parser.add_argument('--issues', '-i', nargs='*', required=True, type=str)
    parser.add_argument('--model', '-m', required=True, type=str)
    parser.add_argument('--scale', '-s', type=str, default='small')
    parser.add_argument('--decoder_only', '-do', action='store_true')
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--is_first', '-if', action='store_true')
    parser.add_argument('--decoder_num', '-dn', type=int, default=1)
    args = parser.parse_args()
    print(args)

    agents = args.agents
    issues = args.issues
    model_path = args.model
    scale = args.scale
    decoder_only = args.decoder_only
    plot = args.plot
    is_first = args.is_first
    decoder_num = args.decoder_num

    global LOAD_PATH
    LOAD_PATH = model_path

    global PLOT
    PLOT = plot

    if isinstance(issues, str):
        issues = [issues]
    if isinstance(agents, str):
        agents = [agents]

    for agent in agents:
        for issue in issues:
            for det, noise in product([False], [False]):
                save_path = LOAD_PATH + ('/img' if PLOT else '/csv') + f'/{agent}/' + f'/{issue}/' + f'/det={det}_noise={noise}/'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                test_trained((issue, agent, det, noise, save_path, decoder_only, scale, decoder_num, is_first))


if __name__ == '__main__':
    main_trained()
