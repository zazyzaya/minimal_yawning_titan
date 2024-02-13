from argparse import ArgumentParser
from math import sqrt 

import torch 
from tqdm import tqdm 

from env.yt_env import build_graph, YTEnv, BlueAgent
from model.inductive_ppo import load_ppo 
from train import simulate

# TODO make these args
ap = ArgumentParser()
ap.add_argument('n', nargs=1, type=int)
ap.add_argument('-s', '--seed', default=0, type=int)
ap.add_argument('-r', '--random', action='store_false')

args = ap.parse_args()

N = args.n[0]
SEED = args.seed 
DETERMINISTIC = args.random 

FNAME = f'ppo_{N}N_{SEED}_last'

model = load_ppo(f'saved_models/og_inductive/{FNAME}.pt')

model.eval()
torch.no_grad()

Z_99_PERCENT = 2.576

def ci(t): 
    return (
        Z_99_PERCENT * 
        (t.std() / sqrt(t.size(0)))
    ).item()

def to_dict(t):
    t = t.float().sort().values
    quartile = t.size(0) // 4
    trunc = t[quartile:-quartile]
    trunc_mean = trunc.mean().item()
    ci_range = ci(trunc)

    return {
        'mean': t.mean().item(),
        'max': t.max().item(),
        'min': t.min().item(),
        'std': t.std().item(),
        'trunc-mean': trunc_mean,
        'CI-low': trunc_mean - ci_range,
        'CI-high': trunc_mean + ci_range
    }

stats = dict()
for n in [10,20,40]:
    if n == N:
        # Already have this data
        continue 

    # 50 envs, for 10 trials each
    rews = []; lens = []
    prog = tqdm(total=500)

    for _ in range(50):
        g = build_graph(n)
        env = YTEnv(*g)

        blue = BlueAgent(env, model, deterministic=True, inductive=True)
        for _ in range(10):
            l,r = simulate(env, blue)
            prog.update()

            rews.append(r)
            lens.append(l) 

    prog.close()
    rews = torch.tensor(rews, dtype=torch.float)
    lens = torch.tensor(lens, dtype=torch.float)

    r = to_dict(rews)
    print(r)
    stats[n] = {'r': r, 'l': to_dict(lens)} 

torch.save(stats, f'results/inductive_extra_tests/{N}_{SEED}.pt')