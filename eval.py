from argparse import ArgumentParser

import torch 
from tqdm import tqdm 

from env.yt_env import build_graph, YTEnv, BlueAgent
from env.heuristic_blue import RestoreMostDangerous
from model.ppo import load_ppo
from model.inductive_ppo import load_ppo as load_ind_ppo
from train import simulate

# TODO make these args
ap = ArgumentParser()
ap.add_argument('n', nargs=1, type=int)
ap.add_argument('-s', '--seed', default=0, type=int)
ap.add_argument('-r', '--random', action='store_false')
ap.add_argument('--inductive', action='store_true')

args = ap.parse_args()

N = args.n[0]
SEED = args.seed 
DETERMINISTIC = args.random 

FNAME = f'ppo_{N}N_{SEED}_last'

if args.inductive:
    model = load_ind_ppo(f'saved_models/{FNAME}.pt')
else:
    model = load_ppo(f'saved_models/{FNAME}.pt')

model.eval()
torch.no_grad()

# 50 envs, for 10 trials each
rews = []; lens = []
prog = tqdm(total=500)

for _ in range(50):
    g = build_graph(N)
    env = YTEnv(*g)

    blue = BlueAgent(env, model, deterministic=True, inductive=args.inductive)
    for _ in range(10):
        l,r = simulate(env, blue)
        prog.update()

        rews.append(r)
        lens.append(l) 

prog.close()
rews = torch.tensor(rews, dtype=torch.float)
lens = torch.tensor(lens, dtype=torch.float)

torch.save(
    {'rews': rews, 'lens': lens}, 
    f'results/{FNAME}_eval.pt'  
)

print(f"R Mean: {rews.mean().item()}, L Mean: {lens.mean().item()}")
print(f"R Std: {rews.mean().item()}, L Std: {lens.mean().item()}")