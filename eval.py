import torch 
from tqdm import tqdm 

from env.yt_env import build_graph, YTEnv, BlueAgent
from env.heuristic_blue import RestoreMostDangerous
from model.ppo import load_ppo
from train import simulate

# TODO make these args
N = 40
SEED = 0

#model = load_ppo(f'saved_models/ppo_{N}N-{SEED}.pt')
#model.eval()
torch.no_grad()

# 50 envs, for 10 trials each
rews = []; lens = []
prog = tqdm(total=500)

for _ in range(50):
    g = build_graph(N)
    env = YTEnv(*g)

    blue = RestoreMostDangerous(env)
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
    f'results/heuristic_eval.pt'  
)

print(f"R Mean: {rews.mean().item()}, L Mean: {lens.mean().item()}")
print(f"R Std: {rews.mean().item()}, L Std: {lens.mean().item()}")