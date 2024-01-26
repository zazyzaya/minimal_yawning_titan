from argparse import ArgumentParser

import torch 

from env.yt_env import YTEnv, BlueAgent, build_graph
from model.ppo import GraphPPO

MAX_STEPS = 5e6

# Using stable baselines hyperparams
EPOCHS = 10
BATCH_SIZE=64
N = 2048
LR = 0.0003

SEED = 0

torch.set_num_threads(16)

@torch.no_grad()
def simulate_n(env: YTEnv, agent: BlueAgent):
    agent.model.eval()
    steps = 0 

    rewards = []
    lens = []
    while True: 
        s = env.reset()
        t = False 
        tot_r = 0 

        while not t: 
            if steps == N:
                return lens, rewards
            
            (act,target), a,v,p = agent.select_action(*s)
            r,t,next_s = env.step(act,target)
            agent.model.remember(s,a,v,p,r,t)
            s = next_s 

            tot_r += r 
            steps += 1

        rewards.append(tot_r) 
        lens.append(env.ts)
    

@torch.no_grad()
def simulate(env: YTEnv, agent: BlueAgent):
    agent.model.eval()
    s = env.reset()
    tot_r = 0
    t = False 

    while not t: 
        (act,target), a,v,p = agent.select_action(*s)
        r,t,next_s = env.step(act,target)
        agent.model.remember(s,a,v,p,r,t)
        s = next_s

        tot_r += r 

    return env.ts, tot_r 

def experiment(env: YTEnv, agent: BlueAgent):
    tr_steps = 0 
    ep_lens, ep_rews = [],[]

    while tr_steps < MAX_STEPS:   
        steps, rew = simulate_n(env, agent)
        
        ep_lens += steps 
        ep_rews += rew 
        tr_steps += N

        r = ep_rews[-100:]
        l = ep_lens[-100:]

        avg_r = sum(r)/len(r)
        agent.model.learn(EPOCHS, verbose=False)

        print(f'[{tr_steps}] Avg r: {avg_r:0.2f}, Avg l: {sum(l)/len(l):0.2f}')
        torch.save({'rews': ep_rews, 'lens': ep_lens}, f'logs/{GRAPH_SIZE}N_{SEED}.pt')

    agent.model.save(f'saved_models/ppo_{GRAPH_SIZE}N_{SEED}_last.pt')

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('num_nodes', nargs=1, type=int)

    args = ap.parse_args()
    GRAPH_SIZE = args.num_nodes[0]

    x,ei = build_graph(GRAPH_SIZE, seed=SEED)
    env = YTEnv(x,ei)
    
    blue = GraphPPO(GRAPH_SIZE, x.size(1), env.blue_action_space, BATCH_SIZE, alr=LR, clr=LR)
    agent = BlueAgent(env, blue)

    experiment(env, agent)