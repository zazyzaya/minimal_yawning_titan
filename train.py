from tqdm import tqdm 

import torch 

from env.yt_env import YTEnv, BlueAgent, build_graph
from model.ppo import GraphPPO

EPOCHS = 5 
MAX_STEPS = 5e6

BATCH_SIZE=64
BUFFER_SIZE=2**16

SEED = 0
GRAPH_SIZE=10

def train(env: YTEnv, agent: BlueAgent):
    s = env.reset()
    tot_r = 0
    t = False 

    while not t: 
        (act,target), a,v,p = agent.select_action(*s)
        r,t,next_s = env.step(act,target)
        agent.model.remember(s,a,v,p,r,t)

        tot_r += r 

    agent.model.learn(EPOCHS, verbose=False)
    return env.ts, tot_r 

def experiment(env: YTEnv, agent: BlueAgent):
    tr_steps = last_print = 0 
    ep_lens, ep_rews = [],[]

    while tr_steps < MAX_STEPS:
        steps, rew = train(env, agent)
        
        ep_lens.append(steps)
        ep_rews.append(rew)
        tr_steps += steps

        if tr_steps - last_print > 10_000: 
            r = ep_rews[-100:]
            l = ep_lens[-100:]

            print(f'[{tr_steps}] Avg r: {sum(r)/100:0.2f}, Avg l: {sum(l)/100:0.2f}')
            torch.save({'rews': ep_rews, 'lens': ep_lens}, f'logs/{GRAPH_SIZE}N_{SEED}.pt')
            agent.model.save()

            last_print = tr_steps 

    print(f'[{tr_steps}] Avg r: {sum(r)/100:0.2f}, Avg l: {sum(l)/100:0.2f}')
    torch.save({'rews': ep_rews, 'lens': ep_lens}, f'logs/{GRAPH_SIZE}N_{SEED}.pt')
    agent.model.save()


if __name__ == '__main__':
    x,ei = build_graph(GRAPH_SIZE, seed=SEED)
    env = YTEnv(x,ei)
    
    blue = GraphPPO(GRAPH_SIZE, x.size(1), env.blue_action_space, BATCH_SIZE, BUFFER_SIZE)
    agent = BlueAgent(env, blue)

    experiment(env, agent)