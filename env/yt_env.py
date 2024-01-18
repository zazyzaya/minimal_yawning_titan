import random 

import networkx as nx
import torch 
from torch_geometric.nn import MessagePassing 
from torch_geometric.utils import to_undirected, degree, add_remaining_self_loops

PRIMES = [3,5,7,11,13,17,19,23,29,31]

def build_graph(num_nodes, p=0.1, min_v=0.01, max_v=1, recurse=1, seed=None):
    if seed is not None:
        seed_val = PRIMES[seed] ** recurse
        random.seed(seed_val)
        torch.random.manual_seed(seed_val)
        g = nx.erdos_renyi_graph(num_nodes, p, seed=seed_val)
    else:
        g = nx.erdos_renyi_graph(num_nodes, p)

    ei = [list(e) for e in g.edges]

    nodes = set(range(num_nodes))
    isolated_nodes = nodes - set(sum(ei, []))
    nonisolated = list(nodes - isolated_nodes)

    # Authors use "amended Erdos-Renyi" s.t. no nodes
    # are isolated. (Note: this could still cause graph 
    # to be disconnected)
    for n in isolated_nodes: 
        ei.append([n,random.choice(nonisolated)])

    # Not sure of a better way to address this. Could also just 
    # add random edges, but let's see how frequent this actually is 
    g = nx.Graph(ei)
    if not nx.is_connected(g):
        print(f"Graph was disconnected. Trying again ({recurse})")
        return build_graph(num_nodes,p,min_v, max_v, recurse=recurse+1, seed=seed)

    # Now that we know graph is connected, it's very simple to spin up 
    # the node features. Just a (n x 2) matrix of each node's 
    # vulnerability score, and a 0 to show it's not comprimised
    span = max_v - min_v 
    x = torch.zeros((num_nodes,2))
    x[:, 0] = (torch.rand(num_nodes) * span) + min_v 

    # Lastly, convert edges to torch-friendly representation 
    ei = torch.tensor(ei, dtype=torch.long).T 
    ei = to_undirected(ei)

    deg = degree(ei[0])
    x = torch.cat([x, deg.unsqueeze(-1)], dim=1)

    ei = add_remaining_self_loops(ei)[0]
    return x, ei 
    

class YTEnv: 
    VULN = 0 
    COMP = 1 
    DEGREE = 2
    
    # Default, finally found in https://github.com/dstl/YAWNING-TITAN/src/yawning_titan/game_modes/_package_data/game_modes.json
    # I think they used scenario 3 of the ones they have. 
    RED_SKILL = 0.5
    ZERO_DAY_RATE = 3

    NODE_VULN_LOWER_BOUND = 0.01

    REPAIR_PROB = 1
    EP_LEN = 500 

    def __init__(self, x,ei, patch_strength=0.2):
        self.orig_x = x
        self.x = x.clone() 
        self.num_nodes = x.size(0)
        
        # Remains static (for now...)
        self.ei = ei 

        # NOOP + patch * N + restore * N 
        self.blue_action_space = 1 + (x.size(0) * 2 )
        self.red = RedAgent(self, self.ZERO_DAY_RATE)

        self.ts = 0 

    def reset(self):
        self.x = self.orig_x.clone() 
        self.ts = 0 

        # Pick a random entry point for the red agent
        self.x[random.randint(0,self.num_nodes-1), self.COMP] = 1
        return self.state() 
    
    def state(self):
        return self.x, self.ei

    def step(self, blue_action, blue_target):
        # Let red go first
        action,target = self.red.select_action(self.x, self.ei)
        self.act(action, target)

        # Then blue performs action 
        r = self.act(blue_action, blue_target)
        
        terminate = False 
        # Every computer is comprimised
        if r == 0: 
            r = -100 
            terminate = True 
        elif self.ts == self.EP_LEN-1:
            r = 100 
            terminate = True 

        self.ts += 1 
        return r, terminate, self.state()

    def act(self, action, target):
        action(target)
        return (
            self.num_nodes - self.x[:, self.COMP].sum()
        ) / self.num_nodes

    def patch(self, nid):
        vscore = self.x[nid, self.VULN]
        if vscore <= self.NODE_VULN_LOWER_BOUND:
            return 0
    
        self.x[nid, self.VULN] = max(self.NODE_VULN_LOWER_BOUND, vscore-0.2)
        return 0

    def restore(self, nid):
        # Penalize restoring hosts that are safe
        if self.x[nid, self.COMP] == 0:
            ret = -1
        else:
            ret = 0 

        if random.random() < self.REPAIR_PROB:
            self.x[nid] = self.orig_x[nid]

        return ret 

    def noop(self, *args): return 0

    def attack(self, nids):
        for nid in nids:
            attack_strength = self.RED_SKILL * self.x[nid, self.VULN]

            # Success
            if random.random() < attack_strength:
                self.x[nid, self.COMP] = 1. 
        return 0

    def zero_day(self, nid):
        self.x[nid, self.COMP] = 1. 
        return 0


class RedAgent:
    def __init__(self, env, zd_rate=3):
        self.zd = self.zd_rate = zd_rate 
        self.env = env 
        self.mp = MessagePassing()

    def select_action(self, x,ei):
        # All nodes one-hop from a comprimised node
        reachable = self.mp.propagate(
            ei, x=x[:, 1:2]
        ).squeeze(-1).bool()

        # Select target node within reach 
        infectable = (x[:, 1] == 0).logical_and(reachable).nonzero().squeeze(-1)
        
        # No infected nodes remaining
        if infectable.size(0) == 0:
            # Emailed author about what to do here. For now, red just NOOPs
            return self.env.noop, None 

        target = [infectable[random.randint(0, infectable.size(0)-1)]]
        if self.zd == self.zd_rate: 
            self.zd = 0 
            fn = self.env.zero_day 
        else: 
            self.zd += 1 
            fn = self.env.attack

        return fn, target 
    
class BlueAgent:
    def __init__(self, env, model, deterministic=False):
        self.env = env 
        self.model = model 
        self.deterministic = deterministic

    def num_to_action(self, i):
        action = [self.env.patch, self.env.restore, self.env.noop][i // self.env.num_nodes]
        target = i % self.env.num_nodes 
        return action, target 

    def select_action(self, x,ei):
        distro,value = self.model(x,ei)

        if self.deterministic:
            a = distro.probs.argmax()
        else:
            a = distro.sample()

        p = distro.log_prob(a) 
        return self.num_to_action(a.item()), a.item(),value.item(),p.item()
