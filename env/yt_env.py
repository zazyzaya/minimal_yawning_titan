from random import choice, random, randint

import networkx as nx
import torch 
from torch_geometric.nn import MessagePassing 
from torch_geometric.utils import to_undirected

def build_graph(n, p=0.1, min_v=0.01, max_v=1):
    g = nx.erdos_renyi_graph(n, p)
    ei = [list(e) for e in g.edges]

    nodes = set(range(n))
    isolated_nodes = nodes - set(sum(ei, []))
    nonisolated = list(nodes - isolated_nodes)

    # Authors use "amended Erdos-Renyi" s.t. no nodes
    # are isolated. (Note: this could still cause graph 
    # to be disconnected)
    for n in isolated_nodes: 
        ei.append([n,choice(nonisolated)])

    # Not sure of a better way to address this. Could also just 
    # add random edges, but let's see how frequent this actually is 
    g = nx.Graph(ei)
    if not nx.is_connected(g):
        return build_graph(n,p,min_v, max_v)

    # Now that we know graph is connected, it's very simple to spin up 
    # the node features. Just a (n x 2) matrix of each node's 
    # vulnerability score, and a 0 to show it's not comprimised
    span = max_v - min_v 
    x = torch.zeros((n,2))
    x[:, 0] = (torch.rand(n) * span) + min_v 

    # Lastly, convert edges to torch-friendly representation 
    ei = torch.tensor(ei, dtype=torch.long).T 
    ei = to_undirected(ei)

    return x, ei 
    

class YTEnv: 
    VULN = 0 
    COMP = 1 
    
    # Default, finally found in https://github.com/dstl/YAWNING-TITAN/src/yawning_titan/game_modes/_package_data/game_modes.json
    # I think they used scenario 3 of the ones they have. 
    RED_SKILL = 0.5 

    def __init__(self, x,ei, patch_strength=0.2):
        self.orig_x = x
        self.x = x.clone() 
        self.num_nodes = x.size(0)
        
        # Remains static (for now...)
        self.ei = ei 

    def reset(self):
        self.x = self.orig_x.clone() 
        return self.state() 
    
    def state(self):
        return self.x, self.ei

    def step(self, action, target):
        action(target)
        return (self.num_nodes - self.x[:, self.COMP].sum()) / self.num_nodes

    def patch(self, nid):
        vscore = self.x[nid, self.VULN]
        vscore = max(0.2, vscore-0.2)
        self.x[nid, self.VULN] = vscore 

    def restore(self, nid):
        self.x[nid] = self.orig_x[nid]

    def noop(self, *args):
        pass 

    def attack(self, nid):
        attack_strength = self.RED_SKILL * self.x[nid, self.VULN]

        # Success
        if random() < attack_strength:
            self.x[nid, self.COMP] = 1. 

    def zero_day(self, nid):
        self.x[nid, self.COMP] = 1. 


class RedAgent:
    def __init__(self, env, zd_rate=3):
        self.zd = self.zd_rate = zd_rate 
        self.env = env 
        self.mp = MessagePassing()

    def select_action(self, x,ei):
        # All nodes one-hop from a comprimised node
        reachable = self.mp.propagate(
            ei, x=x[:, 1:]
        ).squeeze(-1).bool()

        # Select target node within reach 
        infectable = (x[:, 1] == 0).logical_and(reachable).nonzero().squeeze(-1)
        target = infectable[randint(0, infectable.size(0)-1)]

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
            return self.num_to_action(distro.argmax())
        
        a = distro.sample()
        p = distro.log_prob(a) 

        return self.num_to_action(a), a,value,p
