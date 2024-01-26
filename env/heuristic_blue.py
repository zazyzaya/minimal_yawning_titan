import torch 
from torch_geometric.utils import degree

from env.yt_env import BlueAgent

class Dummy:
    def remember(self, *args):
        pass 
    def eval(self):
        pass 
    def train(self):
        pass

class RestoreMostDangerous:
    def __init__(self, env, *args, deterministic=False):
        self.env = env 
        self.deg = degree(env.ei[0])

        self.model = Dummy()

    def select_action(self, x,ei):
        infected = (x[:, self.env.COMP] == 1).nonzero().squeeze(-1)

        if infected.size(0) == 0:
            return (self.env.noop, None), None,None,None
        
        most_risky = self.deg[infected].argmax()
        return (self.env.restore, infected[most_risky]), None,None,None