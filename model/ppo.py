from collections import deque

import torch 
from torch import nn 
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch_geometric.nn import GCNConv

class ActorNetwork(nn.Module):
    def __init__(self, in_dim, num_nodes, action_space, 
                 hidden1=32, hidden2=64, lr=0.005):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2*num_nodes, action_space),
            nn.Softmax(dim=-1)
        )

        self.drop = nn.Dropout()
        self.opt = Adam(self.parameters(), lr)
        self.num_nodes = num_nodes

    def forward(self, x, ei):
        x = torch.relu(self.conv1(x, ei))
        x = torch.relu(self.conv2(x, ei))
    
        nbatches = x.size(0) // self.num_nodes
        dist = self.out(x.reshape(nbatches, self.num_nodes*x.size(1)))

        return Categorical(dist)


class CriticNetwork(nn.Module):
    def __init__(self, in_dim, num_nodes, 
                 hidden1=32, hidden2=64, lr=0.01):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Linear(hidden2*num_nodes, 1)

        self.opt = Adam(self.parameters(), lr)
        self.num_nodes = num_nodes

    def forward(self, x, ei):
        x = torch.relu(self.conv1(x, ei))
        x = torch.relu(self.conv2(x, ei))

        nbatches = x.size(0) // self.num_nodes
        return self.out(x.reshape(nbatches, self.num_nodes*x.size(1)))


class GraphPPO():
    def __init__(self, num_nodes, in_dim, action_space, batch_size, buffer_size,
                 gamma=0.99, lmbda=0.95, clip=0.1, alr=0.005, clr=0.01):
        super().__init__()
        self.args = (num_nodes, in_dim, action_space, batch_size, buffer_size)
        self.kwargs = dict(gamma=gamma, lmbda=lmbda, clip=clip, alr=alr, clr=clr)

        self.actor = ActorNetwork(in_dim, num_nodes, action_space, lr=alr)
        self.critic = CriticNetwork(in_dim, num_nodes, lr=clr)
        self.memory = PPOMemory(batch_size, buffer_size)

        self.mse = nn.MSELoss()
        self.gamma = gamma 
        self.lmbda = lmbda
        self.clip = clip 

    def __step(self):
        self.actor.opt.step()
        self.critic.opt.step()
        
    def __zero_grad(self):
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    def train(self): 
        self.training = True 
        self.actor.train() 
        self.critic.train()

    def eval(self): 
        self.training = False 
        self.actor.eval()
        self.critic.eval()

    def remember(self, s,a,v,p,r,t):
        self.memory.remember(s,a,v,p,r,t)

    def forward(self, x,ei):
        return self.actor(x,ei), self.critic(x,ei)
    
    def __call__(self,x,ei):
        return self.forward(x,ei)

    def learn(self, epochs, verbose=True):
        '''
        Assume that an external process is adding memories to 
        the PPOMemory unit, and this is called every so often
        '''
        for e in range(epochs):
            s,a,v,p,r,t, batches = self.memory.get_batches()

            '''
            advantage = torch.zeros(len(s), dtype=torch.float)

            # Probably a more efficient way to do this in parallel w torch
            for t in range(len(s)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(s)-1):
                    a_t += discount*(r[k] + self.gamma*v[k+1] -v[k])
                    discount *= self.gamma*self.lmbda

                advantage[t] = a_t
            '''
            rewards = []
            discounted_reward = 0 
            for reward, is_terminal in zip(reversed(r), reversed(t)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                rewards.insert(0, discounted_reward)
                    
            r = torch.tensor(rewards, dtype=torch.float)
            r = (r - r.mean()) / (r.std() + 1e-5) # Normalize rewards

            advantages = r - torch.tensor(v)

            for b_idx,b in enumerate(batches):
                b = b.tolist()
                new_probs = []

                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]
                batched_states = combine_subgraphs(s_)
                dist = self.actor(*batched_states)
                
                critic_vals = self.critic(*batched_states)
                new_probs = dist.log_prob(torch.tensor(a_))
                old_probs = torch.tensor([p[i] for i in b])
                entropy = dist.entropy()

                a_t = advantages[b]

                # Equiv to exp(new) / exp(old) b.c. recall: these are log probs                
                r_theta = (new_probs - old_probs).exp()
                clipped_r_theta = torch.clip(
                    r_theta, min=1-self.clip, max=1+self.clip
                )

                # Use whichever one is minimal 
                actor_loss = torch.min(r_theta*a_t, clipped_r_theta*a_t)
                actor_loss = -actor_loss.mean()

                # Critic uses MSE loss between expected value of state and observed
                # reward with discount factor 
                critic_loss = self.mse(r[b].unsqueeze(-1), critic_vals)

                # Not totally necessary but maybe will help?
                entropy_loss = entropy.mean()

                # Calculate gradient and backprop
                total_loss = actor_loss + 0.5*critic_loss - 0.01*entropy_loss
                self.__zero_grad()
                total_loss.backward() 
                self.__step()

                if verbose:
                    print(f'[{e}] C-Loss: {0.5*critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item()*0.01:0.4f}')

        return total_loss.item()
    
    def save(self, outf='saved_models/ppo.pt'):
        torch.save({
            'args': self.args,
            'kwargs': self.kwargs,
            'actor': self.actor.state_dict(), 
            'critic': self.critic.state_dict()
        }, outf)


def load_ppo(fname):
    db = torch.load(fname)
    agent = GraphPPO(*db['args'], **db['kwargs'])

    agent.actor.load_state_dict(db['actor'])
    agent.critic.load_state_dict(db['critic'])

    return agent 

class PPOMemory:
    def __init__(self, bs, buffer_size):
        self.s = deque([], buffer_size)
        self.a = deque([], buffer_size)
        self.v = deque([], buffer_size)
        self.p = deque([], buffer_size)
        self.r = deque([], buffer_size)
        self.t = deque([], buffer_size)

        self.bs = bs 

    def remember(self, s,a,v,p,r,t):
        '''
        Args are state, action, value, log_prob, reward
        '''
        self.s.append(s)
        self.a.append(a)
        self.v.append(v)
        self.p.append(p)
        self.r.append(r) 
        self.t.append(t)

    def get_batches(self):
        idxs = torch.randperm(len(self.a))
        batch_idxs = idxs.split(self.bs)

        return self.s, self.a, self.v, \
            self.p, self.r, self.t, batch_idxs


def combine_subgraphs(states):
    xs,eis = zip(*states)
    
    # ei we need to update each node idx to be
    # ei[i] += len(ei[i-1])
    offset=0
    new_eis=[]
    for i in range(len(eis)):
        new_eis.append(eis[i]+offset)
        offset += xs[i].size(0)

    # X is easy, just cat
    xs = torch.cat(xs, dim=0)
    eis = torch.cat(new_eis, dim=1)

    return xs,eis