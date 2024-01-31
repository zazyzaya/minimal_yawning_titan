import torch 
from torch import nn 
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from model.ppo import GraphPPO, PPOMemory, combine_subgraphs

class ActorNetwork(nn.Module):
    def __init__(self, in_dim, action_space, 
                 hidden1=32, hidden2=64, lr=0.005):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(), 
            nn.Linear(hidden2, action_space),
        )
        self.softmax = nn.Softmax(dim=-1)

        self.drop = nn.Dropout()
        self.opt = Adam(self.parameters(), lr)

    def forward(self, x, ei, num_nodes):
        '''
        Can still batch input, but assumes every batch-member
        has the same number of nodes (othewise, run one at a time)
        '''

        z = torch.relu(self.conv1(x, ei))
        z = torch.relu(self.conv2(z, ei))
        #x = torch.cat([x,z], dim=1)
        
        nbatches = z.size(0) // num_nodes
        dist = self.out(z).reshape(nbatches, num_nodes*z.size(1))
        dist = self.softmax(dist)

        return Categorical(dist)
    
class CriticNetwork(nn.Module):
    def __init__(self, in_dim, hidden1=32, hidden2=64, lr=0.01):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

        self.opt = Adam(self.parameters(), lr)

    def forward(self, x, ei, num_nodes):
        z = torch.relu(self.conv1(x, ei))
        z = torch.relu(self.conv2(z, ei))
        #x = torch.cat([z,x], dim=1)

        nbatches = z.size(0) // num_nodes
        state_vals = self.out(z).reshape(nbatches, num_nodes*z.size(1))
        return state_vals.mean(dim=1)

class InductiveGraphPPO(GraphPPO):
    def __init__(self, in_dim, action_space, batch_size, 
                 gamma=0.99, lmbda=0.95, clip=0.2, alr=0.005, clr=0.01):
        self.args = (in_dim, action_space, batch_size)
        self.kwargs = dict(gamma=gamma, lmbda=lmbda, clip=clip, alr=alr, clr=clr)

        self.actor = ActorNetwork(in_dim, action_space, lr=alr)
        self.critic = CriticNetwork(in_dim, lr=clr)
        self.memory = PPOMemory(batch_size)

        self.mse = nn.MSELoss()
        self.gamma = gamma 
        self.lmbda = lmbda
        self.clip = clip 

    # Copied and pasted private methods
    def __step(self):
        self.actor.opt.step()
        self.critic.opt.step()
        
    def __zero_grad(self):
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    # Change forward to add in num_nodes for vector reshaping
    def forward(self, x,ei, num_nodes):
        return self.actor(x,ei,num_nodes), self.critic(x,ei, num_nodes)
    
    def __call__(self,x,ei, num_nodes):
        return self.forward(x,ei, num_nodes)

    # Just need to change fn signature when it calls "forward" 
    def learn(self, epochs, verbose=True):
        '''
        Assume that an external process is adding memories to 
        the PPOMemory unit, and this is called every so often
        '''
        num_nodes = self.memory.s[0][0].size(0)

        self.train()
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
                self.__zero_grad()
                
                b = b.tolist()
                new_probs = []

                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]
                batched_states = combine_subgraphs(s_)
                
                dist, critic_vals = self.forward(*batched_states, num_nodes)
                
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
                total_loss = actor_loss + 0.5*critic_loss #- 0.01*entropy_loss
                total_loss.backward() 
                self.__step()

                if verbose:
                    print(f'[{e}] C-Loss: {0.5*critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item()*0.01:0.4f}')

        self.memory.clear()
        return total_loss.item()