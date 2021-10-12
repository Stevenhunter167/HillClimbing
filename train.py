import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy.stats import norm
from hill_discrete import Hill
from filesys_manager import ExperimentPath

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2)
        )
    def forward(self, x):
        return self.net(x)

class Trainer:

    def __init__(self):
        self.exp_path = ExperimentPath('exp/simple_linear')
        self.q = Q()
        self.opt = torch.optim.Adam(lr=1e-3, params=self.q.parameters())
        self.batch_size = 5
        def f(x):
            return norm.pdf(x, 10, 2)
        self.env = Hill(f)

    def train(self, i):
        # compute values
        s, a, r, s_ = self.env.sample_batch(self.batch_size)
        q_target = r + self.env.gamma * self.q(s_).detach()
        vs_, _ = q_target.max(dim=1, keepdim=True)
        qsa = self.q(s) * a

        # plot
        if i % 100 == 0:
            name = f'transition_sampled_{i * self.batch_size}'
            plt.title(name)
            self.env.render()
            q_pred = self.q(
                torch.linspace(0, 1, (self.env.s_max - self.env.s_min) * self.env.density).view(-1, 1)
            ).detach().numpy()
            self.env.plot_q(q_pred)
            plt.legend()
            plt.savefig(self.exp_path.plots[name]._path)
            plt.close()

        # optimize
        self.opt.zero_grad()
        loss = torch.mean((qsa - vs_) ** 2)
        loss.backward()
        self.opt.step()

if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)

    # print(list(Q().net.parameters()))

    trainer = Trainer()
    for i in tqdm.trange(300000 // trainer.batch_size):
        trainer.train(i)
