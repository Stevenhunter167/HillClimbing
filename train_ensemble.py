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
        self.ensemble_size = 5
        self.batch_size = 5
        self.q_ensemble = nn.ModuleList(Q() for _ in range(self.ensemble_size))
        self.opt = torch.optim.Adam(lr=1e-3, params=self.q_ensemble.parameters())

        def f(x):
            return norm.pdf(x, 10, 2)
        self.env = Hill(f)

    def mean_q_target(self, s_):
        with torch.no_grad():
            q_preds = []
            for k in range(self.ensemble_size):
                q_preds.append(self.q_ensemble[k](s_).view(1, -1, 2))
            # print(q_preds[0].shape)
            mean_q = torch.cat(q_preds, dim=0).mean(dim=0)
            # print(mean_q.shape)
            return mean_q

    def train(self, i):
        # compute values
        loss = []
        for k in range(self.ensemble_size):
            s, a, r, s_ = self.env.sample_batch(self.batch_size)
            vs_, _ = (r + self.env.gamma * self.mean_q_target(s_)).max(dim=1, keepdim=True)
            qsa = self.q_ensemble[k](s) * a
            loss.append(((qsa - vs_) ** 2).view(1, -1, 1))
        loss = torch.cat(loss, dim=0).mean()

        # plot
        if i % 40 == 0:
            name = f'transition_sampled_{i * self.batch_size * self.ensemble_size}'
            plt.title(name)
            self.env.render()
            x = torch.linspace(0, 1, (self.env.s_max - self.env.s_min) * self.env.density).view(-1, 1)
            q_pred = self.mean_q_target(x).numpy()
            for k in range(self.ensemble_size):
                with torch.no_grad():
                    qk_pred = self.q_ensemble[k](x).numpy()
                    self.env.plot_q(qk_pred, 'green')
            self.env.plot_q(q_pred, 'blue')
            self.env.plot_q()
            plt.legend()
            plt.savefig(self.exp_path.ensemble_plots[name]._path)
            plt.close()

        # optimize
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)

    # print(list(Q().net.parameters()))

    trainer = Trainer()
    for i in tqdm.trange(300000 // (trainer.batch_size * trainer.ensemble_size)):
        trainer.train(i)
