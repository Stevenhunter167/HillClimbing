import code
import copy
import pprint

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
        self.net = nn.Linear(1, 2)
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

        self.param_trace_left_weight = []
        self.param_trace_left_bias = []

    def plot_landscape(self):
        weight = self.q.net.__dict__['_parameters']['weight'].detach().view(-1).numpy()
        bias = self.q.net.__dict__['_parameters']['bias'].detach().view(-1).numpy()
        # left = np.array([weight[0], bias[0]])
        # right = np.array([weight[1], bias[1]])
        self.param_trace_left_weight.append(weight[0])
        self.param_trace_left_bias.append(bias[0])
        plt.xlabel('weight')
        plt.ylabel('bias')
        plt.scatter(self.param_trace_left_weight[:-1], self.param_trace_left_bias[:-1], color='black', label='history')
        plt.scatter(self.param_trace_left_weight[-1:], self.param_trace_left_bias[-1:], color='red', label='current')

        x_step = np.linspace(-1, 2, 100)
        y_step = np.linspace(-1, 2, 100)
        contour_x = np.stack([x_step for _ in range(y_step.shape[0])]).T
        contour_y = np.stack([x_step for _ in range(x_step.shape[0])])

        # for x in range(x_step.shape[0]):
        #     for y in range(y_step.shape[0]):
        #         print(np.round([contour_x[x,y], contour_y[x,y]], 2), end=' ')
        #     print()

        contour_z = np.zeros(contour_x.shape)
        for i, x in enumerate(x_step):
            for j, y in enumerate(y_step):
                q_temp = nn.Linear(1,2)
                state_dict = copy.deepcopy(self.q.net.state_dict())
                state_dict['weight'][0][0] = x
                state_dict['weight'][1][0] = y
                q_temp.load_state_dict(state_dict)
                qsa = q_temp(torch.tensor(self.env.x / self.env.s_max, dtype=torch.float32).view(-1, 1)).detach()
                v, _ = qsa.max(1)
                loss = torch.mean((torch.tensor(self.env.er[:,0]) + self.env.gamma * v - qsa[:,0]) ** 2).item()
                # print(loss, contour_z.shape)
                # code.interact(local=locals())
                contour_z[i, j] = loss
        # pprint.pprint(np.round(contour_z, 5).tolist())
        plt.contour(contour_x, contour_y, contour_z)
        # input()

    def train(self, i):
        # compute values
        s, a, r, s_ = self.env.sample_batch(self.batch_size)
        q_target = r + self.env.gamma * self.q(s_).detach()
        vs_, _ = q_target.max(dim=1, keepdim=True)
        qsa = self.q(s) * a

        # plot
        if i % 200 == 0:
            name = f'transition_sampled_{i * self.batch_size}'
            plt.title(name)
            self.env.render()
            q_pred = self.q(
                torch.linspace(0, 1, (self.env.s_max - self.env.s_min) * self.env.density).view(-1, 1)
            ).detach().numpy()
            self.env.plot_q(q_pred, color="blue")
            self.env.plot_q()
            plt.legend()
            plt.savefig(self.exp_path.plots[name]._path)
            plt.close()

            plt.title(name)
            self.plot_landscape()
            plt.savefig(self.exp_path.plots_landscape[name]._path)
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
