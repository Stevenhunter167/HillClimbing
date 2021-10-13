import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import copy
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
        self.exp_path = ExperimentPath('exp/simple_linear_shiftrwd')
        self.ensemble_size = 5
        self.batch_size = 5
        self.q_ensemble = nn.ModuleList(Q() for _ in range(self.ensemble_size))
        self.opt = torch.optim.Adam(lr=1e-3, params=self.q_ensemble.parameters())

        def f(x):
            return norm.pdf(x, 10, 2)
        self.env = Hill(f, r_shift=-0.1)
        self.param_trace_left_weight = []
        self.param_trace_left_bias = []

    def mean_q_target(self, s_):
        with torch.no_grad():
            q_preds = []
            for k in range(self.ensemble_size):
                q_preds.append(self.q_ensemble[k](s_).view(1, -1, 2))
            # print(q_preds[0].shape)
            mean_q = torch.cat(q_preds, dim=0).mean(dim=0)
            # print(mean_q.shape)
            return mean_q

    def plot_landscape(self):
        weight = self.q_ensemble[0].net.__dict__['_parameters']['weight'].detach().view(-1).numpy()
        bias = self.q_ensemble[0].net.__dict__['_parameters']['bias'].detach().view(-1).numpy()
        self.param_trace_left_weight.append(weight[0])
        self.param_trace_left_bias.append(bias[0])
        plt.xlabel('weight')
        plt.ylabel('bias')
        plt.scatter(self.param_trace_left_weight[:-1], self.param_trace_left_bias[:-1], color='black', label='history')
        plt.scatter(self.param_trace_left_weight[-1:], self.param_trace_left_bias[-1:], color='red', label='current')

        # x_step = np.linspace(-1, 2, 100)
        # y_step = np.linspace(-1, 2, 100)
        x_step = np.linspace(-0.7, 0.5, 100)
        y_step = np.linspace(-1.5, -0.3, 100)
        contour_x = np.stack([x_step for _ in range(y_step.shape[0])]).T
        contour_y = np.stack([y_step for _ in range(x_step.shape[0])])
        contour_z = np.zeros(contour_x.shape)
        state_dict = copy.deepcopy(self.q_ensemble[0].net.state_dict())
        v, _ = self.mean_q_target(torch.tensor(self.env.x / self.env.s_max, dtype=torch.float32).view(-1, 1)).max(1)
        for i, x in enumerate(x_step):
            for j, y in enumerate(y_step):
                q_temp = nn.Linear(1,2)
                state_dict['weight'][0][0] = x
                state_dict['weight'][1][0] = y
                q_temp.load_state_dict(state_dict)
                qsa = q_temp(torch.tensor(self.env.x / self.env.s_max, dtype=torch.float32).view(-1, 1)).detach()
                loss = torch.mean((torch.tensor(self.env.er[:,0]) + self.env.gamma * v - qsa[:,0]) ** 2).item()
                # print(loss, contour_z.shape)
                # code.interact(local=locals())
                contour_z[i, j] = loss
        # pprint.pprint(np.round(contour_z, 5).tolist())
        plt.contour(contour_x, contour_y, contour_z)
        # input()

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
            # self.env.render()
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

            plt.title(name)
            self.plot_landscape()
            plt.savefig(self.exp_path.ensemble_plots_landscape[name]._path)
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
