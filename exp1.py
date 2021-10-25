import argparse
import code
import time
import numpy as np
import torch
import copy
import tqdm
import matplotlib.pyplot as plt
from filesys_manager import ExperimentPath
from scipy.stats import norm
from hill_discrete import Hill
from model import init_single_and_ensemble, flatten_param, visualize_param, Linear, Quadratic


class Trainer:

    def current_t(self):
        return self.t

    def __init__(self, path):
        self.exp_path = ExperimentPath(path)
        # env
        def f(x): return norm.pdf(x, 10, 2)
        self.env = Hill(f)
        self.batch_size = 5
        # model, optimizer
        self.t = 0
        self.lr = 1e-2
        self.single, self.ensemble = init_single_and_ensemble(ensemble_size=5)
        self.opt_single = torch.optim.SGD(params=self.single.parameters(), lr=self.lr)
        ensemble_param_list = []
        for member in self.ensemble:
            ensemble_param_list.extend(list(member.parameters()))
        self.opt_ensemble = torch.optim.SGD(params=ensemble_param_list, lr=self.lr)
        # statistics
        self.x = torch.tensor(self.env.x, dtype=torch.float32).view(-1, 1)
        self.ground_truth = torch.tensor(self.env.q_ground_truth, dtype=torch.float32)
        self.timesteps = []
        self.single_stats = argparse.Namespace(
            squared_bellman_loss=[],
            ground_truth_mse=[]
        )
        self.ensemble_stats = copy.deepcopy(self.single_stats)

    def train(self):
        # visualize_param(self.single, self.ensemble)
        s, a, r, s_ = self.env.sample_batch(self.batch_size)
        # gradient update single
        self.opt_single.zero_grad()
        v_s_, _ = self.single(s_).detach().max(dim=1, keepdim=True)
        loss_single = torch.mean((r + self.env.gamma * v_s_ - self.single(s) * a) ** 2)
        loss_single.backward()
        self.opt_single.step()
        # gradient update ensemble
        self.opt_ensemble.zero_grad()
        v_s_ensemble, _ = torch.cat(
            [member(s_).detach().view(1, -1, 2)
            for member in self.ensemble], dim=0).mean(dim=0).max(dim=1, keepdim=True)
        # verify target equivalence
        # assert torch.allclose(v_s_.view(-1), v_s_ensemble.view(-1), atol=0.001), (v_s_, v_s_ensemble, 'target not equal')

        loss_ensemble_member = []
        for member in self.ensemble:
            loss_ensemble_member.append(((r + self.env.gamma * v_s_ensemble - member(s) * a) ** 2).mean().view(-1, 1))
        loss_ensemble = torch.sum(torch.cat(loss_ensemble_member, dim=0))
        loss_ensemble.backward()
        self.opt_ensemble.step()
        # verify equivalence
        # visualize_param(self.single, self.ensemble)
        # for param, value in self.single.state_dict().items():
        #     ensemble_param = 0
        #     for model in self.ensemble:
        #         ensemble_param += model.state_dict()[param]
        #     ensemble_param /= len(self.ensemble)
        #     assert torch.allclose(ensemble_param, value, atol=0.001), 'param mean not equal'

        self.t += 1
        self.record_stats(loss_single, loss_ensemble)

    def record_stats(self, loss_single, loss_ensemble):
        with torch.no_grad():
            # self.single_stats.squared_bellman_loss.append(loss_single.item())
            # self.ensemble_stats.squared_bellman_loss.append(loss_ensemble.item() / len(self.ensemble))
            single_mse_ground_truth = torch.mean((self.single(self.x) - self.ground_truth) ** 2)
            ensemble_pred = torch.cat(
                [member(self.x).detach().view(1, -1, 2)
                for member in self.ensemble], dim=0).mean(dim=0)
            ensemble_mse_ground_truth = torch.mean((ensemble_pred - self.ground_truth) ** 2)
            # self.single_stats.ground_truth_mse.append(single_mse_ground_truth)
            # self.ensemble_stats.ground_truth_mse.append(ensemble_mse_ground_truth)

            # self.timesteps.append(self.t)
            self.exp_path.stats.csv_writerow([
                self.t,
                loss_single.item(),
                loss_ensemble.item() / len(self.ensemble),
                single_mse_ground_truth.item(),
                ensemble_mse_ground_truth.item(),
            ])

    def train_self(self, maxt):
        for i in tqdm.trange(maxt):
            self.train()


if __name__ == '__main__':
    from multiprocessing import Process, Manager
    from multiprocessing.managers import BaseManager


    stopt = 100_000
    trainers = []
    ps = []

    BaseManager.register('Trainer', Trainer)
    manager = BaseManager()
    manager.start()

    def trainfunc(trainer):
        trainer.train_self(maxt=stopt)

    for run in range(10):
        trainer = manager.Trainer(f'exp/exp1/run{run}')
        trainers.append(trainer)
        p = Process(target=Trainer.train_self, args=(trainer, stopt, ))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()


