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
from exp1 import Trainer

class Trainer(Trainer):

    def __init__(self, path):
        super().__init__(path)
        # replay buffer
        self.replay_buffer = []

    def sample(self):
        samples = [self.replay_buffer[i]
            for i in np.random.randint(len(self.replay_buffer), size=self.batch_size)]
        s = torch.cat([sample[0] for sample in samples], dim=0)
        a = torch.cat([sample[1] for sample in samples], dim=0)
        r = torch.cat([sample[2] for sample in samples], dim=0)
        s_ = torch.cat([sample[3] for sample in samples], dim=0)
        return s, a, r, s_

    def train(self):
        # visualize_param(self.single, self.ensemble)
        # sample from env, store to replay buffer
        self.replay_buffer.append(self.env.sample_batch(1))
        # sample from replay buffer, update model
        batches = [self.sample() for k in range(len(self.ensemble))]
        # gradient update
        loss_single, loss_ensemble = self.train_on_batches(batches)
        # record stats
        self.t += 1
        self.record_stats(loss_single, loss_ensemble)

    def train_on_batches(self, batches):
        # grad update single
        loss_single_total = 0
        for (s, a, r, s_) in batches:
            self.opt_single.zero_grad()
            v_s_, _ = self.single(s_).detach().max(dim=1, keepdim=True)
            loss_single = torch.mean((r + self.env.gamma * v_s_ - self.single(s) * a) ** 2) / len(self.ensemble)
            loss_single_total += loss_single.detach()
            loss_single.backward()
            self.opt_single.step()
        # gradient update ensemble
        loss_ensemble_member = []
        for k, (s, a, r, s_) in enumerate(batches):
            self.opt_ensemble.zero_grad()
            v_s_ensemble, _ = torch.cat(
                [member(s_).detach().view(1, -1, 2)
                 for member in self.ensemble], dim=0).mean(dim=0).max(dim=1, keepdim=True)
            loss_ensemble_member.append(
                ((r + self.env.gamma * v_s_ensemble - self.ensemble[k](s) * a) ** 2).mean().view(-1, 1))
        loss_ensemble = torch.sum(torch.cat(loss_ensemble_member, dim=0))
        loss_ensemble.backward()
        self.opt_ensemble.step()

        return loss_single_total, loss_ensemble.detach()

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
        trainer = manager.Trainer(f'exp/exp1_replay/run{run}')
        trainers.append(trainer)
        p = Process(target=Trainer.train_self, args=(trainer, stopt, ))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()


