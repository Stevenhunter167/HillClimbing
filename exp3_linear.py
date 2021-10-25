import torch
from model import init_single_and_ensemble, DeepLinear
from exp1 import Trainer


class Trainer(Trainer):

    def __init__(self, path):
        super().__init__(path)
        self.single, self.ensemble = init_single_and_ensemble(ensemble_size=5, model_class=DeepLinear)
        self.opt_single = torch.optim.SGD(params=self.single.parameters(), lr=self.lr)
        ensemble_param_list = []
        for member in self.ensemble:
            ensemble_param_list.extend(list(member.parameters()))
        self.opt_ensemble = torch.optim.SGD(params=ensemble_param_list, lr=self.lr)


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
        trainer = manager.Trainer(f'exp/exp3_linear/run{run}')
        trainers.append(trainer)
        p = Process(target=Trainer.train_self, args=(trainer, stopt, ))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()


