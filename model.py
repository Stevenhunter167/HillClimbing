import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 2)
    def forward(self, x):
        return self.net(x)


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 2)
    def forward(self, x):
        res = self.net(x)
        return res + res ** 2

class DeepLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,2),
            nn.Linear(2,2),
            nn.Linear(2,2),
            nn.Linear(2,2),
        )
    def forward(self, x):
        return self.net(x)

class DeepRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,2),
            nn.ReLU(),
            nn.Linear(2,2),
            nn.ReLU(),
            nn.Linear(2,2),
            nn.ReLU(),
            nn.Linear(2,2),
        )
    def forward(self, x):
        return self.net(x)


def flatten_param(model):
    return torch.cat([param.detach().view(-1) for param in model.parameters()])


def init_single_and_ensemble(ensemble_size, model_class=Linear):
    """ initialize single model and ensemble model, s.t. param_mean = single """
    single = model_class()
    ensemble = [model_class() for _ in range(ensemble_size)]
    param = single.state_dict()

    # mean param
    ensemble_params = {k: 0 for k in param}
    for k in range(len(ensemble)):
        for param_name, value in ensemble[k].state_dict().items():
            ensemble_params[param_name] += value
    for param_name in ensemble_params:
        ensemble_params[param_name] /= ensemble_size

    # param diff
    ensemble_diff = {}
    for param_name in ensemble_params:
        ensemble_diff[param_name] = param[param_name] - ensemble_params[param_name]

    # shift param s.t. ensemble param mean = single
    for model in ensemble:
        state_dict = model.state_dict()
        for param_name in state_dict:
            state_dict[param_name] += ensemble_diff[param_name]
        model.load_state_dict(state_dict)

    # verify
    for param, value in single.state_dict().items():
        ensemble_param = 0
        for model in ensemble:
            ensemble_param += model.state_dict()[param]
        ensemble_param /= ensemble_size
        assert torch.allclose(ensemble_param, value), 'error'

    return single, ensemble

def visualize_param(single, ensemble):
    print("visualize param")
    print("single", flatten_param(single))
    sum_of_ensemble_param = 0
    for model in ensemble:
        member_param = flatten_param(model)
        print(member_param)
        sum_of_ensemble_param += member_param
    print("param mean", sum_of_ensemble_param / len(ensemble))
    print("visualize param end")


if __name__ == '__main__':
    single, ensemble = (init_single_and_ensemble(5))
    visualize_param(single, ensemble)
