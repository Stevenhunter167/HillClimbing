import numpy as np
import code
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch


def converge(x, f, rtol=1e-5, atol=1e-8, iters=1e4, close=np.allclose):
    prev = None
    i = 0
    while prev is None or not close(x, prev, rtol=rtol, atol=atol):
        prev = x.copy()
        x = f(x)
        i += 1
        if i == iters:
            # warnings.warn('Maximum number of iterations reached (%d)' % iters)
            break
    return x


class Hill:

    def __init__(self, f, s_min=0, s_max=10,
                 step_size=1, slip_std=0.5, density=5, rstd=0.0, r_shift=0, gamma=0.95):

        self.density = density
        self.step_size = step_size
        self.slip_std = slip_std
        self.s_min, self.s_max = s_min, s_max
        self.f = f

        self.x = np.linspace(self.s_min, self.s_max, (self.s_max - self.s_min) * self.density)
        self.h = self.f(self.x)

        self.s = np.arange(self.x.shape[0])
        self.ns = self.s.shape[0]
        self.na = 2
        self.p = np.zeros((self.s.shape[0], self.na, self.s.shape[0]))
        self.er = np.zeros((self.s.shape[0], self.na))
        self.rstd = rstd
        self.gamma = gamma

        for si in self.s:
            for ai in range(self.na):
                direction = -1 * (1 - ai) + ai
                si_ideal = max(min(si + direction * self.density, self.s.shape[0] - 1), 0)
                assert (si_ideal in self.s, si_ideal), self.s.shape[0]

                ps_ = norm.pdf(
                    x=self.x,
                    loc=si_ideal / self.density + self.s_min,
                    scale=self.slip_std)
                ps_ = ps_ / ps_.sum()
                assert np.allclose(ps_.sum(), 1)
                self.p[si][ai] = ps_
                self.er[si][ai] = np.sum(
                    np.array([self.h[si_] - self.h[si] for si_ in self.s]) * ps_)

        self.er += r_shift
        self.q_ground_truth = self.plan()
        self.n = np.zeros(self.er.shape)

    def reset(self):
        self.n = np.zeros(self.er.shape)

    def sample(self, s=None, a=None):
        if s is None and a is None:
            s = np.random.randint(self.s.shape[0])
            a = np.random.randint(2)
        r = self.er[s][a] + np.random.normal(0, self.rstd)
        s_ = np.random.choice(np.arange(self.s.shape[0]), p=self.p[s][a])
        self.n[s][a] += 1
        return s, a, r, s_

    def sample_batch(self, batch_size):
        samples = [self.sample() for _ in range(batch_size)]
        s = torch.tensor([s for s, a, r, s_ in samples], dtype=torch.float32).view(-1, 1)
        a = torch.tensor([a for s, a, r, s_ in samples], dtype=torch.int64)
        a = torch.nn.functional.one_hot(a, num_classes=2)
        r = torch.tensor([r for s, a, r, s_ in samples], dtype=torch.float32).view(-1, 1)
        s_ = torch.tensor([s_ for s, a, r, s_ in samples], dtype=torch.float32).view(-1, 1)
        return s / self.s.shape[0], a, r, s_ / self.s.shape[0]

    def plan(self):
        q = np.zeros([self.ns, self.na])
        def backward(_q):
            v = _q.max(1)
            return self.er + self.gamma * self.p @ v
        return converge(q, backward)

    def render_coverage(self):
        plt.plot(self.x, self.n.sum(1), label='visits')

    def render(self):
        plt.plot(self.x, self.f(self.x), label='height(s)', color='brown')
        # plt.plot(self.x, self.p[-1][1], color='red', label="p(s'|s[-1],a=1)")
        # plt.plot(self.x, self.p[-1][0], color='orange', label="p(s'|s[-1],a=0)")
        self.plot_q()

    def plot_q(self, q=None, color=None):
        if q is not None:
            plt.plot(self.x, q[:, 1], label='Q_θ(s,1)', color=color)
            plt.plot(self.x, q[:, 0], label='Q_θ(s,0)', color=color, alpha=0.2)
        else:
            q = self.q_ground_truth
            # v = q.max(1)
            plt.plot(self.x, q[:,1], label='Q(s,1)', color='black')
            plt.plot(self.x, q[:,0], label='Q(s,0)', color='gray')


if __name__ == '__main__':
    def f(x):
        return norm.pdf(x, 10, 2)
    # def f(x):
    #     return 0.02 * x
    hill = Hill(f)
    plt.title('hill env')
    plt.xlabel('state')
    hill.render()
    plt.legend()
    plt.savefig("hill_simple")

