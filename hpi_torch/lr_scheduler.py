from torch import optim
import numpy as np

def get_multilrs(milestones, mode='multiply'):
    lrs = [step[0] for step in milestones]
    steps = [step[1] for step in milestones]
    multi_lrs = []
    if mode == 'multiply':
        for i in range(len(steps) - 1):
            gamma = np.float_power(lrs[i+1] / lrs[i], 1.0 / (steps[i+1] - steps[i]))
            lr = lrs[i]
            for i in range(steps[i+1] - steps[i]):
                multi_lrs.append(lr)
                lr = lr * gamma
    else:
        for i in range(len(steps) - 1):
            delta = (lrs[i+1] - lrs[i]) / (steps[i + 1] - steps[i])
            lr = lrs[i]
            for i in range(steps[i + 1] - steps[i]):
                multi_lrs.append(lr)
                lr = lr + delta
    return multi_lrs


class MultiStepLR(optim.lr_scheduler.MultiStepLR):
    """
    milestones: list of (lr, step)
    """
    def __init__(self, optimizer, milestones, last_epoch=-1, mode='multiply'):
        steps = [s[1] for s in milestones]
        self.lrs = get_multilrs(milestones, mode)
        super(MultiStepLR, self).__init__(optimizer, steps, 0.1, last_epoch)

    def get_lr(self):
        if self.last_epoch < len(self.lrs):
            return [self.lrs[self.last_epoch] for base_lr in self.base_lrs]
        else:
            return [self.lrs[-1] for base_lr in self.base_lrs]


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    milestones = [(2e-5, 0), (1e-2, 10), (1e-3, 40), (1e-4, 50), (5e-5, 60), (1e-4, 70), (1e-5, 80), (5e-5, 90),
                  (1e-6, 100)]
    plt.plot(get_multilrs(milestones))
    plt.show()