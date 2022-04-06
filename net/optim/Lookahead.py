import itertools as it

from torch.optim import SGD
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(self, optimizer=SGD, alpha=0.5, k=5):

        if not issubclass(optimizer, Optimizer):
            raise ValueError("Optimizer Invaild: {}".format(optimizer))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Slow Update Rate Invaild: {}".format(alpha))
        if not k >= 1:
            raise ValueError("Lookahead Period Invaild: {}".format(k))

        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        for group in self.param_groups:
            group["step"] = 0

        self.alpha = alpha
        self.k = k
        # slow weights don't need to calculate gradient
        self.slow_weights = [[param.clone().detach() for param in param_group['params']] for param_group in
                             self.param_groups]
        for w in it.chain(self.slow_weights):
            w.requires_grad = False
        self.state = optimizer.state

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()

        for param_group, slow_weights in zip(self.param_groups, self.slow_weights):
            param_group["step"] += 1
            if param_group["step"] % self.k != 0:
                continue
            for fast_weight, slow_weight in zip(param_group["params"], slow_weights):
                if fast_weight.grad is None:
                    continue
                slow_weight.add_(fast_weight.data - slow_weight.data, alpha=self.alpha)
                fast_weight.copy_(slow_weight.data)
        return loss
