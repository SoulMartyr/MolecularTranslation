from collections import defaultdict

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not lr >= 0.0:
            raise ValueError("Learning Rate Invaild: {}".format(lr))
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError("First Order Moment Ratio Invaild: {}".format(betas[0]))
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError("Second Order Moment Ratio Invaild: {}".format(betas[1]))
        if not eps >= 0.0:
            raise ValueError("Correction Ratio Invaild: {}".format(eps))
        if not weight_decay >= 0.0:
            raise ValueError("L2 Regularization Ratio Invaild: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.state = defaultdict(dict)
        self.sma_max = 2 / (1 - betas[1]) - 1
        self.rt_buffer = [[None, None] for _ in range(50)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = None

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data.float()

                if grad.is_sparse:
                    raise ValueError('RAdam does not support sparse gradients')

                param_data = param.data.float()

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 1
                    state["moment_1st"] = torch.zeros_like(param_data)
                    state["moment_2nd"] = torch.zeros_like(param_data)
                else:
                    state["moment_1st"] = state["moment_1st"].type_as(param_data)
                    state["moment_2nd"] = state["moment_2nd"].type_as(param_data)

                lr = param_group["lr"]
                t = state["step"]
                moment_1st = state["moment_1st"]
                moment_2nd = state["moment_2nd"]
                beta1, beta2 = param_group["betas"]

                moment_1st.mul_(beta1).add_(grad, alpha=1 - beta1)
                moment_2nd.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                beta1_t = beta1 ** t
                beta2_t = beta2 ** t
                sma_max = self.sma_max
                sma = sma_max - 2 * t * beta2_t / (1 - beta2)
                bc_moment_1st = moment_1st / (1 - beta1_t)

                if param_group["weight_decay"] != 0:
                    param_data.add_(-param_group["weight_decay"] * lr)

                buffer = self.rt_buffer[int(state['step'] % 50)]
                if sma >= 5:
                    if t == buffer[0]:
                        r_t = buffer[1]
                    else:
                        buffer[0] = t
                        r_t = np.sqrt((1 - beta2_t) * (sma - 4) / (sma_max - 4) * (sma - 2) / (
                                sma_max - 2) * sma_max / sma)
                        buffer[1] = [r_t]
                    bc_moment_2nd = np.sqrt(moment_2nd / (1 - beta2_t)) + param_group["eps"]
                    bc_moment_1st.add_(r_t)
                    param_data.addcdiv_(bc_moment_1st, bc_moment_2nd.add_(param_group["eps"]), value=lr * r_t)
                else:
                    param_data.add_(-bc_moment_1st, alpha=lr)

                param.data.copy_(param_data)

        return loss
