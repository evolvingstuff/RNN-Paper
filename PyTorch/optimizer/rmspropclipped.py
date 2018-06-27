import torch
from torch.optim.optimizer import Optimizer

class RMSpropclipped(Optimizer):
    """Modification of the default RMSprop implemnetation in PyTorch
       to allow gradient clipping in numerator"""

    def __init__(self, params, lr=1e-3, alpha=0.999, eps=1e-8, weight_decay=0.001, clip=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= clip:
            raise ValueError("Invalid clip value: {}".format(clip))

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, clip=clip)
        super(RMSpropclipped, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropclipped, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                clipped_grad = torch.clamp(grad, -group['clip'], group['clip'])

                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps']) #TODO: eps added outside of sqrt?
                p.data.addcdiv_(-group['lr'], clipped_grad, avg)

        return loss