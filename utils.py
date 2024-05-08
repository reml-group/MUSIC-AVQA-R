import logging
import torch

class Logger(object):
    def __init__(self, config):
        self.logger = logging.getLogger(name='RAVQA')
        self.logger.setLevel(level=logging.INFO)
        file_handler = logging.FileHandler(filename=config['path']['log_name'])
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s ")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_log(self):
        return self.logger

class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)