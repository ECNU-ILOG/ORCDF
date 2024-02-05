import torch
class _NoneNegClipper(object):
    def __init__(self):
        super(_NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)



none_neg_clipper = _NoneNegClipper()