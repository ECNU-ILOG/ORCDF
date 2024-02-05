import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction


def dot_product(A, B):

    # A = F.normalize(A, p=2, dim=1)
    # B = F.normalize(B, p=2, dim=1)
    # return torch.sigmoid(F.cosine_similarity(A.unsqueeze(1) / 1.5, B.unsqueeze(0)/1.5, dim=2))
    return torch.sigmoid(torch.matmul(A, B.T))


def polynomial_kernel_dot_product(A, B, **kwargs):
    constant = kwargs.get('constant', 0.1)
    degree = kwargs.get('degree', 2.0)
    return F.sigmoid((torch.matmul(A, B.T) + constant) ** degree)


def rbf_kernel_dot_product(A, B, **kwargs):
    sigma = kwargs.get('sigma', 0.1)
    return F.sigmoid(torch.exp(-torch.square(torch.norm(A[:, None] - B, dim=2)) / (2 * sigma ** 2)))


class DP_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num: int, hidden_dims: list, dropout, device, dtype, kernel):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.kernel = kernel
        self.transform_kernel = self.get_kernel()
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.knowledge_num, hidden_dim, dtype=self.dtype),
                        'activation0': nn.Tanh()
                    }
                )
            else:
                layers.update(
                    {
                        'dropout{}'.format(idx): nn.Dropout(p=self.dropout),
                        'linear{}'.format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim, dtype=self.dtype),
                        'activation{}'.format(idx): nn.Tanh()
                    }
                )
        layers.update(
            {
                'dropout{}'.format(len(self.hidden_dims)): nn.Dropout(p=self.dropout),
                'linear{}'.format(len(self.hidden_dims)): nn.Linear(
                    self.hidden_dims[len(self.hidden_dims) - 1], 1, dtype=self.dtype
                ),
                'activation{}'.format(len(self.hidden_dims)): nn.Sigmoid()
            }
        )

        self.mlp = nn.Sequential(layers).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def get_kernel(self):
        if self.kernel == 'dp-linear':
            return dot_product
        elif self.kernel == 'dp-poly':
            return polynomial_kernel_dot_product
        elif self.kernel == 'dp-rbf':
            return rbf_kernel_dot_product
        else:
            raise ValueError('We don not support such kernel')

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        knowledge_ts = kwargs["knowledge_ts"]
        q_mask = kwargs["q_mask"]
        input_x = torch.sigmoid(disc_ts) * (
                self.transform_kernel(student_ts, knowledge_ts) - self.transform_kernel(diff_ts, knowledge_ts)
        ) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(self.transform_kernel(mastery, knowledge))

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)
