import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction


class NCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num: int, hidden_dims: list, dropout, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

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

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        q_mask = kwargs["q_mask"]
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(student_ts) - torch.sigmoid(diff_ts)) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)
