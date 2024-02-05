import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction


class CDMFKC_IF(_InteractionFunction, nn.Module):
    def __init__(self, g_impact_a, g_impact_b, knowledge_num: int, hidden_dims: list, dropout, device,
                 dtype, latent_dim=None):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.g_impact_a = g_impact_a
        self.g_impact_b = g_impact_b
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.latent_dim = latent_dim
        if latent_dim is not None:
            self.transform_impact = nn.Linear(latent_dim, knowledge_num, dtype=dtype).to(self.device)

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
        if self.latent_dim is not None:
            h_impact = self.transform_impact(torch.sigmoid(kwargs['other']['knowledge_impact']))
        else:
            h_impact = torch.sigmoid(kwargs['other']['knowledge_impact'])

        g_impact = torch.sigmoid(self.g_impact_a * h_impact +
                                 self.g_impact_b * torch.sigmoid(diff_ts) * torch.sigmoid(disc_ts))
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(student_ts) + g_impact - torch.sigmoid(diff_ts)) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)
