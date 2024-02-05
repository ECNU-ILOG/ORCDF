import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction


class KANCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num: int, latent_dim: int, hidden_dims: list, dropout, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.k_diff_full = nn.Linear(self.latent_dim, 1, dtype=dtype).to(self.device)
        self.stat_full = nn.Linear(self.latent_dim, 1, dtype=dtype).to(self.device)

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
        knowledge_ts = kwargs['knowledge_ts']
        q_mask = kwargs["q_mask"]

        batch, dim = student_ts.size()
        stu_emb = student_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.knowledge_num, -1)
        exer_emb = diff_ts.view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
        input_x = torch.sigmoid(disc_ts) * (torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
                                            - torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)) * q_mask
        return self.mlp(input_x).view(-1)

    def transform(self, mastery, knowledge):
        self.eval()
        blocks = torch.split(torch.arange(mastery.shape[0]).to(device=self.device), 5)
        mas = []
        for block in blocks:
            batch, dim = mastery[block].size()
            stu_emb = mastery[block].view(batch, 1, dim).repeat(1, self.knowledge_num, 1)
            knowledge_emb = knowledge.repeat(batch, 1).view(batch, self.knowledge_num, -1)
            mas.append(torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1))
        return torch.vstack(mas)

    def monotonicity(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.apply(none_neg_clipper)
