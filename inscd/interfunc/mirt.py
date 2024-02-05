import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .._base import _InteractionFunction


class MIRT_IF(_InteractionFunction, nn.Module):
    def __init__(self, knowledge_num, latent_dim, device, dtype, utlize=False):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        self.ultize = utlize
        # self.linear_stu = nn.Linear(latent_dim, latent_dim, dtype=dtype).to(self.device)
        # self.linear_exer = nn.Linear(latent_dim, latent_dim, dtype=dtype).to(self.device)
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param)

    @staticmethod
    def irt2pl(theta, a, b, F=torch):
        return (1 / (1 + F.exp(- F.sum(F.multiply(a, theta), dim=-1) + b))).view(-1)

    #
    # def irf(self, theta, a, b, D=1.702):
    #     return torch.sigmoid(torch.mean(D * a * (theta - b), dim=1)).to(self.device).view(-1)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        student_ts = torch.squeeze(student_ts, dim=-1)
        diff_ts = torch.squeeze(diff_ts, dim=-1)
        disc_ts = torch.squeeze(disc_ts, dim=-1)
        return self.irt2pl(student_ts, F.softplus(diff_ts), disc_ts)

        # if self.ultize:
        #     return torch.sigmoid((self.linear_stu(student_ts) * self.linear_exer(diff_ts)).sum(dim=1, keepdim=True) + disc_ts).view(-1)
        # else:
        # return torch.sigmoid((student_ts * diff_ts).sum(dim=1, keepdim=True) + disc_ts).view(-1)

        # return self.irt2pl(student_ts, diff_ts, disc_ts)
        # else:
        #     return self.irf(torch.sigmoid(student_ts), torch.sigmoid(disc_ts),
        #                     torch.sigmoid(diff_ts))
        # if student_ts.shape[1] == knowledge_ts.shape[0]:
        #     return self.irt2pl(torch.sigmoid(student_ts) * q_mask, torch.sigmoid(diff_ts) * q_mask, torch.sigmoid(disc_ts).view(-1))
        # else:
        #     return self.irt2pl(torch.sigmoid(student_ts), torch.sigmoid(diff_ts), torch.sigmoid(disc_ts).view(-1))

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)
