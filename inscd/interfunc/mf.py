import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from ._util import none_neg_clipper
from .._base import _InteractionFunction


class MF_IF(_InteractionFunction, nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        disc_ts = kwargs["disc_ts"]
        q_mask = kwargs["q_mask"]
        input_x = torch.sigmoid(torch.sigmoid(disc_ts) * ((student_ts * q_mask) @ (diff_ts * q_mask).T))
        return input_x.view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)

    def monotonicity(self):
        return
