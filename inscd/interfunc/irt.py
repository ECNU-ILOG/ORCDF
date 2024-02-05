import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _InteractionFunction


class IRT_IF(_InteractionFunction, nn.Module):
    def __init__(self, device, dtype, latent_dim=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.latent_dim = latent_dim
        if self.latent_dim is not None:
            self.transform_student = nn.Linear(latent_dim, 1, dtype=dtype).to(self.device)
            self.transform_exercise = nn.Linear(latent_dim, 1, dtype=dtype).to(self.device)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        if self.latent_dim is not None:
            input_x = torch.sigmoid(self.transform_student(student_ts) - self.transform_exercise(diff_ts))
        else:
            input_x = torch.sigmoid(student_ts - diff_ts)
        return input_x.view(-1)

    def transform(self, mastery, knowledge):
        return F.sigmoid(mastery)
