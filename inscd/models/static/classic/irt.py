import warnings
import torch
import scipy.stats as ss
import numpy as np
import torch.nn as nn
from inscd._base import _CognitiveDiagnosisModel, _InteractionFunction
from inscd.datahub import DataHub


class StatisticsIRT(_InteractionFunction):
    def __init__(self, student_num, exercise_num, knowledge_num):
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num

        self.alpha = ss.norm.rvs(loc=0.75, scale=0.01, size=(exercise_num, 1))
        self.beta = ss.norm.rvs(size=(exercise_num, 1))
        self.gamma = ss.norm.rvs(size=exercise_num)

    def fit(self, response_logs_data, *args, **kwargs):
        pass

    def compute(self, response_logs_data):
        pass

    def __getitem__(self, item):
        pass


class NeuralIRT(_InteractionFunction, nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num, D, device, dtype):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.device = device
        self.dtype = dtype
        self.D = D

        self.theta = nn.Embedding(self.student_num, 1, dtype=dtype)
        self.disc = nn.Embedding(self.exercise_num, 1, dtype=dtype)
        self.diff = nn.Embedding(self.exercise_num, 1, dtype=dtype)
        self.guess = nn.Embedding(self.exercise_num, 1, dtype=dtype)

    def forward(self, student_id, exercise_id):
        # TODO not a good way to write, change it
        theta = torch.squeeze(self.theta(student_id), dim=-1).to(self.device)
        disc = torch.squeeze(self.disc(exercise_id), dim=-1).to(self.device)
        diff = torch.squeeze(self.diff(exercise_id), dim=-1).to(self.device)
        guess = torch.squeeze(self.guess(exercise_id), dim=-1).to(self.device)
        disc = nn.functional.softplus(disc)
        # D = 1.702
        return guess + (1 - guess) / (1 + torch.exp(-self.D * disc * (theta - diff)))

    def fit(self, response_logs_data, *args, **kwargs):
        pass

    def compute(self, response_logs_data):
        pass

    def __getitem__(self, item):
        pass


# class IRTNet(nn.Module):
#     def __init__(self, user_num, item_num, value_range, a_range, irf_kwargs=None):
#         super(IRTNet, self).__init__()
#         self.user_num = user_num
#         self.item_num = item_num
#         self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
#         self.theta = nn.Embedding(self.user_num, 1)
#         self.a = nn.Embedding(self.item_num, 1)
#         self.b = nn.Embedding(self.item_num, 1)
#         self.c = nn.Embedding(self.item_num, 1)
#         self.value_range = value_range
#         self.a_range = a_range
#
#     def forward(self, user, item):
#         theta = torch.squeeze(self.theta(user), dim=-1)
#         a = torch.squeeze(self.a(item), dim=-1)
#         b = torch.squeeze(self.b(item), dim=-1)
#         c = torch.squeeze(self.c(item), dim=-1)
#         c = torch.sigmoid(c)
#         if self.value_range is not None:
#             theta = self.value_range * (torch.sigmoid(theta) - 0.5)
#             b = self.value_range * (torch.sigmoid(b) - 0.5)
#         if self.a_range is not None:
#             a = self.a_range * torch.sigmoid(a)
#         else:
#             a = F.softplus(a)
#         if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
#             raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
#         return self.irf(theta, a, b, c, **self.irf_kwargs)
#
#     @classmethod
#     def irf(cls, theta, a, b, c, **kwargs):
#         return irt3pl(theta, a, b, c, F=torch, **kwargs)

class ItemResponseTheory(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, method: str):
        """
        Description:
        IRT is an ...

        Parameters:
        student_num: the number of students in the response logs
        exercise_num: the number of exercises in the response logs
        knowledge_num: since IRT is unidimensional model, this parameter will be ignored. Keep this merely to match
        the requirement of interface
        method: the way to estimate parameters in this model, we provide ["neural", "em"]
        """
        super().__init__(student_num, exercise_num, knowledge_num, method)
        if self.method not in ["neural", "em"]:
            raise ValueError("Do not support method {}. Currently we only support \"neural\", "
                             "\"em\" and \"grad\"".format(self.method))

    def build(self, device=None, dtype=torch.float32):
        if self.method == "neural" and device == "cuda" and not torch.cuda.is_available():
            warnings.warn("We find that you try to use \"neural\" method and \"cuda\" device to build IRT interaction "
                          "function, but \"cuda\" is not available. We have set it as \"cpu\".")

        if self.method == "neural" and device is None:
            warnings.warn("We find that you try to use \"neural\" method to build IRT interaction function but forget"
                          "pass parameter \"device\". We have set it as \"cpu\".")


    def train(self, response_logs: DataHub, set_type, valid_set_type=None, valid_metrics=None, *args, **kwargs):
        pass

    def predict(self, response_logs: DataHub, set_type):
        pass

    def score(self, response_logs: DataHub, set_type, metrics: list) -> dict:
        pass

    def diagnose(self):
        pass

    def load(self, path: str):
        pass

    def save(self, name: str, path: str):
        pass