import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import _InteractionFunction
from ._util import none_neg_clipper


class KSCD_IF(_InteractionFunction, nn.Module):
    def __init__(self, dropout, knowledge_num: int, latent_dim: int, device, dtype):
        super().__init__()
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        # self.disc_mlp = nn.Linear(self.latent_dim, 1, dtype=self.dtype).to(self.device)
        # self.f_sk = nn.Linear(self.knowledge_num + self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        # self.f_ek = nn.Linear(self.knowledge_num + self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        # self.f_se = nn.Linear(self.knowledge_num, 1, dtype=self.dtype).to(self.device)

        self.prednet_full1 = nn.Linear(self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False,
                                       dtype=dtype).to(self.device)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = nn.Linear(self.knowledge_num + self.latent_dim, self.knowledge_num, bias=False,
                                       dtype=dtype).to(self.device)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = nn.Linear(1 * self.knowledge_num, 1, dtype=dtype).to(self.device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def compute(self, **kwargs):
        student_ts = kwargs["student_ts"]
        diff_ts = kwargs["diff_ts"]
        q_mask = kwargs["q_mask"]
        knowledge_ts = kwargs['knowledge_ts']
        # stu_ability = torch.sigmoid(student_ts @ knowledge_ts.T)
        # diff_emb = torch.sigmoid(diff_ts @ knowledge_ts.T)
        # disc_ts = torch.sigmoid(self.disc_mlp(diff_ts))
        # batch, dim = student_ts.size()
        # stu_emb = stu_ability.unsqueeze(1).repeat(1, self.knowledge_num, 1)
        # diff_emb = diff_emb.unsqueeze(1).repeat(1, self.knowledge_num, 1)
        # Q_relevant = q_mask.unsqueeze(2).repeat(1, 1, self.knowledge_num)
        # knowledge_emb = knowledge_ts.repeat(batch, 1).view(batch, self.knowledge_num, -1)
        # s_k_concat = torch.sigmoid(self.f_sk(torch.cat([stu_emb, knowledge_emb], dim=-1)))
        # e_k_concat = torch.sigmoid(self.f_ek(torch.cat([diff_emb, knowledge_emb], dim=-1)))
        # return torch.sigmoid(disc_ts * self.f_se(torch.mean((s_k_concat - e_k_concat) * Q_relevant, dim=1))).view(-1)
        stu_ability = torch.mm(student_ts, knowledge_ts.T).sigmoid()
        exer_diff = torch.mm(diff_ts, knowledge_ts.T).sigmoid()
        batch_stu_vector = stu_ability.repeat(1, self.knowledge_num).reshape(stu_ability.shape[0], self.knowledge_num,
                                                                             stu_ability.shape[1])
        batch_exer_vector = exer_diff.repeat(1, self.knowledge_num).reshape(exer_diff.shape[0], self.knowledge_num,
                                                                            exer_diff.shape[1])

        kn_vector = knowledge_ts.repeat(stu_ability.shape[0], 1).reshape(stu_ability.shape[0], self.knowledge_num,
                                                                         self.latent_dim)

        # CD
        preference = torch.tanh(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.tanh(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * q_mask.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(q_mask, dim=1).unsqueeze(1)
        y_pd = sum_out / count_of_concept
        return y_pd.view(-1)

    def transform(self, mastery, knowledge):
        stu_mastery = torch.mm(mastery, knowledge.T).sigmoid()
        stu_vector = stu_mastery.repeat(1, self.knowledge_num).reshape(stu_mastery.shape[0], self.knowledge_num,
                                                                             stu_mastery.shape[1])
        kn_vector = knowledge.repeat(stu_mastery.shape[0], 1).reshape(stu_mastery.shape[0], self.knowledge_num,
                                                                         self.latent_dim)
        preference = torch.tanh(self.prednet_full1(torch.cat((stu_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference))
        return o.squeeze(-1)

    def monotonicity(self):
        self.prednet_full1.apply(none_neg_clipper)
        self.prednet_full2.apply(none_neg_clipper)
        self.prednet_full3.apply(none_neg_clipper)
