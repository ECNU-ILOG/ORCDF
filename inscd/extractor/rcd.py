import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor


class GraphLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, dtype=torch.float32):
        super(GraphLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False, dtype=dtype)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False, dtype=dtype)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class Fusion(nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, local_map, device: str, dtype=torch.float32):
        self.knowledge_num = knowledge_num
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.device = device

        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.s_from_e = local_map['s_from_e'].to(self.device)
        self.e_from_s = local_map['e_from_s'].to(self.device)

        super(Fusion, self).__init__()
        self.k_from_e = GraphLayer(self.k_from_e, self.knowledge_num, self.knowledge_num, dtype=dtype).to(self.device)
        self.e_from_k = GraphLayer(self.e_from_k, self.knowledge_num, self.knowledge_num, dtype=dtype).to(self.device)

        self.s_from_e = GraphLayer(self.s_from_e, self.knowledge_num, self.knowledge_num, dtype=dtype).to(self.device)
        self.e_from_s = GraphLayer(self.e_from_s, self.knowledge_num, self.knowledge_num, dtype=dtype).to(self.device)

        self.k_attn_fc1 = nn.Linear(2 * self.knowledge_num, 1, bias=True, dtype=dtype).to(self.device)
        self.k_attn_fc2 = nn.Linear(2 * self.knowledge_num, 1, bias=True, dtype=dtype).to(self.device)
        self.k_attn_fc3 = nn.Linear(2 * self.knowledge_num, 1, bias=True, dtype=dtype).to(self.device)

        self.e_attn_fc1 = nn.Linear(2 * self.knowledge_num, 1, bias=True, dtype=dtype).to(self.device)
        self.e_attn_fc2 = nn.Linear(2 * self.knowledge_num, 1, bias=True, dtype=dtype).to(self.device)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)

        k_from_e_graph = self.k_from_e(e_k_graph)
        e_from_k_graph = self.e_from_k(e_k_graph)

        e_s_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.s_from_e(e_s_graph)
        e_from_u_graph = self.e_from_s(e_s_graph)

        # update concepts
        A = kn_emb
        D = k_from_e_graph[self.exercise_num:]

        concat_c_3 = torch.cat([A, D], dim=1)

        score3 = self.k_attn_fc3(concat_c_3)
        score3 = F.softmax(score3, dim=1)

        kn_emb = A + score3[:, 0].unsqueeze(1) * D

        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0: self.exercise_num]
        C = e_from_u_graph[0: self.exercise_num]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)  # dim = 1, 按行SoftMax, 行和为1
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated students
        all_stu_emb = all_stu_emb + u_from_e_graph[self.exercise_num:]

        return kn_emb, exer_emb, all_stu_emb


class RCD_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, if_type='rcd'):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.if_type = if_type
        self.latent_dim = latent_dim

        self.__student_emb = nn.Embedding(self.student_num, latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }
        self.k_index = torch.LongTensor(list(range(self.knowledge_num))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.student_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exercise_num))).to(self.device)

        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_local_map(self, local_map):
        self.local_map = local_map
        self.k_from_e = self.local_map['k_from_e'].to(self.device)
        self.e_from_k = self.local_map['e_from_k'].to(self.device)
        self.s_from_e = self.local_map['s_from_e'].to(self.device)
        self.e_from_s = self.local_map['e_from_s'].to(self.device)
        self.FusionLayer1 = Fusion(self.student_num, self.exercise_num, self.latent_dim, self.local_map, self.device, self.dtype)
        self.FusionLayer2 = Fusion(self.student_num, self.exercise_num, self.latent_dim, self.local_map, self.device, self.dtype)

    def __common_forward(self):
        all_stu_emb = self.__student_emb(self.stu_index).to(self.device)
        exer_emb = self.__exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.__knowledge_emb(self.k_index).to(self.device)
        # Fusion layer 1
        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)
        return all_stu_emb2, exer_emb2, kn_emb2

    def extract(self, student_id, exercise_id, q_mask):
        stu_forward, exer_forward, knows_forward = self.__common_forward()
        batch_stu_emb = stu_forward[student_id]
        batch_exer_emb = exer_forward[exercise_id]
        disc_ts = self.__disc_emb(exercise_id)

        if self.if_type == 'rcd':
            batch_stu_ts = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])

            # get batch exercise data
            batch_exer_ts = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0],
                                                                                      batch_exer_emb.shape[1],
                                                                                      batch_exer_emb.shape[1])

            # get batch knowledge concept data
            knowledge_ts = knows_forward.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0],
                                                                                   knows_forward.shape[0],
                                                                                   knows_forward.shape[1])
        else:
            batch_stu_ts = batch_stu_emb
            batch_exer_ts = batch_exer_emb
            knowledge_ts = knows_forward

        return batch_stu_ts, batch_exer_ts, disc_ts, knowledge_ts

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        stu_forward, exer_forward, knows_forward = self.__common_forward()
        if self.if_type == 'rcd':
            student_ts = stu_forward.repeat(1, stu_forward.shape[1]).reshape(stu_forward.shape[0],
                                                                                   stu_forward.shape[1],
                                                                                   stu_forward.shape[1])

            # get batch exercise data
            diff_ts = exer_forward.repeat(1, exer_forward.shape[1]).reshape(exer_forward.shape[0],
                                                                                      exer_forward.shape[1],
                                                                                      exer_forward.shape[1])

            # get batch knowledge concept data
            knowledge_ts = knows_forward.repeat(stu_forward.shape[0], 1).reshape(stu_forward.shape[0],
                                                                                   knows_forward.shape[0],
                                                                                   knows_forward.shape[1])
        else:
            student_ts = stu_forward
            diff_ts = exer_forward
            knowledge_ts = knows_forward

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]
