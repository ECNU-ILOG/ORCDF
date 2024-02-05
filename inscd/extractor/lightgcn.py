import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor


class LIGHTGCN_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }

        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_graph(self, graph):
        self.graph = graph

    def __dropout(self, graph, keep_prob):
        if self.training:
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.DoubleTensor(index.t(), values, size)
            return g
        else:
            return graph

    def __graph_drop(self, graph):
        g_dropped = self.__dropout(graph, self.keep_prob)
        return g_dropped

    def __common_forward(self):
        stu_emb = self.__student_emb.weight
        exer_emb = self.__exercise_emb.weight
        all_emb = torch.cat([stu_emb, exer_emb]).to(self.device)
        embs = [all_emb]
        for layer in range(self.gcn_layers):
            if isinstance(self.graph, list):
                g_droped = self.__graph_drop(self.graph[layer])
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            else:
                g_droped = self.__graph_drop(self.graph)
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        stus, exers = torch.split(light_out, [self.student_num, self.exercise_num])
        return stus, exers

    def extract(self, student_id, exercise_id, q_mask):
        stu_forward, exer_forward = self.__common_forward()
        student_ts = F.embedding(student_id, stu_forward)
        diff_ts = F.embedding(exercise_id, exer_forward)
        disc_ts = self.__disc_emb(exercise_id)
        knowledge_ts = self.__knowledge_emb.weight
        return student_ts, diff_ts, disc_ts, knowledge_ts

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        stu_forward, exer_forward = self.__common_forward()
        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = stu_forward
        self.__emb_map["diff"] = exer_forward
        self.__emb_map["disc"] = disc_ts
        return self.__emb_map[item]
