import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor


class ULCDF_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9, activation='ELU', mode='all', **kwargs):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.mode = mode

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.gcn_drop = True
        self.activation = self.choose_activation(activation)
        self.graph_dict = ...

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

        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)
        self.concat_layer_1 = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)
        self.transfer_student_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_exercise_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_knowledge_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.apply(self.initialize_weights)

    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    @staticmethod
    def choose_activation(activation):
        if activation == 'LeakyReLU':
            return nn.LeakyReLU(negative_slope=0.8)
        elif activation == 'ELU':
            return nn.ELU()
        elif activation == 'ReLU':
            return nn.ReLU()
        elif activation == "GeLU":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def convolution(self, graph):
        stu_emb, exer_emb, know_emb = (self.__student_emb.weight,
                                       self.__exercise_emb.weight,
                                       self.__knowledge_emb.weight)
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        emb = [all_emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.__graph_drop(graph), all_emb)
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb

    def __common_forward(self):
        out_hol_emb, right_emb, wrong_emb = self.convolution(self.graph_dict['all']), self.convolution(
            self.graph_dict['right']), self.convolution(self.graph_dict['wrong'])
        if self.mode == 'hol':
            out_emb = self.activation(self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
                                      )
        elif self.mode == 'dis':
            out_emb = out_hol_emb
        else:
            out_dis_emb = self.activation(self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
                                          )
            out_emb = self.activation(self.concat_layer_1(torch.cat([out_dis_emb, out_hol_emb], dim=1))
                                      )

        student_ts, exercise_ts, knowledge_ts = torch.split(out_emb,
                                                            [self.student_num, self.exercise_num, self.knowledge_num])

        return student_ts, exercise_ts, knowledge_ts

    def __dropout(self, graph, keep_prob):
        if self.gcn_drop and self.training:
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

    def extract(self, student_id, exercise_id, q_mask):
        stu_forward, exer_forward, knows_forward = self.__common_forward()
        student_ts = self.transfer_student_layer(
            F.embedding(student_id, stu_forward)) if self.mode != 'tf' else F.embedding(student_id, stu_forward)
        diff_ts = self.transfer_exercise_layer(
            F.embedding(exercise_id, exer_forward)) if self.mode != 'tf' else F.embedding(exercise_id, exer_forward)
        knowledge_ts = self.transfer_knowledge_layer(knows_forward) if self.mode != 'tf' else knows_forward
        disc_ts = self.__disc_emb(exercise_id)

        return student_ts, diff_ts, disc_ts, knowledge_ts

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        stu_forward, exer_forward, know_forward = self.__common_forward()

        student_ts = self.transfer_student_layer(stu_forward) if self.mode != 'tf' else stu_forward
        diff_ts = self.transfer_exercise_layer(exer_forward) if self.mode != 'tf' else exer_forward
        knowledge_ts = self.transfer_knowledge_layer(know_forward) if self.mode != 'tf' else know_forward

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]
