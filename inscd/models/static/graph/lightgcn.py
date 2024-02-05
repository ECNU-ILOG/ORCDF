import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF, DP_IF, MIRT_IF, MF_IF, KANCD_IF
from ....extractor import LIGHTGCN_Extractor


class LIGHTGCN(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        """
        Description:
        ULCDF ...

        Parameters:
        student_num: int type
            The number of students in the response logs
        exercise_num: int type
            The number of exercises in the response logs
        knowledge_num: int type
            The number of knowledge concepts in the response logs
        method: Ignored
            Not used, present here for API consistency by convention.
        """
        super().__init__(student_num, exercise_num, knowledge_num)

    def build(self, device: str = "cpu", gcn_layers: int = 3, if_type='dp-linear'
              , keep_prob=0.9,
              dtype=torch.float32, hidden_dims: list = None, **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        if if_type == 'kancd':
            latent_dim = 32
        else:
            latent_dim = self.knowledge_num


        self.extractor = LIGHTGCN_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
        )

        self.device = device
        if if_type == 'ncd':
            self.inter_func = NCD_IF(knowledge_num=self.knowledge_num,
                                     hidden_dims=hidden_dims,
                                     dropout=0,
                                     device=device,
                                     dtype=dtype)
        elif 'dp' in if_type:
            self.inter_func = DP_IF(knowledge_num=self.knowledge_num,
                                    hidden_dims=hidden_dims,
                                    dropout=0,
                                    device=device,
                                    dtype=dtype,
                                    kernel=if_type)
        elif 'mirt' in if_type:
            self.inter_func = MIRT_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=16,
                device=device,
                dtype=dtype,
                utlize=True)

        elif 'kancd' in if_type:
            self.inter_func = KANCD_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype,
                hidden_dims=hidden_dims,
                dropout=0.5
            )
        else:
            raise ValueError("Remain to be aligned....")

    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=5e-4, weight_decay=0.0005, batch_size=256):
        graph = self.__sp_mat_to_sp_tensor(self.__create_adj_mat(datahub["train"])).to(self.device)
        self.extractor.get_graph(graph)
        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa", 'ap']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr, "weight_decay": weight_decay},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr, "weight_decay": weight_decay}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self._train(datahub=datahub, set_type=set_type,
                        valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                        batch_size=batch_size, loss_func=loss_func, optimizer=optimizer)

    def predict(self, datahub: DataHub, set_type, batch_size=256, **kwargs):
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub: DataHub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
        if metrics is None:
            metrics = ["acc", "auc", "f1", "doa", 'ap']
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics, batch_size=batch_size)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def load(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.extractor.load_state_dict(torch.load(ex_path))
        self.inter_func.load_state_dict(torch.load(if_path))

    def save(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        torch.save(self.extractor.state_dict(), ex_path)
        torch.save(self.inter_func.state_dict(), if_path)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_mat(self, np_train):
        n_nodes = self.student_num + self.exercise_num
        train_stu = np_train[:, 0]
        train_exer = np_train[:, 1]
        ratings = np.ones_like(train_stu, dtype=np.float64)
        tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.student_num)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix
