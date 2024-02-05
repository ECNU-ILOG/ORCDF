import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF, DP_IF, MIRT_IF, MF_IF, KANCD_IF, CDMFKC_IF, IRT_IF, KSCD_IF
from ....extractor import ORCDF_Extractor

from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score


class ORCDF(_CognitiveDiagnosisModel, AbstractModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, save_flag=False):
        """
        Description:
        SSCDF ...

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
        super().__init__(student_num, exercise_num, knowledge_num, save_flag)

    def build(self, latent_dim=32, device: str = "cpu", gcn_layers: int = 3, if_type='dp-linear'
              , keep_prob=0.9,
              dtype=torch.float64, hidden_dims: list = None, mode='all', flip_ratio=0.1, ssl_temp=0.8, ssl_weight=1e-2,
              **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.device = device
        self.mode = mode
        self.flip_ratio = flip_ratio

        # if 'tf' in self.mode:
        #     if if_type != 'kancd':
        #         latent_dim = self.knowledge_num
        # if if_type == 'kancd' or if_type == 'kscd' or if_type == 'mirt' or if_type == 'irt':
        #     latent_dim = 32
        # else:
        #     latent_dim = self.knowledge_num

        self.extractor = SSCDF_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
            mode=mode,
            ssl_temp=ssl_temp,
            ssl_weight=ssl_weight
        )

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
                latent_dim=32,
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
        elif 'cdmfkc' in if_type:
            self.inter_func = CDMFKC_IF(
                g_impact_a=0.5,
                g_impact_b=0.5,
                knowledge_num=self.knowledge_num,
                hidden_dims=hidden_dims,
                dropout=0.5,
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )
        elif 'irt' in if_type:
            self.inter_func = IRT_IF(
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )
        elif 'kscd' in if_type:
            self.inter_func = KSCD_IF(
                dropout=0.5,
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype)
        else:
            raise ValueError("Remain to be aligned....")

    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=5e-4, weight_decay=0.0005, batch_size=256):

        if self.mode == 'Q':
            ek_graph = np.zeros(shape=datahub.q_matrix.shape)
        else:
            ek_graph = datahub.q_matrix.copy()

        se_graph_right, se_graph_wrong = [self.__create_adj_se(datahub[set_type], is_subgraph=True)[i] for i in
                                          range(2)]
        se_graph = self.__create_adj_se(datahub[set_type], is_subgraph=False)

        if self.flip_ratio:
            def get_flip_data():
                np_response_flip = datahub[set_type].copy()
                column = np_response_flip[:, 2]
                probability = np.random.choice([True, False], size=column.shape,
                                               p=[self.flip_ratio, 1 - self.flip_ratio])
                column[probability] = 1 - column[probability]
                np_response_flip[:, 2] = column
                return np_response_flip

        graph_dict = {
            'right': self.__final_graph(se_graph_right, ek_graph),
            'wrong': self.__final_graph(se_graph_wrong, ek_graph),
            'response': datahub[set_type],
            'Q_Matrix': datahub.q_matrix.copy(),
            'flip_ratio': self.flip_ratio,
            'all': self.__final_graph(se_graph, ek_graph)
        }

        self.extractor.get_graph_dict(graph_dict)
        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa", 'ap']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr, "weight_decay": weight_decay},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr, "weight_decay": weight_decay}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self.extractor.get_flip_graph()
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

    def get_attribute(self, attribute_name):
        if attribute_name == 'mastery':
            return self.diagnose().detach().cpu().numpy()
        elif attribute_name == 'diff':
            return self.inter_func.transform(self.extractor["diff"],
                                             self.extractor["knowledge"]).detach().cpu().numpy()
        elif attribute_name == 'knowledge':
            return self.extractor["knowledge"].detach().cpu().numpy()
        else:
            return None

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

    def update_graph(self, data, q_matrix):
        se_graph_right, se_graph_wrong = [self.__create_adj_se(data, is_subgraph=True)[i] for i in
                                          range(2)]
        self.extractor.graph_dict['right'] = self.__final_graph(se_graph_right, q_matrix)
        self.extractor.graph_dict['wrong'] = self.__final_graph(se_graph_wrong, q_matrix)

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num)), np.zeros(
                    shape=(self.student_num, self.exercise_num))

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self.__get_csr(train_stu_right, train_exer_right,
                                          shape=(self.student_num, self.exercise_num))
            adj_se_wrong = self.__get_csr(train_stu_wrong, train_exer_wrong,
                                          shape=(self.student_num, self.exercise_num))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num))
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def __final_graph(self, se, ek):
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num: se_num] = se
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)

    @property
    def name(self):
        return 'sscdf'

    def adaptest_update(self, adaptest_data: AdapTestDataset, epoch=10, lr=5e-4, weight_decay=0, batch_size=256):
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr, "weight_decay": weight_decay},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr, "weight_decay": weight_decay}])

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        from tqdm import tqdm
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)
        for epoch_i in tqdm(range(0, epoch), 'Updating'):
            loss = 0.0
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(self.device)
                question_ids = question_ids.to(self.device)
                labels = labels.to(self.device)
                concepts_emb = concepts_emb.to(self.device)
                _ = self.extractor.extract(student_ids, question_ids, concepts_emb)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                pred: torch.Tensor = self.inter_func.compute(
                    student_ts=student_ts,
                    diff_ts=diff_ts,
                    disc_ts=disc_ts,
                    q_mask=concepts_emb,
                    knowledge_ts=knowledge_ts
                )

                bz_loss = loss_func(pred, labels.double())
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.inter_func.monotonicity()
                loss += bz_loss.data.float()

    def evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map

        real = []
        pred = []
        from tqdm import tqdm
        with torch.no_grad():
            self.extractor.eval()
            self.inter_func.eval()
            for sid in tqdm(data, 'Evaluating Selection Prediction'):
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(self.device)
                question_ids = torch.LongTensor(question_ids).to(self.device)
                concepts_embs = torch.Tensor(concepts_embs).to(self.device)

                _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                output: torch.Tensor = self.inter_func.compute(
                    student_ts=student_ts,
                    diff_ts=diff_ts,
                    disc_ts=disc_ts,
                    q_mask=concepts_embs,
                    knowledge_ts=knowledge_ts
                )

                pred += output.tolist()
            self.extractor.train()
            self.inter_func.train()

        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages)

        real = np.array(real)
        real = np.where(real < 0.5, 0.0, 1.0)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)

        # Calculate accuracy
        threshold = 0.5  # You may adjust the threshold based on your use case
        binary_pred = (pred >= threshold).astype(int)
        acc = accuracy_score(real, binary_pred)

        return {
            'auc': auc,
            'cov': cov,
            'acc': acc
        }

    def init_model(self, data: Dataset):
        pass

    def adaptest_save(self, path):
        extractor_dict = self.extractor.state_dict()
        torch.save(extractor_dict, path[0])

        interaction_dict = self.inter_func.state_dict()
        torch.save(interaction_dict, path[0])

    def adaptest_load(self, path):
        self.extractor.load_state_dict(torch.load(path[0]), strict=False)
        self.extractor.to(self.device)

        self.inter_func.load_state_dict(torch.load(path[1]), strict=False)
        self.inter_func.to(self.device)

    def get_BE_weights(self, pred_all):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        d = 100
        Pre_true = {}
        Pre_false = {}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        w_ij_matrix = {}
        for i, _ in pred_all.items():
            w_ij_matrix[i] = {}
            for j, _ in pred_all.items():
                w_ij_matrix[i][j] = 0
        for i, _ in pred_all.items():
            for j, _ in pred_all.items():
                criterion_true_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 1)
                criterion_false_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 1)
                criterion_true_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 0)
                criterion_false_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 0)
                tensor_11 = torch.tensor(Pre_true[i], requires_grad=True)
                tensor_12 = torch.tensor(Pre_true[j], requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                import math
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = Pre_false[i] * Pre_false[j] * diff_norm_00 + Pre_false[i] * Pre_true[j] * diff_norm_01 + \
                         Pre_true[i] * Pre_false[j] * diff_norm_10 + Pre_true[i] * Pre_true[j] * diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self, S_set, w_ij_matrix):
        res = 0.0
        for w_i in w_ij_matrix:
            if (w_i not in S_set):
                mx = float('-inf')
                for j in S_set:
                    if w_ij_matrix[w_i][j] > mx:
                        mx = w_ij_matrix[w_i][j]
                res += mx

        return res

    def delta_q_S_t(self, question_id, pred_all, S_set, sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            v: float, Each weight information
        """

        Sp_set = list(S_set)
        b_array = np.array(Sp_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
        sampled_dict = {key: value for key, value in pred_all.items() if key in sampled_elements}

        w_ij_matrix = self.get_BE_weights(sampled_dict)

        F_s = self.F_s_func(Sp_set, w_ij_matrix)

        Sp_set.append(question_id)
        F_sp = self.F_s_func(Sp_set, w_ij_matrix)
        return F_sp - F_s

    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map

        pred_all = {}
        with torch.no_grad():
            self.extractor.eval()
            self.inter_func.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(self.device)
                question_ids = torch.LongTensor(question_ids).to(self.device)
                concepts_embs = torch.Tensor(concepts_embs).to(self.device)

                _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                output: torch.Tensor = self.inter_func.compute(
                    student_ts=student_ts,
                    diff_ts=diff_ts,
                    disc_ts=disc_ts,
                    q_mask=concepts_embs,
                    knowledge_ts=knowledge_ts
                )

                output = output.detach().cpu().numpy().tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.extractor.train()
            self.inter_func.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict, config):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': config['lr'], "weight_decay": 0},
                                {'params': self.inter_func.parameters(),
                                 'lr': config['lr'], "weight_decay": 0}])

        # for name, param in self.extractor.named_parameters():
        #     param.requires_grad = False
        for name, param in self.inter_func.named_parameters():
            param.requires_grad = False

        original_weights = self.extractor['mastery'].data.clone()

        import copy
        original_state_dict = copy.deepcopy(self.extractor.state_dict())

        student_ids = torch.LongTensor([sid]).to(self.device)
        question_ids = torch.LongTensor([qid]).to(self.device)
        concepts = adaptest_data.concept_map[qid]
        concepts_embs = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_embs[concept] = 1.0
        concepts_embs = torch.Tensor([concepts_embs]).to(self.device)
        correct = torch.LongTensor([1]).to(self.device)
        wrong = torch.LongTensor([0]).to(self.device)

        for ep in range(config['epoch']):
            optimizer.zero_grad()
            _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
            student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
            pred: torch.Tensor = self.inter_func.compute(
                student_ts=student_ts,
                diff_ts=diff_ts,
                disc_ts=disc_ts,
                q_mask=concepts_embs,
                knowledge_ts=knowledge_ts
            )
            loss = loss_func(pred, correct.double())
            loss.backward()
            optimizer.step()

        pos_weights = self.extractor['mastery'].data.clone()
        self.extractor.load_state_dict(original_state_dict)

        for ep in range(config['epoch']):
            optimizer.zero_grad()
            _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
            student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
            pred: torch.Tensor = self.inter_func.compute(
                student_ts=student_ts,
                diff_ts=diff_ts,
                disc_ts=disc_ts,
                q_mask=concepts_embs,
                knowledge_ts=knowledge_ts
            )
            loss = loss_func(pred, wrong.double())
            loss.backward()
            optimizer.step()

        neg_weights = self.extractor['mastery'].data.clone()
        self.extractor.load_state_dict(original_state_dict)

        for name, param in self.inter_func.named_parameters():
            param.requires_grad = True

        # pred = self.model(student_id, question_id, concepts_emb).item()
        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
            (1 - pred) * torch.norm(neg_weights - original_weights).item()
