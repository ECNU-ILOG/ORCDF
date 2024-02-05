import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, average_precision_score
from sklearn.metrics import f1_score as f1
from joblib import Parallel, delayed


class _Ruler:
    """
    Description:
    A singleton class to measure all metrics in cognitive diagnosis model. The model.__score() is implemented by
    this class. Currently, we support "accuracy (acc)", "area under curve (auc)", "F1-score (f1)", "root-mean-square error (rmse)",
    "degree of agreement (doa)" and "mean average distance (mad)"
    """

    def __init__(self, threshold=0.5, partial_doa=30, top_k=10):
        self.threshold = threshold
        self.partial_doa = partial_doa
        self.top_k_doa = top_k
        self.__method_map = {
            "acc": self.accuracy,
            "auc": self.area_under_curve,
            "f1": self.f1_score,
            "rmse": self.root_mean_square_error,
            "doa": self.degree_of_agreement,
            "mad": self.mean_average_distance,
            'ap': self.average_precision_score,
        }

    def accuracy(self, true_r, pred_r):
        return accuracy_score(true_r, np.array(pred_r) >= self.threshold)

    @staticmethod
    def area_under_curve(true_r, pred_r):
        return roc_auc_score(true_r, pred_r)

    def f1_score(self, true_r, pred_r):
        return f1(true_r, np.array(pred_r) >= self.threshold)

    @staticmethod
    def average_precision_score(true_r, pred_r):
        return average_precision_score(true_r, pred_r)

    @staticmethod
    def average_precision_score_right(true_r, pred_r):
        return average_precision_score(true_r, pred_r, pos_label=1)

    @staticmethod
    def average_precision_score_wrong(true_r, pred_r):
        return average_precision_score(true_r, pred_r, pos_label=0)

    @staticmethod
    def root_mean_square_error(true_r, pred_r):
        return np.sqrt(mean_squared_error(true_r, pred_r))

    @staticmethod
    def __calculate_doa_k(mas_level, q_matrix, r_matrix, k):
        n_questions, _ = q_matrix.shape
        numerator = 0
        denominator = 0
        delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
        question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
        for j in question_hask:
            row_vector = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
            column_vector = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * column_vector
            delta_r_matrix = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
            I_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
            numerator += np.sum(delta_matrix * numerator_)
            denominator += np.sum(delta_matrix * denominator_)
        if denominator == 0:
            DOA_k = 0
        else:
            DOA_k = numerator / denominator
        return DOA_k

    @staticmethod
    def __calculate_doa_k_block(mas_level, q_matrix, r_matrix, k, block_size=50):
        n_students, n_skills = mas_level.shape
        n_questions, _ = q_matrix.shape
        numerator = 0
        denominator = 0
        question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
        for start in range(0, n_students, block_size):
            end = min(start + block_size, n_students)
            mas_level_block = mas_level[start:end, :]
            delta_matrix_block = mas_level[start:end, k].reshape(-1, 1) > mas_level[start:end, k].reshape(1, -1)
            r_matrix_block = r_matrix[start:end, :]
            for j in question_hask:
                row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
                columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
                mask = row_vector * columen_vector
                delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
                I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
                numerator_ = np.logical_and(mask, delta_r_matrix)
                denominator_ = np.logical_and(mask, I_matrix)
                numerator += np.sum(delta_matrix_block * numerator_)
                denominator += np.sum(delta_matrix_block * denominator_)
        if denominator == 0:
            DOA_k = 0
        else:
            DOA_k = numerator / denominator
        return DOA_k

    def degree_of_agreement(self, mastery_level, datahub, set_type):
        q_matrix = datahub.q_matrix
        know_n = q_matrix.shape[1]
        r_matrix = datahub.r_matrix(set_type)
        if know_n > self.partial_doa:
            concepts = datahub.top_k_concepts(self.top_k_doa)
            doa_k_list = Parallel(n_jobs=-1)(
                delayed(self.__calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
        else:
            doa_k_list = Parallel(n_jobs=-1)(
                delayed(self.__calculate_doa_k)(mastery_level, q_matrix, r_matrix, k) for k in range(know_n))
        doa_k_list = [x for x in doa_k_list if x != 0]
        return np.mean(doa_k_list)

    @staticmethod
    def mean_average_distance(mastery_level, eval_func='mse'):
        n = mastery_level.shape[0]
        if eval_func == 'mse':
                # diff_square = (mastery_level[:, np.newaxis, :] - mastery_level[np.newaxis, :, :]) ** 2
                # sum_square = diff_square.sum(axis=2)
                # mse = np.sqrt(sum_square / mastery_level.shape[1])
                # return np.sum(mse) / n / (n - 1)
                # more efficient
            row_sums = np.sum(mastery_level ** 2, axis=1)
            sum_square_diff = row_sums[:, np.newaxis] + row_sums - 2 * np.dot(mastery_level, mastery_level.T)
            sum_square_diff = np.maximum(sum_square_diff, 0)
            rmse = np.sqrt(sum_square_diff / mastery_level.shape[1])
            return np.sum(rmse) / n / (n - 1)
        else:
            dot_product = np.dot(mastery_level, mastery_level.T)
            norms = np.linalg.norm(mastery_level, axis=1)
            norm_product = np.outer(norms, norms)
            cosine_similarity_matrix = dot_product / norm_product
            return np.sum(np.ones(shape=(n, n)) - cosine_similarity_matrix) / n / (n - 1)

    def __str__(self):
        return "Support method keys are {}".format(self.__method_map.keys())

    def __getitem__(self, item):
        if item not in self.__method_map.keys():
            raise ValueError(
                "Currently, we do not support metric {}. We support {}".format(item, self.__method_map.keys()))
        return self.__method_map[item]

    def __call__(self, model, datahub, set_type: str, pred_r: list, metrics: list):
        mastery_level = model.diagnose().detach().cpu().numpy()
        true_r = datahub.detach_labels(set_type)
        results = {}
        for metric in metrics:
            if metric in ["acc", "auc", "f1", "rmse", 'ap']:
                results[metric] = self.__method_map[metric](true_r, pred_r)
            elif metric in ["doa"]:
                results[metric] = self.__method_map[metric](mastery_level, datahub, set_type)
            elif metric in ["mad"]:
                results[metric] = self.__method_map[metric](mastery_level)
            else:
                raise ValueError(
                    "Currently, we do not support metric {}. We support {}".format(metric, self.__method_map.keys()))
        return results
