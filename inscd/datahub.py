import pandas as pd
import numpy as np
import torch
import json
import torch.utils.data as tud

from typing import Union
from sklearn.model_selection import train_test_split


class DataHub:
    def __init__(self, data_source: Union[str, np.ndarray], **kwargs):
        # ellipsis object for loading
        self.config: dict = ...
        self.response: np.ndarray = ...
        self.q_matrix: np.ndarray = ...
        self.student_num: int = ...
        self.exercise_num: int = ...
        self.knowledge_num: int = ...
        # scan the datasets to load all available files
        if type(data_source) is str:
            with open(data_source + "/config.json") as file:
                self.config = json.load(file)
            for file_name, file_path in self.config["files"].items():
                read_func = lambda path: pd.read_csv(data_source + "/" + path, header=None).to_numpy(dtype=float)
                exec("self.{} = {}".format(file_name, "read_func(\"{}\")".format(file_path)))
            for info_name, info_var in self.config["info"].items():
                exec("self.{} = {}".format(info_name, info_var))
        else:
            raise ValueError("Unexpected type of \"data_source\" {}".format(type(data_source)))

        self.__set_type_map = {
            "total": self.response
        }

    def q_density(self):
        return np.sum(self.q_matrix) / self.q_matrix.shape[0]

    def top_k_concepts(self, top_k: int, set_type="total"):
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        tmp_set = self.__set_type_map[set_type]
        counts = np.sum(self.q_matrix[np.array(tmp_set[:, 1], dtype=int), :], axis=0)
        return np.argsort(counts).tolist()[:-top_k - 1:-1]

    def load_data(self, new_entry, new_label=None, name="test"):
        if hasattr(self, name) and name not in self.__set_type_map.keys():
            raise ValueError("Conflict with the existing members of this object.")
        else:
            if new_label is None:
                exec("self.__{} = new_entry".format(name))
            else:
                temp_data = np.hstack([new_entry, new_label])
                exec("self.__{} = temp_data".format(name))
            exec("self.__set_type_map[{}] = self.__{}".format(name, name))

    def detach_labels(self, set_type) -> list:
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        return self.__set_type_map[set_type][:, -1].T.tolist()

    def random_split(self, slice_out=0.8, source="total", to: list = None, seed=6594):
        if not 0 < slice_out < 1:
            raise ValueError("\"train_rate\" should be in (0, 1).")

        if to is None:
            to = ["train", "valid"]
        elif len(to) != 2:
            raise ValueError("The length of \"to\" can only be 2.")
        elif hasattr(self, to[0]) and to[0] not in self.__set_type_map.keys():
            raise ValueError("Conflict with the existing members of this object.")
        elif hasattr(self, to[1]) and to[1] not in self.__set_type_map.keys():
            raise ValueError("Conflict with the existing members of this object.")

        if source not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(source, self.__set_type_map.keys()))

        tmp_set = self.__set_type_map[source]
        set0, set1 = train_test_split(tmp_set,
                                      train_size=int(slice_out * self.__set_type_map[source].shape[0]), shuffle=True,
                                      random_state=seed)

        self.__set_type_map[to[0]] = set0
        self.__set_type_map[to[1]] = set1

    def add_noise(self, noise_ratio=0.2, source="train"):
        if not 0 < noise_ratio < 1:
            raise ValueError("\"noise ratio\" should be in (0, 1).")
        tmp_set = self.__set_type_map[source]
        noise_index = np.random.choice(np.arange(tmp_set.shape[0]),
                                       size=int(noise_ratio * tmp_set.shape[0]))
        for index in noise_index:
            init_score = tmp_set[index, 2]
            if init_score == 1:
                tmp_set[index, 2] = 0
            else:
                tmp_set[index, 2] = 1

    def group_split(self, slice_out=0.8, source="total", to: list = None, seed=6594):
        if not 0 < slice_out < 1:
            raise ValueError("\"train_rate\" should be in (0, 1).")

        if to is None:
            to = ["train", "valid"]
        elif len(to) != 2:
            raise ValueError("The length of \"to\" can only be 2.")
        elif hasattr(self, to[0]) and to[0] not in self.__set_type_map.keys():
            raise ValueError("Conflict with the existing members of this object.")
        elif hasattr(self, to[1]) and to[1] not in self.__set_type_map.keys():
            raise ValueError("Conflict with the existing members of this object.")

        if source not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(source, self.__set_type_map.keys()))

        tmp_set = self.__set_type_map[source]
        student_id = np.unique(tmp_set[:, 0].T)

        np.random.seed(seed)
        candidate = np.random.choice(student_id,
                                     size=int(slice_out * len(student_id)), replace=False, )

        self.__set_type_map[to[0]] = tmp_set[np.isin(tmp_set[:, 0], candidate)]
        self.__set_type_map[to[1]] = tmp_set[~np.isin(tmp_set[:, 0], candidate)]
        return len(candidate)

    def mean_correct_rate(self, set_type="total"):
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        tmp_set = self.__set_type_map[set_type]

        if tmp_set.shape[1] == 3:
            return np.sum(tmp_set[:, 2]) / tmp_set.shape[0]
        else:
            raise RuntimeError("Dataset \"{}\" without labels are unable to calculate correct rate".format(set_type))

    def category(self, set_type="total"):
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        tmp_set = self.__set_type_map[set_type]

        if tmp_set.shape[1] != 3:
            raise RuntimeError("Dataset \"{}\" without labels are unable to category".format(set_type))

        student_dict = {}
        for k in range(tmp_set.shape[0]):
            stu_id = tmp_set[k, 0]
            score = tmp_set[k, 2]
            if student_dict.get(stu_id) is None:
                student_dict[stu_id] = score
            else:
                student_dict[stu_id] += score
        sorted_dict = dict(sorted(student_dict.items(), key=lambda x: x[1], reverse=True))
        keys = list(sorted_dict.keys())
        slices = len(keys) // 4
        high_student_id = keys[:slices]
        middle_student_id = keys[slices:slices * 3]
        low_student_id = keys[slices * 3:]
        return high_student_id, middle_student_id, low_student_id

    def to_dataloader(self, batch_size, dtype, set_type="total", label=True):
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        tmp_set = self.__set_type_map[set_type]

        if label is True:
            if tmp_set.shape[1] == 3:
                tensor_dataset = tud.TensorDataset(
                    torch.tensor(tmp_set[:, 0], dtype=torch.int64),
                    torch.tensor(tmp_set[:, 1], dtype=torch.int64),
                    torch.tensor(self.q_matrix[np.array(tmp_set[:, 1], dtype=int), :], dtype=dtype),
                    torch.tensor(tmp_set[:, 2], dtype=dtype)
                )
            else:
                raise RuntimeError("Dataset \"{}\" without labels are unable to transform to pytorch Dataloader "
                                   "with labels.".format(set_type))
        else:
            tensor_dataset = tud.TensorDataset(
                torch.tensor(tmp_set[:, 0], dtype=torch.int64),
                torch.tensor(tmp_set[:, 1], dtype=torch.int64),
                torch.tensor(self.q_matrix[np.array(tmp_set[:, 1], dtype=int), :], dtype=dtype),
            )
        return tud.DataLoader(tensor_dataset, batch_size, shuffle=label)

    def r_matrix(self, set_type="total"):
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        tmp_set = self.__set_type_map[set_type]

        r_matrix = -1 * np.ones(shape=(self.student_num, self.exercise_num))
        for line in tmp_set:
            student_id = int(line[0])
            exercise_id = int(line[1])
            score = line[2]
            r_matrix[student_id, exercise_id] = int(score)
        return r_matrix

    def log_matrix(self, set_type="total"):
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        tmp_set = self.__set_type_map[set_type]

        log_matrix = np.zeros(shape=(self.student_num, self.exercise_num))
        for line in tmp_set:
            student_id = int(line[0])
            exercise_id = int(line[1])
            score = line[2]
            if int(score) == 1:
                log_matrix[student_id, exercise_id] = int(score)
            else:
                log_matrix[student_id, exercise_id] = -1
        return log_matrix

    def get_concept_map(self):
        q2k = {}
        for exercise_id in range(self.q_matrix.shape[0]):
            q2k[exercise_id] = []
        for exercise_id, knowledge_id in zip(*np.where(self.q_matrix != 0)):
            q2k[exercise_id].append(knowledge_id)
        return q2k

    def __getitem__(self, set_type):
        if set_type not in self.__set_type_map.keys():
            raise ValueError("Dataset \"{}\" does not exist. If you create your new dataset via \"load_data()\", "
                             "the parameter \"dataset\" is one of the {}".format(set_type, self.__set_type_map.keys()))
        return self.__set_type_map[set_type]

    def __str__(self):
        return ("A response logs of {} with {} students, {} exercises, {} knowledge concepts and {} entries in original"
                "response logs (except new-coming data set)").format(self.config["dataset"],
                                                                     self.config["info"]["student_num"],
                                                                     self.config["info"]["exercise_num"],
                                                                     self.config["info"]["knowledge_num"], len(self))

    def __len__(self):
        return self.response.shape[0]
