from abc import abstractmethod

from . import listener
from . import ruler
from . import unifier


class _Extractor:
    @abstractmethod
    def extract(self, **kwargs):
        ...

    @abstractmethod
    def __getitem__(self, item):
        ...

class _InteractionFunction:
    @abstractmethod
    def compute(self, **kwargs):
        ...

    @abstractmethod
    def transform(self, mastery, knowledge):
        ...

    def monotonicity(self):
        ...


class _CognitiveDiagnosisModel:
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, save_flag=False):
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.save_flag = save_flag
        # ellipsis members
        self.method = ...
        self.device: str = ...
        self.inter_func: _InteractionFunction = ...
        self.extractor: _Extractor = ...
        self.mastery_list: list = []
        self.diff_list: list = []
        self.knowledge_list: list = []


    def _train(self, datahub, set_type="train",
               valid_set_type=None, valid_metrics=None, **kwargs):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        unifier.train(datahub, set_type, self.extractor, self.inter_func, **kwargs)
        if valid_set_type is not None:
            self.score(datahub, valid_set_type, valid_metrics, **kwargs)
        if self.save:
            self.mastery_list.append(self.get_attribute('mastery'))
            self.diff_list.append(self.get_attribute('diff'))
            self.knowledge_list.append(self.get_attribute('knowledge'))

    def _predict(self, datahub, set_type: str, **kwargs):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return unifier.predict(datahub, set_type, self.extractor, self.inter_func, **kwargs)

    @listener
    def _score(self, datahub, set_type: str, metrics: list, **kwargs):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        pred_r = unifier.predict(datahub, set_type, self.extractor, self.inter_func, **kwargs)
        return ruler(self, datahub, set_type, pred_r, metrics)

    @abstractmethod
    def build(self, *args, **kwargs):
        ...

    @abstractmethod
    def train(self, datahub, set_type, valid_set_type=None, valid_metrics=None, **kwargs):
        ...

    @abstractmethod
    def predict(self, datahub, set_type, **kwargs):
        ...

    @abstractmethod
    def score(self, datahub, set_type, metrics: list, **kwargs) -> dict:
        ...

    @abstractmethod
    def diagnose(self):
        ...

    @abstractmethod
    def load(self, ex_path: str, if_path: str):
        ...

    @abstractmethod
    def save(self, ex_path: str, if_path: str):
        ...

    @abstractmethod
    def get_attribute(self, attribute_name):
        ...
