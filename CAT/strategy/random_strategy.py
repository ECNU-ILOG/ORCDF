import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset
from tqdm import tqdm

class RandomStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Random Select Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in tqdm(adaptest_data.student_list):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            selection[sid] = untested_questions[np.random.randint(len(untested_questions))]
        return selection