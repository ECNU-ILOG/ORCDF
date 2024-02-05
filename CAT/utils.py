import numpy as np
import torch

import CAT as CAT
from inscd.datahub import DataHub


def get_concept_map():
    datahub = DataHub(f"../datasets/Math1")

    def set_seeds(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    set_seeds(1)
    train_student_num = datahub.group_split(source="total", to=["train", "other"], seed=1, slice_out=0.7)
    valid_student_num = datahub.group_split(source="other", to=["valid", "test"], seed=1, slice_out=1 - 1 / 3)
    concept_map = datahub.get_concept_map()

    def renumber_student(data):
        unique_values, indices = np.unique(data[:, 0], return_inverse=True)
        remapping = {original_value: new_number for new_number, original_value in enumerate(unique_values)}
        data[:, 0] = np.vectorize(remapping.get)(data[:, 0])
        return data
    test_length = 20


    config = {
        'learning_rate': 0.002,
        'batch_size': 256,
        'num_epochs': 10,
        'num_dim': 10,
        'device': 'cpu',
        'prednet_len1': 128,
        'prednet_len2': 64,
    }

    train_data = CAT.dataset.TrainDataset(renumber_student(datahub['train']).astype(int), concept_map,
                                          train_student_num,
                                          datahub.exercise_num,
                                          datahub.knowledge_num)

    method = 'ncd'
    if method == 'racdf':
        from inscd.models.static.graph.orcdf import SSCDF
        model = SSCDF(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        model.build(latent_dim=32, device=config['device'], if_type='ncd',
                gcn_layers=1, keep_prob=1.0,
                dtype=torch.float64, ssl_weight=3e-3, ssl_temp=3,
                flip_ratio=0.05, mode='cl')
        model.train(datahub, valid_metrics=['auc', 'ap'], lr=1e-3, batch_size=256, epoch=1)
    elif method == 'irt':
        model = CAT.model.IRTModel(**config)
        model.init_model(train_data)
        model.train(train_data, log_step=10)

    elif method == 'ncd':
        model = CAT.model.NCDModel(**config)
        model.init_model(train_data)
        model.train(train_data)

    if model == 'racdf':
        test_data = CAT.dataset.AdapTestDataset(datahub['test'].astype(int), concept_map,
                                            datahub.student_num - train_student_num - valid_student_num,
                                            datahub.exercise_num,
                                            datahub.knowledge_num)
    else:
        test_data = CAT.dataset.AdapTestDataset(renumber_student(datahub['test']).astype(int), concept_map,
                                            datahub.student_num - train_student_num - valid_student_num,
                                            datahub.exercise_num,
                                            datahub.knowledge_num)
    import random
    strategies = [CAT.strategy.BECATstrategy()]
    for strategy in strategies:
        test_data.reset()
        print('-----------')
        print(f'start adaptive testing with {strategy.name} strategy')
        print(f'Iteration 0')
        results = model.evaluate(test_data)
        for name, value in results.items():
            print(f'{name}:{value}')
        if model.name != 'racdf':
            student_list = range(test_data.num_students)
        else:
            student_list = np.unique(datahub['test'][:, 0]).astype(int).tolist()
        test_data.student_list = student_list

        S_sel = {}
        for sid in student_list:
            key = sid
            S_sel[key] = []
        selected_questions = {}
        for it in range(1, test_length + 1):
            print(f'Iteration {it}')
            # select question
            if it == 1 and strategy.name == 'BECAT Strategy':
                for sid in student_list:
                    untested_questions = np.array(list(test_data.untested[sid]))
                    random_index = random.randint(0, len(untested_questions) - 1)
                    selected_questions[sid] = untested_questions[random_index]
                    S_sel[sid].append(untested_questions[random_index])
            elif strategy.name == 'BECAT Strategy':
                selected_questions = strategy.adaptest_select(model, test_data, S_sel)
                for sid in student_list:
                    S_sel[sid].append(selected_questions[sid])
            else:
                selected_questions = strategy.adaptest_select(model, test_data)

            select_data = []
            for student, question in selected_questions.items():
                test_data.apply_selection(student, question)
                select_data.append([student, question, test_data.data[student][question]])

            if model.name == 'racdf':
                model.update_graph(np.vstack([datahub['train'], np.array(select_data)]), datahub.q_matrix)

            # update models
                model.adaptest_update(test_data, lr=1e-3, batch_size=256, epoch=10)
            else:
                model.adaptest_update(test_data)
            # evaluate models
            results = model.evaluate(test_data)
            # log results
            for name, value in results.items():
                print(f'{name}:{value}')


get_concept_map()
