import torch
import numpy as np
import argparse
import sys
import os
import wandb as wb
from pprint import pprint
from inscd import listener
from inscd.datahub import DataHub
from inscd.models.static.graph import ORCDF
from inscd.models.static.neural import KANCD
from inscd.models.static.neural import NCDM
from inscd.models.static.neural import KSCD
from inscd.models.static.graph import LIGHTGCN
from inscd.models.static.graph import RCD
from inscd.models.static.neural import CDMFKC
from inscd.models.static.classic import MIRT


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='orcdf', type=str,
                    help='ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems', required=True)
parser.add_argument('--data_type', default='junyi', type=str, help='benchmark', required=True)
parser.add_argument('--test_size', default=0.2, type=float, help='test size of benchmark', required=True)
parser.add_argument('--epoch', type=int, help='epoch of method', default=10)
parser.add_argument('--seed', default=0, type=int, help='seed for exp', required=True)
parser.add_argument('--dtype', default=torch.float64, help='dtype of tensor')
parser.add_argument('--device', default='cuda', type=str, help='device for exp')
parser.add_argument('--gcn_layers', type=int, help='numbers of gcn layers')
parser.add_argument('--latent_dim', type=int, help='dimension of hidden layer', default=32)
parser.add_argument('--batch_size', type=int, help='batch size of benchmark', default=1024)
parser.add_argument('--exp_type', help='experiment type', default='cdm')
parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
parser.add_argument('--if_type', type=str, help='interaction type')
parser.add_argument('--keep_prob', type=float, default=1.0, help='edge drop probability')
parser.add_argument('--noise_ratio', type=float, help='the proportion of noise which added into response logs')
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--ssl_temp', type=float, default=3)
parser.add_argument('--ssl_weight', type=float, default=3e-3)
parser.add_argument('--flip_ratio', type=float, default=0.05)
parser.add_argument('--mode', type=str, default='')
config_dict = vars(parser.parse_args())

method_name = config_dict['method']

if config_dict['method'] == 'orcdf':
    mode = f'-w|o-{config_dict["mode"]}' if config_dict["mode"] != 'all' else ''
    method_name += '-' + config_dict['if_type'] + mode
    if config_dict['mode'] != 'all':
        config_dict['method'] = config_dict['method'] + f'-w|o-{config_dict["mode"]}'
elif config_dict['method'] == 'lightgcn' or config_dict['method'] == 'rcd':
    method_name += '-' + config_dict['if_type']
name = f"{method_name}-{config_dict['data_type']}-seed{config_dict['seed']}"
tags = [config_dict['method'], config_dict['data_type'], str(config_dict['seed'])]
config_dict['name'] = name
method = config_dict['method']
datatype = config_dict['data_type']

if config_dict.get('if_type', None) is None:
    config_dict['if_type'] = config_dict['method']

if 'orcdf' in method:
    if config_dict.get('weight_reg') is None:
        config_dict['weight_reg'] = 0.05
pprint(config_dict)
run = wb.init(project="orcdf", name=name,
              tags=tags,
              config=config_dict)
config_dict['id'] = run.id


def main(config):
    def print_plus(tmp_dict, if_wandb=True):
        pprint(tmp_dict)
        if if_wandb:
            wb.log(tmp_dict)

    listener.update(print_plus)
    set_seeds(config['seed'])
    datahub = DataHub(f"datasets/{config['data_type']}")
    if config['exp_type'] == 'cat':
        train_student_num = datahub.group_split(source="total", to=["train", "other"], seed=1, slice_out=0.7)
        valid_student_num = datahub.group_split(source="other", to=["valid", "test"], seed=1, slice_out=1 - 1 / 3)
        concept_map = datahub.get_concept_map()
    else:
        datahub.random_split(source="total", to=["train", "test"], seed=config['seed'],
                             slice_out=1 - config['test_size'])

    if config['exp_type'] != 'cdm':
        if config['exp_type'] == 'gnn':
            validate_metrics = ['auc', 'acc', 'ap', 'rmse', 'f1', 'doa', 'mad']
        else:
            validate_metrics = ['auc', 'acc', 'ap', 'rmse', 'f1', 'doa']
    else:
        if config['if_type'] != 'irt' and config['if_type'] != 'mirt':
            validate_metrics = ['auc', 'acc', 'ap', 'rmse', 'f1', 'doa', 'mad']
        else:
            validate_metrics = ['auc', 'acc', 'ap', 'rmse', 'f1', 'mad']

    print("Number of response logs {}".format(len(datahub)))

    if config.get('noise_ratio', None) is not None:
        datahub.add_noise(config['noise_ratio'], "train")

    if 'orcdf' in config['method']:
        if config['if_type'] == 'kancd' or config['if_type'] == 'mirt' or config['if_type'] == 'kscd' or config[
            'if_type'] == 'irt':
            config['mode'] = 'tf' + config['mode']
        # if config['if_type'] == 'ncd' or config['if_type'] == 'cdmfkc':
        #     config['epoch'] =
        orcdf = ORCDF(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        orcdf.build(latent_dim=config['latent_dim'], device=config['device'], if_type=config['if_type'],
                    gcn_layers=config['gcn_layers'], keep_prob=config['keep_prob'],
                    dtype=config['dtype'], ssl_weight=config['ssl_weight'], ssl_temp=config['ssl_temp'],
                    flip_ratio=config['flip_ratio'], mode=config['mode'])

        orcdf.train(datahub, "train", "test", valid_metrics=validate_metrics,
                    batch_size=config['batch_size'], epoch=config['epoch'], lr=config['lr'],
                    weight_decay=config['weight_decay'])

    elif config['method'] == 'kancd':
        kancd = KANCD(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        kancd.build(32, device=config['device'])
        kancd.train(datahub, "train", "test", valid_metrics=validate_metrics, batch_size=config['batch_size'],
                    epoch=config['epoch'], weight_decay=0, lr=4e-3)

    elif config['method'] == 'ncdm':
        ncdm = NCDM(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        ncdm.build(device=config['device'])
        ncdm.train(datahub, "train", "test", valid_metrics=validate_metrics, batch_size=config['batch_size'],
                   epoch=config['epoch'], weight_decay=0, lr=4e-3)

    elif config['method'] == 'mirt':
        mirt = MIRT(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        mirt.build(latent_dim=32, device=config['device'], if_type='sum')
        mirt.train(datahub, "train", "test", valid_metrics=validate_metrics, batch_size=config['batch_size'],
                   lr=4e-3, weight_decay=0, epoch=config['epoch'])

    elif config['method'] == 'irt':
        mirt = MIRT(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        mirt.build(latent_dim=1, device=config['device'], if_type='sub')
        mirt.train(datahub, "train", "test", valid_metrics=validate_metrics, batch_size=config['batch_size'],
                   lr=4e-3, weight_decay=0, epoch=config['epoch'])

    elif config['method'] == 'kscd':
        kscd = KSCD(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        kscd.build(latent_dim=32, device=config['device'], dtype=torch.float64)
        kscd.train(datahub, "train", "test", valid_metrics=validate_metrics, batch_size=config['batch_size'],
                   weight_decay=0, epoch=config['epoch'], lr=4e-3)

    elif config['method'] == "lightgcn":
        lightgcn = LIGHTGCN(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        lightgcn.build(device=config['device'], if_type=config['if_type'], gcn_layers=3,
                       dtype=torch.float64)
        lightgcn.train(datahub, "train", "test", valid_metrics=validate_metrics,
                       batch_size=config['batch_size'], lr=4e-3, weight_decay=0)

    elif config['method'] == 'rcd':
        rcd = RCD(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        rcd.build(device=config['device'], if_type=config['if_type'], dtype=torch.float64)
        rcd.train(datahub, "train", "test", valid_metrics=validate_metrics,
                  batch_size=config['batch_size'], weight_decay=0, lr=4e-3)

    elif config['method'] == 'cdmfkc':
        cdmfkc = CDMFKC(datahub.student_num, datahub.exercise_num, datahub.knowledge_num)
        cdmfkc.build(device=config['device'], dtype=torch.float64)
        cdmfkc.train(datahub, "train", "test", valid_metrics=validate_metrics, batch_size=config['batch_size'],
                     weight_decay=0, epoch=config['epoch'], lr=4e-3)


if __name__ == '__main__':
    sys.exit(main(config_dict))
