from cmath import nan
import os
import shutil
import time
import pprint
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import random

def save_list_to_txt(name,input_list):
    f=open(name,mode='w')
    for item in input_list:
        f.write(item+'\n')
    f.close()

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print ('use gpu:',gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print ('create folder:',path)
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()




_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm



def load_model(model,dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']
    if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    model.load_state_dict(model_dict)

    return model

def load_model_cl(model,dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']

    # load from a pretrained model
    pretrained_dict_q = {'encoder.' + k: v for k, v in pretrained_dict.items()}
    pretrained_dict_q = {k: v for k, v in pretrained_dict_q.items() if k in model_dict}

    pretrained_dict_k = {'encoder_m.' + k: v for k, v in pretrained_dict.items()}
    pretrained_dict_k = {k: v for k, v in pretrained_dict_k.items() if k in model_dict}

    # update the param in encoder & encoder_m, remain others still
    model_dict.update(pretrained_dict_q)
    model_dict.update(pretrained_dict_k)

    model.load_state_dict(model_dict)

    return model

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def detect_grad_nan(model):
    for name, param in model.named_parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()

def detect_model_param(model1, model2):
    # print model1
    for name1, param1 in model1.named_parameters():
        print(name1)
        print(param1)
        break
    # print model2
    for name2, param2 in model2.named_parameters():
        print(name2)
        print(param2)
        break

    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if param1.equal(param2):
            print(name1, '==', name2)
        else:
            print(name1, '!=', name2)
