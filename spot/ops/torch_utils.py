
import time

import torch

import numpy as np

def get_device():
    '''
    Returns GPU device if available, else CPU.
    '''
    gpu_device_str = 'cuda:0'
    device_str = gpu_device_str if torch.cuda.is_available() else 'cpu'
    if device_str == gpu_device_str:
        print('Using detected GPU!')
    else:
        print('No detected GPU...using CPU.')
    device = torch.device(device_str)
    return device

def is_sorted(tensor):
    if tensor.size(0) == 0:
        return True
    return torch.all((tensor[1:] - tensor[:-1]) > 0)

def torch_to_numpy(tensor_list):
    return [x.to('cpu').data.numpy() if x is not None else None for x in tensor_list ]

def torch_to_scalar(tensor_list):
    return [x.to('cpu').item() for x in tensor_list]

def torch_sets_unique(t1, t2):
    """
    Finds values that are not in both x and y.
    """
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params