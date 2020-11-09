import torch
import numpy as np

from wann_gd.model import Model, ActModule

def load_gd_model(path):
    d = torch.load(path)
    shared_weight = torch.autograd.Variable(torch.Tensor([1]))
    model = Model.from_dict(shared_weight, d['state'])
    
    return model

def load_genetic_model(ind):
    """!This does not load activations correctly and is just used for visualisation!"""
    ind.express()
    net = ind.network

    shared_weight = torch.nn.Parameter(torch.Tensor([1]))


    weights = list()
    for indices in net.layers(include_output=True):
        weights.append(torch.Tensor(net.weight_matrix[:np.min(indices), indices - net.offset].T))

    state_dict = dict()
    state_dict[f'shared_weight'] = shared_weight

    for i, w in enumerate(weights[:-1]): # hidden layers
        state_dict[f'hidden_layers.{i}.shared_weight'] = shared_weight
        state_dict[f'hidden_layers.{i}.linear.weight'] = w
        state_dict[f'hidden_layers.{i}.activation.weight'] = torch.zeros((len(ActModule.available_act_functions), w.shape[0]))

    w = weights[-1] # output layer
    state_dict[f'output_layer.shared_weight'] = shared_weight
    state_dict['output_layer.linear.weight'] = w
    state_dict['output_layer.activation.weight'] = torch.zeros((len(ActModule.available_act_functions), w.shape[0]))

    model = Model.from_dict(shared_weight, state_dict)
    
    return model