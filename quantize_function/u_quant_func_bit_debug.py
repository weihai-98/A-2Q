#*************************************************************************
#   > Filename    : u_quant_func_bit_debug.py
#   > Description : Quantization Function
#*************************************************************************
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset,Planetoid
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction, Function
from torch_geometric.nn import GCNConv,GINConv
from torch_geometric.nn.inits import glorot, zeros
from tqdm import tqdm
import argparse
# from qmax_quantize import u_quant_vec_func_sigma
from quantize_method.quant_method_bit_debug import *
import pdb


class u_quant_weight(nn.Module):
    '''
        weight uniform quantization.
    '''
    def __init__(self, in_channels, out_channels, bit, alpha_init=0.1,alpha_std=0.1,):
        super(u_quant_weight, self).__init__()
        self.bit = bit
        # quantizer
        self.quant_weight_func = u_quant_w_func_alpha_linear_div
        # parameters initialize
        _init_alpha = alpha_init
        # initialize the step size by normal distribution
        _alpha = torch.Tensor(out_channels,1)
        self.alpha = torch.nn.Parameter(_alpha)
        torch.nn.init.normal_(self.alpha, mean=alpha_init, std=alpha_std)
        self.alpha = torch.nn.Parameter(self.alpha.abs())
        # initialize the bit bit=4
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels,1))
        torch.nn.init.constant_(self.bit,_init_bit)
    def forward(self, weight):
        # quantization
        weight_q = self.quant_weight_func.apply(weight, self.alpha, self.bit)
        return weight_q

class u_quant_fea(nn.Module):
    '''
        feature uniform quantization.
    '''
    def __init__(self, dim_fea, bit, gama_init=0.001,gama_std=0.001, quant_fea=True):
        super(u_quant_fea, self).__init__()
        self.bit   = bit
        self.cal_mode = 1
        self.quant_fea = quant_fea
        self.deg_inv_sqrt = 1.0
        # quantizer
        self.quant_fea_func = u_quant_fea_func_gama_div
        # parameters initialization
        # initialize step size
        self.gama = torch.nn.Parameter(torch.Tensor(dim_fea,1))
        torch.nn.init.normal_(self.gama, mean=gama_init, std=gama_std)
        self.gama = torch.nn.Parameter(self.gama.abs())
        # initialize the bit, bit=4
        self.bit = torch.nn.Parameter(torch.Tensor(torch.Tensor(dim_fea,1)))
        _init_bit = bit
        torch.nn.init.constant_(self.bit,_init_bit)

    def forward(self, fea):
        if(not self.quant_fea):
            fea_q = fea
        else:
            fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit,)
        return fea_q


class QLinear(nn.Linear):
    ''' 
    Quantized linear layers.
    '''
    def __init__(self, in_features, out_features, num_nodes, bit, gama_init=1e-3, bias=True,all_positive=False,
                para_dict={'alpha_init':0.01,'alpha_std':0.02,'gama_init':0.1,'gama_std':0.2},
                quant_fea=True):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.bit = bit
        # if the value is all positive then we use the unsign quantization
        if(all_positive):
            bit_fea = bit + 1
        else:
            bit_fea = bit
        alpha_init = para_dict['alpha_init']
        gama_init = para_dict['gama_init']
        alpha_std = para_dict['alpha_std']
        gama_std = para_dict['gama_std']
        # weight quantization module
        self.weight_quant = u_quant_weight(in_features, out_features, bit, alpha_init=alpha_init,alpha_std=alpha_std)
        # features quantization module
        self.fea_quant = u_quant_fea(num_nodes,bit_fea, gama_init=gama_init,gama_std=gama_std,)
        if(quant_fea==False):
            # Do not quantize the feature when the value is 0 or 1
            self.fea_quant = nn.Identity()
        # glorot(self.weight)
    
    def forward(self, x):
        # weight quantization
        weight_q = self.weight_quant(self.weight)
        # weight_q  = self.weight
        fea_q = self.fea_quant(x)
        # fea_q = x
        return F.linear(fea_q, weight_q, self.bias)

    
class u_quant_xw(nn.Module):
    '''
        xw uniform quantization.
    '''
    def __init__(self, in_channels, out_channels, bit,alpha_init=0.1, alpha_std=0.1,):
        super(u_quant_xw, self).__init__()
        self.bit = bit
        # quantizer
        self.quant_weight_func = u_quant_xw_func_alpha_linear_div
        # parameters
        _init_alpha = 0.1
        _alpha = torch.Tensor(out_channels,1)
        self.alpha = torch.nn.Parameter(_alpha)
        torch.nn.init.normal_(self.alpha, mean=alpha_init, std=alpha_std)
        self.alpha = torch.nn.Parameter(self.alpha.abs())
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels,1))
        torch.nn.init.constant_(self.bit,_init_bit)
    def forward(self, fea):
        # quantization
        fea_q = self.quant_weight_func.apply(fea, self.alpha, self.bit)
        return fea_q