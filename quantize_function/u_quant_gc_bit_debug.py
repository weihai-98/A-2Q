#*************************************************************************
#   > Filename    : u_quant_gc_bit_debug.py
#   > Description : Quantization function for graph-level tasks
#*************************************************************************
import os
import math
import time
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset,Planetoid
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean,scatter_max,scatter_min
from torch.autograd.function import InplaceFunction, Function
from torch_geometric.nn import GCNConv,GINConv
from torch_geometric.nn.inits import glorot, zeros
from tqdm import tqdm
from quantize_function.get_scale_index import get_deg_index, get_scale_index, get_scale_index_naive,get_scale_index_uniform
import argparse
from matplotlib.pyplot import MultipleLocator
from quantize_method.quant_method_red_uniform_bit_debug import *
import pdb

class u_quant_weight(nn.Module):
    '''
        weight uniform quantization.
    '''
    def __init__(self, in_channels, out_channels, bit, alpha_init=0.01,alpha_std=0.01,):
        super(u_quant_weight, self).__init__()
        self.bit = bit
        # quantizer
        
        self.quant_weight_func = u_quant_w_func_alpha_linear_div
        # parameters
        # initialize the step size by normal distribution
        _init_alpha = alpha_init
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
    
class u_quant_xw(nn.Module):
    '''
        The result of XW uniform quantization.
    '''
    def __init__(self, in_channels, out_channels, bit,alpha_init=0.01,alpha_std=0.01):
        super(u_quant_xw, self).__init__()
        self.bit = bit
        # quantizer
        self.quant_weight_func = u_quant_xw_func_alpha_linear_div
        # parameters
        _init_alpha = alpha_init
        _alpha = torch.Tensor(out_channels,1)
        self.alpha = torch.nn.Parameter(_alpha)
        torch.nn.init.normal_(self.alpha, mean=alpha_init, std=alpha_std)
        self.alpha = torch.nn.Parameter(self.alpha.abs())
        _init_bit = bit
        # _max_w = torch.Tensor(1,in_channels)
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels,1))
        torch.nn.init.constant_(self.bit,_init_bit)
    def forward(self, fea):
        # quantization
        fea_q = self.quant_weight_func.apply(fea, self.alpha, self.bit)
        return fea_q

class u_quant_fea(nn.Module):
    '''
        Features uniform quantization.
    '''
    def __init__(self, dim_fea, bit, gama_init=0.001,gama_std=0.001, uniform=False,is_naive=False,quant_fea=True,init='norm'):
        super(u_quant_fea, self).__init__()
        self.bit   = bit
        self.cal_mode = 1
        self.quant_fea = quant_fea
        self.uniform = uniform
        # if is_naive==True do not use the Nearest Neighbor Strategy
        self.is_naive = is_naive
        
        self.num_deg = dim_fea
        self.deg_inv_sqrt = 1.0
        # quantizer
        self.quant_fea_func = u_quant_fea_func_gama_div
        self.quant_fea_func_no_index = u_quant_fea_func_gama_div
        # parameters initialization
        # initialize step size
        _init_gama = gama_init
        self.gama = torch.nn.Parameter(torch.Tensor(dim_fea,1))
        # Norm init or Uniform init
        if(init=='norm'):
            torch.nn.init.normal_(self.gama, mean=gama_init, std=gama_std)
            self.gama = torch.nn.Parameter(self.gama.abs())
        else:
            torch.nn.init.uniform_(self.gama,0,1)
        
        # initialize the bit, bit=4
        self.bit = torch.nn.Parameter(torch.Tensor(torch.Tensor(dim_fea,1)))
        _init_bit = bit
        torch.nn.init.constant_(self.bit,_init_bit)


    def forward(self, fea, edge_index):
        if((edge_index!=None)&(self.uniform==False)&(self.is_naive==False)):
            deg_index = get_deg_index(fea=fea,edge_index=edge_index)
            scale_index = get_scale_index(fea=fea,deg_index=deg_index,scale=self.gama,bit=self.bit)
            # analysis_fea(fea,edge_index)
            unique_index = torch.unique(scale_index)
            bit_sum = fea.size(1)*self.bit[unique_index].sum()/8./1024.
            fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit, self.deg_inv_sqrt, self.cal_mode, scale_index)
        elif((edge_index!=None)&(self.uniform==True)&(self.is_naive==False)):
            scale_index = get_scale_index_uniform(fea=fea,deg_index=edge_index,scale=self.gama,bit=self.bit)
            unique_index = torch.unique(scale_index)
            # pdb.set_trace()
            bit_sum = fea.size(1)*self.bit[unique_index].sum()/8./1024.
            fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit, self.deg_inv_sqrt, self.cal_mode,scale_index)
        elif((edge_index!=None)&(self.is_naive==True)):
            scale_index = get_scale_index_naive(fea,edge_index,self.num_deg)
            unique_index = torch.unique(scale_index)
            bit_sum = fea.size(1)*self.bit[unique_index].sum()/8./1024.
            fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit, self.deg_inv_sqrt, self.cal_mode,scale_index)
        else:
            fea_q = self.quant_fea_func_no_index.apply(fea, self.gama, self.bit, self.deg_inv_sqrt, self.cal_mode)
            bit_sum = fea_q.new_zeros(1)
        if(not self.quant_fea):
            fea_q = fea
            bit_sum = 0
        return fea_q,bit_sum

class QLinear(nn.Linear):
    ''' 
    Quantized linear layers.
    '''
    def __init__(self, in_features, out_features, num_nodes, bit, gama_init=1e-3, bias=True,all_positive=False, 
                para_dict={'alpha_init':0.01,'alpha_std':0.02,'gama_init':0.1,'gama_std':0.2},
                quant_fea=True,
                uniform=False,
                is_naive=False,
                init='norm'):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.bit = bit
        if(all_positive):
            bit_fea = bit+1
        else:
            bit_fea = bit
        alpha_init = para_dict['alpha_init']
        gama_init = para_dict['gama_init']
        alpha_std = para_dict['alpha_std']
        gama_std = para_dict['gama_std']
        # weight quantization module
        self.weight_quant = u_quant_weight(in_features, out_features, bit, alpha_init=alpha_init,alpha_std=alpha_std,)
        self.fea_quant = u_quant_fea(num_nodes,bit_fea, gama_init=gama_init,gama_std=gama_std,uniform=uniform,is_naive=is_naive,init=init)
        if(quant_fea==False):
            self.fea_quant = nn.Identity()
        # glorot(self.weight)
    
    def forward(self, x, edge_index, bit_sum):
        # weight quantization
        weight_q = self.weight_quant(self.weight)
        # weight_q = self.weight
        if(isinstance(self.fea_quant,nn.Identity)):
            fea_q = self.fea_quant(x)
            bit_sum_layer = 0
        else:
            fea_q,bit_sum_layer = self.fea_quant(x,edge_index)
        bit_sum+=bit_sum_layer
        # fea_q = x
        return F.linear(fea_q, weight_q, self.bias), edge_index, bit_sum
