#*************************************************************************
#   > Filename    : quant_method_red_uniform_bit_debug.py
#   > Description : Quantization method on graph-level tasks
#*************************************************************************
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch_scatter import scatter_add,scatter_mean,scatter_max,scatter_min
import pdb

from torch_scatter import scatter_add

class u_quant_w_func_alpha_linear_div(Function):
    @staticmethod
    def forward(ctx, weight, alpha, bit):
        """
        weight:[in_features, out_features]
        alpha:[in_features,1]
        bit:[in_features,1]
        """
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit.abs())
        qmax = 2**(bit-1)-1
        qmax_ = qmax.expand_as(weight)
        w_max = qmax_*alpha
        alpha_sign = torch.sign(alpha)
        alpha = alpha.abs()
        w_sign = torch.sign(weight)
        weight_div = weight.div(alpha)
        weight_div[weight_div>qmax_] = qmax_[weight_div>qmax_]
        weight_div[weight_div<-qmax_] = -qmax_[weight_div<-qmax_]
        # weight_div = weight_div.abs()
        weight_q = torch.floor(weight_div.abs()+0.5)*w_sign
        ctx.save_for_backward(weight, weight_div, weight_q, w_max, alpha, alpha_sign)
        ctx.qmax = qmax_
        return weight_q.mul(alpha)

    @staticmethod
    def backward(ctx, grad_output):
        grad_weight = grad_output  # grad for weights will not be clipped
        grad_w_max = None
        # grad alpha
        w0, weight, weight_q, w_max, alpha, alpha_sign = ctx.saved_variables
        # Gradient for step size
        i = (w0.abs() <= w_max).float()
        w_q_sign = torch.sign(weight_q.mul(alpha)-w0)
        grad_alpha_0 = (weight_q - weight)* i
        grad_alpha_1 = (1-i)  *ctx.qmax*torch.sign(w0)
        grad_alpha = 1*(1*grad_output*(grad_alpha_0 + grad_alpha_1)).sum(1).reshape(-1,1)
        # torch.ten()
        # grad for bit
        grad_b_0 = (1-i) * torch.sign(weight)*(ctx.qmax+1)*math.log(2)*alpha
        grad_b = (grad_output * grad_b_0).sum(1).reshape(-1,1)
        return grad_weight, grad_alpha, grad_b
    
class u_quant_xw_func_alpha_linear_div(Function):
    @staticmethod
    def forward(ctx, weight, alpha, bit):
        """
        To quantize the result of XW or the X along the feature dimension
        xw:[features_dim, num_nodes]
        alpha:[features_dim,1]
        bit:[features_dim,1]
        """
        bit = torch.round(bit.abs())
        # bit[bit>8] = 8
        qmax = 2**(bit-1)-1
        qmax_ = qmax.expand_as(weight)
        w_max = qmax_*alpha
        alpha_sign = torch.sign(alpha)
        alpha = alpha.abs()
        w_sign = torch.sign(weight)
        weight_div = weight.div(alpha)
        weight_div[weight_div>qmax_] = qmax_[weight_div>qmax_]
        weight_div[weight_div<-qmax_] = -qmax_[weight_div<-qmax_]
        weight_q = torch.floor(weight_div.abs()+0.5)*w_sign
        ctx.save_for_backward(weight, weight_div, weight_q, w_max, alpha, alpha_sign)
        ctx.qmax = qmax_
        return weight_q.mul(alpha)

    @staticmethod
    def backward(ctx, grad_output):
        grad_weight = grad_output  # Gradient for xw will not be clipped by STE
        grad_w_max = None
        # grad alpha
        w0, weight, weight_q, w_max, alpha, alpha_sign = ctx.saved_variables
        # Gradient for step size
        i = (w0.abs() <= w_max).float()
        w_q_sign = torch.sign(weight_q.mul(alpha)-w0)
        grad_alpha_0 = (weight_q - weight)* i
        grad_alpha_1 = (1-i)  *ctx.qmax*torch.sign(w0)
        grad_alpha = 1*(1*w_q_sign*(grad_alpha_0 + grad_alpha_1)).mean(1).reshape(-1,1)
        # grad b
        grad_b_0 = (1-i) * torch.sign(weight)*(ctx.qmax+1)*math.log(2)*alpha
        grad_b = (grad_output * grad_b_0).sum(1).reshape(-1,1)
        return grad_weight, grad_alpha, grad_b

class u_quant_fea_func_gama_div(Function):
    @staticmethod
    def forward(ctx, feature, gama, bit, deg_inv_sqrt, cal_mode, scale_index):
        """
            feature     :       [N, F]
            gama        :       [N, 1]
            bit         :       [N, 1]
            scale_index :       [N, 1]
        """
        ctx.size = gama.size()[0]
        gama = gama[scale_index]
        bit = bit[scale_index]
        bit = torch.round(bit.abs())
        # bit[bit>9] = 9
        qmax = 2**(bit-1)-1
        qmax_ = qmax.expand_as(feature)
        gama = gama.abs()
        fea_max = qmax_*gama
        gama_sign = torch.sign(gama)
        fea_sign = torch.sign(feature)
        fea_div = feature.div(gama)
        fea_div[fea_div>qmax_] = qmax_[fea_div>qmax_]
        fea_div[fea_div<-qmax_] = -qmax_[fea_div<-qmax_]
        fea_q = torch.floor(fea_div.abs()+0.5)*fea_sign
        if(cal_mode==2):
            bit = bit+1
        ctx.save_for_backward(feature, fea_div, fea_q, fea_max, gama, gama_sign)
        ctx.qmax = qmax_
        ctx.cal_mode = cal_mode
        ctx.index = scale_index
        return fea_q.mul(gama)

    @staticmethod
    def backward(ctx, grad_output):
        grad_feature = grad_output  # grad for features will not be clipped
        # grad alpha
        fea0, feature, feature_q, fea_max, gama, gama_sign = ctx.saved_variables
        grad_feature[feature<-ctx.qmax] = 0
        grad_feature[feature>ctx.qmax] = 0
        # Gradient for gama
        i = (fea0.abs() <= fea_max).float()
        grad_gama_0 = (feature_q - feature)* i
        grad_gama_1 = (1-i) * ctx.qmax*torch.sign(fea0)
        grad_gama = 1*(1*grad_output*(grad_gama_0 + grad_gama_1)).sum(1).reshape(-1,1)
        # pdb.set_trace()
        # torch.ten()
        grad_gama_out = grad_gama.new_zeros((ctx.size,1))
        # Gather the gradients
        grad_gama_out = scatter_add(grad_gama,ctx.index,dim=0,out=grad_gama_out)
        
        # Gradient for bitwidth
        grad_b_0 = (1-i)*torch.sign(fea0)*(ctx.qmax+1)*math.log(2)*gama
        grad_b = (grad_output * grad_b_0).sum(1).reshape(-1,1)
        grad_b_out = grad_b.new_zeros((ctx.size,1))
        grad_b_out = scatter_add(grad_b,ctx.index,dim=0,out=grad_b_out)
        return grad_feature, grad_gama_out, grad_b_out, None, None, None

