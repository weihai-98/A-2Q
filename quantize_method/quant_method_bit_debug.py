#*************************************************************************
#   > Filename    : quant_method_bit_debug.py
#   > Description : Quantizer for features and weights
#*************************************************************************
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
import pdb
import numpy as np

class u_quant_w_func_alpha_linear_div(Function):
    @staticmethod
    def forward(ctx, weight, alpha, bit):
        """
        weight:[out_features, in_features]
        alpha:[out_features,1]
        w_max:[out_features,1]
        """
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit.abs())
        qmax = 2**(bit-1)-1
        qmax_ = qmax.expand_as(weight)
        alpha = alpha.abs()
        w_max = qmax_*alpha
        alpha_sign = torch.sign(alpha)
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
        weight0, weight, weight_q, w_max, alpha, alpha_sign = ctx.saved_variables
        # Gradient for weight
        grad_weight = grad_output 
        grad_weight[weight>ctx.qmax] = 0
        grad_weight[weight<-ctx.qmax] = 0
        # Gradient for step size
        i = (weight0.abs() <= w_max).float()
        grad_alpha_0 = (weight_q - weight) * i
        grad_alpha_1 = (1-i) * torch.sign(weight) * ctx.qmax
        grad_alpha = 1*(grad_output*(grad_alpha_0 + grad_alpha_1)).sum(1).reshape(-1,1)
        # Gradient for bit
        grad_b_0 = (1-i) * torch.sign(weight)*(ctx.qmax+1)*math.log(2)*alpha
        grad_b = (grad_output * grad_b_0).sum(1).reshape(-1,1)
        return grad_weight, grad_alpha, grad_b
    
class u_quant_xw_func_alpha_linear_div(Function):
    @staticmethod
    def forward(ctx, xw, alpha, bit):
        """
        xw:[in_features, out_features]
        alpha:[in_features,1]
        w_max:[in_features,1]
        """
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit.abs())
        # bit[bit>8] = 8
        qmax = 2**(bit-1)-1
        qmax_ = qmax.expand_as(xw)
        alpha = alpha.abs()
        w_max = qmax_*alpha
        alpha_sign = torch.sign(alpha)
        w_sign = torch.sign(xw)
        xw_div = xw.div(alpha)
        xw_div[xw_div>qmax_] = qmax_[xw_div>qmax_]
        xw_div[xw_div<-qmax_] = -qmax_[xw_div<-qmax_]
        # xw_div = xw_div.abs()
        xw_q = torch.floor(xw_div.abs()+0.5)*w_sign
        ctx.save_for_backward(xw, xw_div, xw_q, w_max, alpha, alpha_sign)
        ctx.qmax = qmax_
        return xw_q.mul(alpha)

    @staticmethod
    def backward(ctx, grad_output):
        grad_xw = grad_output
        xw0, xw, xw_q, w_max, alpha, alpha_sign = ctx.saved_variables
        # STE
        grad_xw[xw>ctx.qmax] = 0
        grad_xw[xw<-ctx.qmax] = 0
        # Grad for scale
        w_q_sign = torch.sign(xw_q.mul(alpha)-xw0)
        i = (xw0.abs() <= w_max).float()
        grad_alpha_0 = (xw_q - xw) * i
        grad_alpha_1 = (1-i) * ctx.qmax*torch.sign(xw)
        grad_alpha = 1*(w_q_sign*(grad_alpha_0 + grad_alpha_1)).mean(1).reshape(-1,1)
        # grad b
        grad_b_0 = (1-i) * torch.sign(xw0)*(ctx.qmax+1)*math.log(2)*alpha
        grad_b = (w_q_sign * grad_b_0).mean(1).reshape(-1,1)
        return grad_xw, grad_alpha, grad_b

class u_quant_fea_func_gama_div(Function):
    @staticmethod
    def forward(ctx, feature, gama, bit,):
        """
            args:
                feature: [N, F]
                gama:[N, 1]
                bit: [N, 1]
            output:
                fea_a : [N, F]
        """
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit.abs())
        
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
        ctx.save_for_backward(feature, fea_div, fea_q, fea_max, gama, gama_sign)
        ctx.qmax = qmax_
        return fea_q.mul(gama)

    @staticmethod
    def backward(ctx, grad_output):
        # grad alpha
        fea_0, feature, feature_q, fea_max, gama, gama_sign = ctx.saved_variables
        grad_feature = grad_output 
        grad_feature[feature<-ctx.qmax] = 0
        grad_feature[feature>ctx.qmax] = 0
        
        # L1-norm of quantization error
        fea_q_sign = torch.sign(feature_q.mul(gama)-fea_0)
        
        # L2-norm of quantization error
        fea_q_norm2 = feature_q.mul(gama)-fea_0
        
        # The gradient of L w.r.t. gama
        i = (fea_0.abs() <= fea_max).float()
        grad_gama_0 = (feature_q - feature) * i
        grad_gama_1 = (1-i) * torch.sign(feature) * ctx.qmax
        grad_gama = 1*(1*fea_q_sign*(grad_gama_0 + grad_gama_1)).mean(1).reshape(-1,1)
        
        # The gradient of L w.r.t. bit
        grad_b_0 = (1-i) *(ctx.qmax+1)*math.log(2)*gama*torch.sign(fea_0)
        grad_b = (1 * fea_q_sign*grad_b_0).mean(1).reshape(-1,1)
        return grad_feature, grad_gama, grad_b, None, None
