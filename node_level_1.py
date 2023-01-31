#*************************************************************************
#   > Filename    : make_bit_great_again.py
#   > Description : For GCN & GIN on CiteSeer, Cora and PubMed
#*************************************************************************
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import random
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
from quantize_function.u_quant_func_bit_debug import *
from quantize_function.qGINConv import GIN
from utils.quant_utils import analysis_bit
import pdb
def paras_group(model):
    all_params = model.parameters()
    weight_paras=[]
    quant_paras_bit_weight = []
    quant_paras_bit_fea = []
    quant_paras_scale_weight = []
    quant_paras_scale_fea = []
    quant_paras_scale_xw = []
    quant_paras_bit_xw = []
    other_paras = []
    for name,para in model.named_parameters():
        if('quant' in name and 'bit' in name and 'weight' in name):
            quant_paras_bit_weight+=[para]
            # para.requires_grad = False
        elif('quant' in name and 'bit' in name and 'fea' in name):
            quant_paras_bit_fea+=[para]
        elif('quant' in name and 'bit' not in name and 'weight' in name):
            quant_paras_scale_weight+=[para]
            # para.requires_grad = False
        elif('quant' in name and 'bit' not in name and 'fea' in name):
            quant_paras_scale_fea+=[para]
        elif('xw'in name and 'q' in name and 'bit' not in name):
            quant_paras_scale_xw+=[para]
        elif('xw'in name and 'q' in name and 'bit' in name):
            quant_paras_bit_xw+=[para]
        elif('weight' in name and 'quant' not in name ):
            weight_paras+=[para]
    params_id = list(map(id,quant_paras_bit_fea))+list(map(id,quant_paras_bit_weight))+list(map(id,quant_paras_scale_weight))+list(map(id,quant_paras_scale_fea))\
        +list(map(id,quant_paras_scale_xw))+list(map(id,weight_paras))+list(map(id,quant_paras_bit_xw))
    other_paras = list(filter(lambda p: id(p) not in params_id, all_params))
    return weight_paras,quant_paras_bit_weight,quant_paras_bit_fea,quant_paras_scale_weight,quant_paras_scale_fea,quant_paras_scale_xw,quant_paras_bit_xw,other_paras

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def parameter_stastic(model,dataset,hidden_units):
    # Cal the memory size
    w_Byte = torch.tensor(0)
    a_Byte = torch.tensor(0)
    for name, par in model.named_parameters():
        if(('bit' in name)&('fea' in name)):
            a_scale = hidden_units
            a_Byte = a_scale*par.abs().sum()/8./1024.+a_Byte
    return w_Byte, a_Byte

def load_checkpoint(model, checkpoint):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        new_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict.keys()))}
        model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
    return model


class qGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_nodes, bit, all_positive=False,
                para_dict={'alpha_init':0.01,'alpha_std':0.02,'gama_init':0.1,'gama_std':0.2},
                quant_fea=True):
        super().__init__(aggr='add')  
        num_nodes = dataset.data.num_nodes
        self.lin = QLinear(in_channels, out_channels, num_nodes, bit, all_positive=all_positive, para_dict=para_dict, quant_fea=quant_fea)
        # Quant the result of XW
        self.q_xw = u_quant_xw(in_channels,out_channels,bit,alpha_init=0.01,alpha_std=0.01)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix.
        x = self.lin(x)
        
        # Quantize the result of the XW
        x = x.T
        x = self.q_xw(x)
        x = x.T

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN(torch.nn.Module):
    def __init__(self, hidden_units,bit, is_q=False, drop_out=0):
        super().__init__()
        # The mean and var for initialization
        para_list=[{'alpha_init':0.1,'gama_init':0.01,'alpha_std':0.2,'gama_std':0.01}]
        num_nodes = dataset.data.num_nodes
        self.drop_out = drop_out
        if(is_q==False):
            self.conv1 = GCNConv(dataset.num_node_features, hidden_units, bias=True,improved=False)
        else:
            self.conv1 = qGCNConv(dataset.num_node_features, hidden_units, num_nodes, bit, all_positive=True,
                                para_dict=para_list[0],
                                quant_fea=False)
        if(is_q==False):
            self.conv2 = GCNConv(hidden_units, dataset.num_classes,bias=True,improved=False)
        else:
            self.conv2 = qGCNConv(hidden_units,dataset.num_classes, num_nodes, bit,
                                para_dict=para_list[0],
                                quant_fea=True)
    def forward(self,data):
        x,edge_index = data.x, data.edge_index
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x,edge_index)
        return F.log_softmax(x,dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='GIN')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--dataset_name',type=str,default='Cora')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_units',type=int,default=16)
    parser.add_argument('--bit',type=int,default=4)
    parser.add_argument('--a_loss',type=float,default=0.1)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--max_cycle',type=int,default=2)
    parser.add_argument('--resume',type=bool,default=False)
    parser.add_argument('--store_ckpt',type=bool,default=True)
    parser.add_argument('--drop_out',type=float,default=0.5)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--is_q',type=bool,default=True)
    #############################################################################
    parser.add_argument('--lr_quant_scale_fea',type=float,default=0.1)
    parser.add_argument('--lr_quant_scale_xw',type=float,default=0.005)
    parser.add_argument('--lr_quant_scale_weight',type=float,default=0.01)
    parser.add_argument('--lr_quant_bit_fea',type=float,default=0.04)  
    #############################################################################
    # The target memory size of nodes features
    parser.add_argument('--a_storage',type=float,default=5)
    # Path to results
    parser.add_argument('--result_folder',type=str,default='result')
    # Path to checkpoint
    parser.add_argument('--check_folder',type=str,default='checkpoint')
    # Path to dataset
    parser.add_argument('--path2dataset',type=str,default='/')
    args = parser.parse_args()
    print(args)
    
    # os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_id
    dataset_name = args.dataset_name
    num_layers = args.num_layers
    hidden_units=args.hidden_units
    bit=args.bit
    max_epoch = args.max_epoch
    resume = args.resume
    path2result = args.result_folder+'/'+args.model+'_'+dataset_name
    path2check = args.check_folder+'/'+args.model+'_'+dataset_name
    if not os.path.exists(path2result):  
        os.makedirs(path2result)
    if not os.path.exists(path2check):  
        os.makedirs(path2check)
    dataset = Planetoid(root=args.path2dataset,name=dataset_name,)
    device = torch.device('cuda',args.gpu_id)
    data = dataset[0].to(device)

    # Record the accuracy
    if(resume==True):
        file_name = path2result+'/'+args.model+'_'+str(hidden_units)+'_'+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.txt'
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                for key, value in vars(args).items():
                    f.write('%s:%s\n'%(key, value))
    accu = []
    max_acc = 0
    for k in range(1):
        accu=[]
        for i in range(args.max_cycle):
            print_max_acc = 0
            if(args.model=='GCN'):
                model = GCN(hidden_units, bit, drop_out=args.drop_out,is_q=args.is_q,).to(device)
            elif(args.model=='GIN'):
                model = GIN(dataset, num_layers, hidden_units, bit, drop_out=args.drop_out, is_q=args.is_q,).to(device)
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    glorot(m.weight)
            # Group the parametes: weight, the scale of weight, the scale of feature, the bit of feature, other_paras(e.g. BN, bias) 
            weight_paras,quant_paras_bit_weight, quant_paras_bit_fea, quant_paras_scale_weight, quant_paras_scale_fea, quant_paras_scale_xw, quant_paras_bit_xw, other_paras = paras_group(model)
            if(args.model=='GIN' or args.model=='GCN'):
                optimizer = torch.optim.Adam([
                                            {'params':weight_paras}, 
                                            {'params':quant_paras_scale_weight,'lr':args.lr_quant_scale_weight,'weight_decay':0},
                                            {'params':quant_paras_scale_xw,'lr':args.lr_quant_scale_xw,'weight_decay':0},
                                            {'params':quant_paras_scale_fea,'lr':args.lr_quant_scale_fea,'weight_decay':0},
                                            {'params':quant_paras_bit_fea,'lr':args.lr_quant_bit_fea,'weight_decay':0},
                                            {'params':other_paras}
                                            ],
                                            lr=args.lr, weight_decay=args.weight_decay)
            # if (os.path.exists(path2check)):
            #     model = load_checkpoint(model,path2check)
            

            for epoch in range(args.max_epoch):
                t = tqdm(epoch)
                # Train
                model.train()
                optimizer.zero_grad()
                out = model(data)
                wByte, aByte = parameter_stastic(model,dataset,hidden_units)
                loss_a = F.relu(aByte-args.a_storage)**2
                # pdb.set_trace()
                loss_store = args.a_loss*loss_a
                loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
                if(args.is_q==True):
                    loss_store.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                
                # Val
                model.eval()
                out=model(data)
                val_loss = F.nll_loss(out[data.val_mask],data.y[data.val_mask])
                
                # Test
                model.eval()
                out = model(data)
                pred = out.argmax(dim=1)
                correct = (pred[data.test_mask]==data.y[data.test_mask]).sum()
                acc = correct/data.test_mask.sum()
                accu.append(acc)
                t.set_postfix(
                            {
                                "Train_Loss": "{:05.3f}".format(loss),
                                "Acc": "{:05.3f}".format(acc),
                                "Epoch":"{:05.1f}".format(epoch),
                            }
                        )
                t.update(1)
                if(acc>print_max_acc):
                    print_max_acc = acc
                if((acc>max_acc)&(args.store_ckpt==True)):
                    path = path2check+'/'+args.model+'_'+str(hidden_units)+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.pth.tar'
                    max_acc = acc
                    torch.save({'state_dict': model.state_dict(), 'best_accu': acc, 'hidden_units':args.hidden_units, 'layers':
                    args.num_layers, 'aByte':aByte}, path)
            print(print_max_acc)
            if(resume==True):
                f = open(file_name,'a')
                f.write(str(print_max_acc))
                f.write('\n')
        
        accu = torch.tensor(accu)
        accu = accu.view(args.max_cycle,args.max_epoch)
        _,indices = accu.max(dim=1)
        accu = accu[torch.arange(args.max_cycle, dtype=torch.long),indices]
        acc_mean = accu.mean()
        acc_std = accu.std()
        desc = "{:.3f} ± {:.3f}".format(acc_mean,acc_std)
        print("Result - {}".format(desc))
        if(resume==True):
            f = open(file_name,'a')
            f.write(desc)
            f.write('\n')
    # Observe the learned bitwidth
    state = torch.load(path)
    dict=state['state_dict']
    analysis_bit(data,dict,all_positive=True)
    print("Result - {}".format(desc))
    