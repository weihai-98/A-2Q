#*************************************************************************
#   > Filename    : gat_nc_lsb.py
#   > Description : GAT for Cora, CiteSeer, and PubMed
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
from torch_geometric.utils import add_self_loops, degree,softmax,remove_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset,Planetoid
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction, Function
from torch_geometric.nn import GCNConv,GINConv,GATConv
from torch_geometric.nn.inits import glorot, zeros
from tqdm import tqdm
import argparse
from quantize_function.u_quant_func_bit_debug import *
import pdb

def paras_group(model):
    all_params = model.parameters()
    weight_paras=[]
    quant_paras_bit_weight = []
    quant_paras_bit_fea = []
    quant_paras_bit_gat_fea = []
    quant_paras_bit_gat = []
    quant_paras_scale_weight = []
    quant_paras_scale_fea = []
    quant_paras_scale_gat_fea = []
    quant_paras_scale_gat = []
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
        elif('q'in name and 'gat' in name and 'fea' in name and 'bit' in name):
            quant_paras_bit_gat_fea+=[para]
        elif('q'in name and 'gat' in name and 'fea' in name and 'bit' not in name):
            quant_paras_scale_gat_fea+=[para]
        elif('q'in name and 'gat' in name and 'fea' not in name and 'bit' not in name):
            quant_paras_scale_gat+=[para]
        elif('q'in name and 'gat' in name and 'fea' not in name and 'bit' in name):
            quant_paras_bit_gat+=[para]
        elif('weight' in name and 'quant' not in name ):
            weight_paras+=[para]
    params_id = list(map(id,quant_paras_bit_fea))+list(map(id,quant_paras_bit_weight))+list(map(id,quant_paras_scale_weight))+list(map(id,quant_paras_scale_fea))+list(map(id,weight_paras))\
                +list(map(id,quant_paras_bit_gat_fea))+list(map(id,quant_paras_bit_gat))+list(map(id,quant_paras_scale_gat))+list(map(id,quant_paras_scale_gat_fea))
    other_paras = list(filter(lambda p: id(p) not in params_id, all_params))
    return weight_paras,quant_paras_bit_weight,quant_paras_bit_fea,quant_paras_bit_gat,quant_paras_bit_gat_fea,quant_paras_scale_weight,quant_paras_scale_fea,quant_paras_scale_gat,quant_paras_scale_gat_fea,other_paras
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def parameter_stastic(model,dataset,hidden_units,heads):
    w_Byte = torch.tensor(0)
    a_Byte = torch.tensor(0)
    for name, par in model.named_parameters():
        if(('bit' in name)&('weight' in name)):
            if('conv1' in name):
                scale = dataset.num_node_features
            else:
                scale = hidden_units
            # par = torch.floor(par)
            w_Byte = scale*par.sum()/8./1024.+w_Byte
        elif(('bit' in name)&('fea' in name) and ('gat' not in name)):
            a_scale = hidden_units
            # a_scale = dataset.data.num_nodes
            # par = torch.floor(par)
            a_Byte = a_scale*par.sum()/8./1024.+a_Byte
        elif(('bit' in name)&('fea' in name) and ('gat' in name)):
            a_scale = hidden_units
            a_Byte = heads*a_scale*par.sum()/8./1024.+a_Byte
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

class qGATConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        num_nodes=1,
        bit=4,
        all_positive=True,
        quant_fea=True,
        is_q=True,
    ):
        super(qGATConv, self).__init__(
            aggr="add",node_dim=0
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_q = is_q

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        
        if(self.is_q):
            self.lin = QLinear(in_channels,heads*out_channels,num_nodes,bit=bit,all_positive=all_positive,quant_fea=quant_fea)
        else:
            self.lin = nn.Linear(in_channels,heads*out_channels,)
        # attn->(1,heads,2*out_c)->(heads,2*out_c), each head has a quantization step size, bitwidth=4
        self.q_gat_attn = u_quant_weight(1,heads,bit=bit,)
        self.q_gat_fea = u_quant_fea(dim_fea=num_nodes,bit=bit,)
        # (-1,heads,1)->(-1,heads) each head has a quantization step size, bitwidth=4
        self.q_gat_edge = u_quant_weight(1,heads,bit=bit,)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None,first_layer=True):
        # quantizing input
        fea = self.lin(x)
        if(self.is_q):
            fea_q = self.q_gat_fea(fea)
        else:
            fea_q = fea

        if size is None and torch.is_tensor(fea_q):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=fea_q.size(0))

        return self.propagate(
            edge_index, size=size, x=fea_q,
        )

    def message(self, edge_index_i, x_i, x_j, size_i):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        att = self.att
        # Quantize the attn weight
        if(self.is_q):
            att = self.att.squeeze(0)
            att = self.q_gat_attn(att)
            att = att.unsqueeze(0)
        

        if x_i is None:
            alpha = (x_j * att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)
        
        # Quantize the edge weight
        if(self.is_q):
            alpha = alpha.T
            alpha = self.q_gat_edge(alpha)
            alpha = alpha.T

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

class GAT(torch.nn.Module):
    def __init__(self, hidden_units, bit, is_q=False, heads=4, drop_out=0, drop_attn=0,):
        super().__init__()
        num_nodes = dataset.data.num_nodes
        self.is_q = is_q
        self.drop_out=drop_out
        self.conv1 = qGATConv(dataset.num_node_features, hidden_units, heads=heads,dropout=drop_attn,num_nodes=num_nodes, bit=bit, all_positive=False,
                                quant_fea=False,is_q=is_q,)
        self.conv2 = qGATConv(hidden_units*heads,dataset.num_classes, heads=1, dropout=drop_attn, num_nodes=num_nodes, bit=bit, all_positive=True,
                                quant_fea=True,is_q=is_q)
    def forward(self,data):
        x,edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv1(x,edge_index,first_layer=True)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x,edge_index,first_layer=False)
        return F.softmax(x,dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='GAT')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--act_type',type=str,default='QReLU')
    parser.add_argument('--dataset_name',type=str,default='Cora')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_units',type=int,default=8)
    parser.add_argument('--heads',type=int,default=8)
    parser.add_argument('--bit',type=int,default=4)
    parser.add_argument('--a_loss',type=float,default=0.5)
    parser.add_argument('--max_epoch',type=int,default=250)
    parser.add_argument('--max_cycle',type=int,default=2)
    parser.add_argument('--resume',type=bool,default=False)
    parser.add_argument('--store_ckpt',type=bool,default=True)
    parser.add_argument('--weight_decay',type=float,default=1e-3)
    parser.add_argument('--lr',type=float,default=0.005)
    parser.add_argument('--is_q',type=bool,default=True)
    parser.add_argument('--drop_out',type=float,default=0.6)
    parser.add_argument('--drop_attn',type=float,default=0.3)
    #############################################################################
    # Set learning rate for different parameters group
    parser.add_argument('--lr_quant_scale_fea',type=float,default=0.05)
    parser.add_argument('--lr_quant_scale_weight',type=float,default=0.005)
    parser.add_argument('--lr_quant_scale_gat_fea',type=float,default=0.05)
    parser.add_argument('--lr_quant_scale_gat',type=float,default=0.05)
    parser.add_argument('--lr_quant_bit_fea',type=float,default=0.1)
    #############################################################################
    # The target memory size of nodes features
    parser.add_argument('--a_storage',type=float,default=10)
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

    if(resume==True):
        file_name = path2result+'/'+args.model+'_'+str(args.heads)+'_'+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.txt'
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                for key, value in vars(args).items():
                    f.write('%s:%s\n'%(key, value))
    accu = []
    max_acc = 0.0
    for k in range(1):
        accu = []
        for i in range(args.max_cycle):

            print_max_acc = 0
            model = GAT(hidden_units, bit, is_q=args.is_q, heads=args.heads, drop_out=args.drop_out, drop_attn=args.drop_attn).to(device)
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    glorot(m.weight)
            # divide the parameters
            weight_paras,\
            _,quant_paras_bit_fea,_,_,\
            quant_paras_scale_weight,quant_paras_scale_fea,quant_paras_scale_gat,quant_paras_scale_gat_fea,\
            other_paras = paras_group(model)
            optimizer = torch.optim.Adam([{'params':weight_paras}, 
                                        {'params':quant_paras_scale_weight,'lr':args.lr_quant_scale_weight,'weight_decay':0},
                                        {'params':quant_paras_scale_fea,'lr':args.lr_quant_scale_fea,'weight_decay':0},
                                        {'params':quant_paras_bit_fea,'lr':args.lr_quant_bit_fea,'weight_decay':0},
                                        {'params':quant_paras_scale_gat,'lr':args.lr_quant_scale_gat,'weight_decay':0},
                                        {'params':quant_paras_scale_gat_fea,'lr':args.lr_quant_scale_gat_fea,'weight_decay':0},
                                        {'params':other_paras}],
                                        lr=args.lr, weight_decay=args.weight_decay)
            # if (os.path.exists(path2check)):
            #     model = load_checkpoint(model,path2check)
            
            
            for epoch in range(args.max_epoch):
                t = tqdm(epoch)
                # Train
                model.train()
                optimizer.zero_grad()
                out = model(data)
                wByte, aByte = parameter_stastic(model,dataset,hidden_units,args.heads)
                loss_a = F.relu(aByte-args.a_storage)**2
                loss_store = args.a_loss*loss_a
                loss = F.cross_entropy(out[data.train_mask],data.y[data.train_mask])
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
            f.write('****************************************')
            f.write('\n')
            f.write(desc)
            f.write('\n')
        torch.cuda.empty_cache()
    state = torch.load(path)
    dict=state['state_dict']
    analysis_bit(data,dict,all_positive=True)
    print("Result - {}".format(desc))