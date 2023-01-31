#*************************************************************************
#   > Filename    : lr_gcn_mnist_bit.py
#   > Description : GCN trained on MNIST or CIFAR10
#*************************************************************************
from ast import parse
import os
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, Identity, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree,remove_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset,Planetoid,GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch.autograd.function import InplaceFunction
from torch_geometric.nn import GCNConv,GINConv,global_mean_pool,global_add_pool, TopKPooling
from tqdm import tqdm
import random
import numpy as np
from quantize_function.u_quant_gc_bit_debug import *
import argparse

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
    params_id = list(map(id,quant_paras_bit_fea))+list(map(id,quant_paras_bit_weight))+list(map(id,quant_paras_scale_weight))+list(map(id,quant_paras_scale_fea))+list(map(id,weight_paras))\
    +list(map(id,quant_paras_scale_xw))+list(map(id,quant_paras_bit_xw))
    other_paras = list(filter(lambda p: id(p) not in params_id, all_params))
    return weight_paras,quant_paras_bit_weight,quant_paras_bit_fea,quant_paras_scale_weight,quant_paras_scale_fea,quant_paras_scale_xw,quant_paras_bit_xw,other_paras

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

def parameter_stastic(model,dataset,hidden_units):
    w_Byte = 0
    a_Byte = 0
    for name, par in model.named_parameters():
        if(('bit' in name)&('weight' in name)):
            if('conv1' in name):
                scale = dataset.num_node_features
            else:
                scale = hidden_units
            par = torch.floor(par)
            w_Byte = scale*par.sum()/8./1024.+w_Byte
        elif(('bit' in name)&('fea' in name)):
            if('conv1' in name):
                a_scale = 0
            else:
                a_scale = hidden_units
            # a_scale = dataset.data.num_nodes
            par = torch.floor(par)
            a_Byte = a_scale*par.sum()/8./1024.+a_Byte
    return w_Byte, a_Byte

def load_checkpoint(model, checkpoint):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        new_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict.keys())}
        model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
    return model


class ResettableSequential(nn.Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
    def forward(self,input,edge_index):
        for model in self:
            input = model(input,edge_index)[0]
        return input 

class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bit, all_positive=False, add_self_loops=False, num_deg=200,
                para_dict={'alpha_init':0.01,'alpha_std':0.01,'gama_init':0.01,'gama_std':0.01},
                is_q=False, 
                uniform=True,
                init='norm'):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.self_loop = add_self_loops
        self.is_q = is_q
        if(is_q):
            self.lin = QLinear(in_channels, out_channels, num_deg,bit, para_dict=para_dict,all_positive=all_positive,
                                    uniform=uniform,
                                    init=init)
        else:
            self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.q_xw = u_quant_xw(in_channels,out_channels,bit,alpha_init=0.05,alpha_std=0.02)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        torch.nn.init.zeros_(self.bias)
        torch.nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, bit_sum):

        # Add self-loops to the adjacency matrix.
        if(self.self_loop):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix.
        if(self.is_q):
            x,_,bit_sum = self.lin(x, edge_index, bit_sum)
        else:
            x = self.lin(x)
            bit_sum = 0
        # Quantize the result of the XW
        x = x.T
        x = self.q_xw(x,)
        x = x.T

        # Do not need to quantize the normalize matrix
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out=self.propagate(edge_index, x=x, norm=norm)

        out = out + self.bias
        return out, bit_sum
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GCNlayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=True,
                bit=4,
                all_positive=False,
                para_dict={'alpha_init':0.01,'alpha_std':0.01,'gama_init':0.01,'gama_std':0.01},
                num_deg=200,
                add_self_loops=False,
                is_q=False,
                uniform=True,
                init='norm'):
        super().__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        if(in_dim!=out_dim):
            self.residual = False
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = MyGCNConv(in_dim, out_dim,bit=bit,
                                all_positive=all_positive,add_self_loops=add_self_loops, 
                                is_q=is_q,
                                num_deg=num_deg,
                                para_dict=para_dict,
                                uniform=uniform,
                                init=init)

    def forward(self,x, edge_index,bit_sum):
        x_in = x.clone()
        x,bit_sum = self.conv(x,edge_index,bit_sum)
        if self.batch_norm:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.residual:
            x = x + x_in
        
        x = self.dropout(x)
        return x, bit_sum

class GCN(torch.nn.Module):
    def __init__(self,dataset, hidden_units,num_layers,dropout=0,batch_norm=True,residual=True,bit=4,add_self_loops=False,
                is_q=False,
                num_deg = 200,
                uniform = False,
                init='norm'
                ):
        super().__init__()
        self.is_q = is_q
        para_list=[[{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.70,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.6,'gama_std':0.7}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.76,'gama_std':0.68}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.6,'gama_std':0.5}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}]]
        if(is_q==True):
            self.embeding = QLinear(dataset[0].x.size()[1]+dataset[0].pos.size()[1],hidden_units, num_deg, bit,para_dict=para_list[0][0], all_positive=False,
                                    quant_fea=True,
                                    uniform=uniform,
                                    init=init)
        else:
            self.embeding = nn.Linear(dataset[0].x.size()[1]+dataset[0].pos.size()[1],hidden_units,bias=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNlayer(hidden_units,hidden_units,F.relu,dropout=0,
                                        batch_norm=batch_norm,
                                        residual=residual,
                                        bit=bit,
                                        all_positive=False,
                                        add_self_loops=add_self_loops,
                                        is_q=is_q,
                                        uniform=uniform,
                                        para_dict=para_list[i+1][0],
                                        num_deg=num_deg,
                                        init=init
                                        ))
        # The readout MLP
        if(is_q):
            self.lin1 = QLinear(hidden_units,hidden_units, num_deg, bit,para_dict=para_list[-2][0], all_positive=False,
                                    quant_fea=True,
                                    uniform=uniform,
                                    init=init)
            self.lin2 = QLinear(hidden_units,dataset.num_classes, num_deg, bit,para_dict=para_list[-1][0], all_positive=True, 
                                    quant_fea=True,
                                    uniform=uniform,
                                    init=init)
            
        else:
            self.lin1 = nn.Linear( hidden_units , hidden_units , bias=True ) 
            self.lin2 = nn.Linear( hidden_units , dataset.num_classes , bias=True )
        
        self.L = 1

    def forward(self, data):
        x, pos, edge_index, edge_weight, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        x = torch.cat((x,pos),dim=1)
        bit_sum=x.new_zeros(1)
        if(self.is_q):
            x,_,bit_sum = self.embeding(x,edge_index,bit_sum)
        else:
            x = self.embeding(x)
        for conv in self.convs:
            x,bit_sum = conv(x, edge_index,bit_sum)
        x = global_mean_pool(x, batch)
        if(self.is_q):
            x,_,bit_sum = self.lin1(x,edge_index,bit_sum)
        else:
            x = self.lin1(x)
        x = F.relu(x)
        if(self.is_q):
            x,_,bit_sum = self.lin2(x,edge_index,bit_sum)
        else:
            x = self.lin2(x)
        return x,bit_sum

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader,a_loss):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out,bit_sum = model(data)
        loss_store = a_loss*F.relu(bit_sum-1)**2
        loss_store.backward(retain_graph=True)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)[0].max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)[0]
        loss += F.cross_entropy(out, data.y.view(-1), reduction="sum").item()
    return loss / len(loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='GCN')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--act_type',type=str,default='QReLU')
    parser.add_argument('--dataset_name',type=str,default='CIFAR10')
    parser.add_argument('--num_deg',type=int,default=1000)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_units',type=int,default=146)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--bit',type=int,default=4)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--max_cycle',type=int,default=10)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--a_loss',type=float,default=0.0004)
    parser.add_argument('--init',type=str,default='norm')
    parser.add_argument('--weight_decay',type=float,default=0)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--lr_quant_scale_fea',type=float,default=1e-2)
    parser.add_argument('--lr_quant_scale_xw',type=float,default=2e-2)
    parser.add_argument('--lr_quant_scale_weight',type=float,default=1e-2)
    parser.add_argument('--lr_quant_bit_fea',type=float,default=0.0003)
    parser.add_argument('--lr_quant_bit_weight',type=float,default=0.0001) 
    parser.add_argument('--lr_step_size',type=int, default=50)
    parser.add_argument('--lr_decay_factor',type=float,default=0.5)
    parser.add_argument('--lr_schedule_patience',type=int,default=10)
    
    ###############################################################
    parser.add_argument('--resume',type=bool,default=False)
    parser.add_argument('--store_ckpt',type=bool,default=True)
    parser.add_argument('--q_qmax',type=str,default='q_4layers_lr_bit_1')
    parser.add_argument('--uniform',type=bool,default=True)
    parser.add_argument('--use_norm_quant',type=bool,default=True)
    parser.add_argument('--quant_method',type=int,default=2)
    parser.add_argument('--quant_weight',type=int,default=1)
    ###############################################################
    # The target memory size of nodes features
    parser.add_argument('--a_storage',type=float,default=1)
    # Path to results
    parser.add_argument('--result_folder',type=str,default='result')
    # Path to checkpoint
    parser.add_argument('--check_folder',type=str,default='checkpoint')
    # Path to dataset
    parser.add_argument('--path2dataset',type=str,default='/')
    args = parser.parse_args()
    ###############################################################
    model = args.model
    act_type = args.act_type
    dataset_name = args.dataset_name
    num_layers = args.num_layers
    hidden_units=args.hidden_units
    bit=args.bit
    max_epoch = args.max_epoch
    q_qmax = args.q_qmax
    quant_method = args.quant_method
    resume = args.resume
    path2result = args.result_folder+'/'+args.model+'_'+dataset_name
    path2check = args.check_folder+'/'+args.model+'_'+dataset_name
    if not os.path.exists(path2result):  
        os.makedirs(path2result)
    if not os.path.exists(path2check):  
        os.makedirs(path2check)
    ##############################################################
    setup_seed(41)
    if(args.resume==True):
        file_name = path2result+'/'+args.model+'_'+str(hidden_units)+'_'+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.txt'
        if not os.path.exists(file_name):
                with open(file_name, 'w') as f:
                    for key, value in vars(args).items():
                        f.write('%s:%s\n'%(key, value))
    train_dataset = GNNBenchmarkDataset(root=args.path2dataset,name=dataset_name,split="train")
    val_dataset = GNNBenchmarkDataset(root=args.path2dataset,name=dataset_name,split="val")
    test_dataset = GNNBenchmarkDataset(root=args.path2dataset,name=dataset_name,split="test")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # writer = SummaryWriter(log_dir=path2log)
    
    accu = []
    max_acc = 0.49
    for k in range(1):
        for i in range(args.max_cycle):
            print_max_acc=0
            model=GCN(dataset=train_dataset,hidden_units=args.hidden_units,num_layers=args.num_layers,is_q=True,
                        num_deg=args.num_deg,
                        uniform=args.uniform,
                        init = args.init
                        ).to(device)
            weight_paras,quant_paras_bit_weight, quant_paras_bit_fea, quant_paras_scale_weight, quant_paras_scale_fea, quant_paras_scale_xw, quant_paras_bit_xw, other_paras = paras_group(model)
            optimizer = torch.optim.Adam([{'params':weight_paras}, 
                                        {'params':quant_paras_scale_weight,'lr':args.lr_quant_scale_weight,'weight_decay':0},
                                        {'params':quant_paras_scale_fea,'lr':args.lr_quant_scale_fea,'weight_decay':0},
                                        {'params':quant_paras_scale_xw,'lr':args.lr_quant_scale_xw,'weight_decay':0},
                                        # {'params':quant_paras_bit_weight,'lr':args.lr_quant_bit_weight,'weight_decay':0},
                                        {'params':quant_paras_bit_fea,'lr':args.lr_quant_bit_fea,'weight_decay':0},
                                        {'params':other_paras}],
                                        lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=args.lr_decay_factor,
                                                        patience=args.lr_schedule_patience,
                                                        verbose=True,threshold=0.0001)
            # if (os.path.exists(path2check)):
            #     model = load_checkpoint(model,path2check)
            for epoch in range(args.max_epoch):
                t = tqdm(epoch)
                train_loss = train(model,optimizer,train_loader,args.a_loss)
                val_loss = eval_loss(model,val_loader)
                acc = eval_acc(model,test_loader)
                scheduler.step(val_loss)
                t.set_postfix(
                                {
                                    "Train_Loss": "{:05.3f}".format(train_loss),
                                    "Val_Loss": "{:05.3f}".format(val_loss),
                                    "Acc": "{:05.3f}".format(acc),
                                    "Epoch":"{:05.1f}".format(epoch),
                                }
                            )
                accu.append(acc)
                if(acc>print_max_acc):
                    print_max_acc = acc
                if(acc>=max_acc):
                    path = path2check+'/'+args.model+'_'+str(hidden_units)+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.pth.tar'
                    max_acc = acc
                    torch.save({'state_dict': model.state_dict(), 'best_accu': acc,}, path)
                if(args.resume==True):
                        f = open(file_name,'a')
                        f.write(str(acc))
                        f.write('\n')
            print(print_max_acc)
            if(resume==True):
                f = open(file_name,'a')
                f.write('The max accu of the {} runs is:'.format(i))
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
            f.write('The result is:')
            f.write(desc)
            f.write('\n')
        print("finish")