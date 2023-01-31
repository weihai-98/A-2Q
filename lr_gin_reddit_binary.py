#*************************************************************************
#   > Filename    : make_gc_bit_great_again.py
#   > Description : GIN trained on REDDIT-BINARY
#*************************************************************************
from quantize_function.u_quant_gc_bit_debug import *
from quantize_function.MessagePassing_gc_bit import GINConvMultiQuant
import os
import numpy as np
import random
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
from torch_geometric.nn import GCNConv,GINConv,global_mean_pool,TopKPooling
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from quantize_function.get_scale_index import get_deg_index, get_scale_index
import time
import argparse

class relu(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self,x,edge_index,bit_sum):
        x[x<0] = 0
        return x,edge_index,bit_sum

class bn(nn.Module):
    def __init__(self,hidden_units):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_units)
    def forward(self,x,edge_index,bit_sum):
        x = self.bn(x)
        return x,edge_index,bit_sum

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

class ResettableSequential(nn.Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
    def forward(self,input,edge_index,bit_sum):
        for model in self:
            input,_,bit_sum = model(input,edge_index,bit_sum)
        return input,bit_sum 
        



class qGIN(nn.Module):
    def __init__(self, dataset, num_layers, hidden_units, bit, num_deg=1000, is_q=False,
                    uniform=False,init='norm'):
        super(qGIN, self).__init__()
        gin_layer = GINConvMultiQuant
        self.bit = bit
        para_list=[[{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.70,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.6,'gama_std':0.7}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.76,'gama_std':0.68}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.6,'gama_std':0.5}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.6,'gama_std':0.3}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}],
                   [{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1}]]
        if(is_q):
            # As the DQ, we either don't quantize the input features of the REDDIT-BINARY dataset because the feature is only 1-dimension.
            self.conv1 = gin_layer(
                ResettableSequential(
                    QLinear(dataset.num_features,hidden_units, num_deg, bit,para_dict=para_list[0][0], all_positive=True,
                            quant_fea=False,
                            uniform=uniform,init=init),
                    relu(),
                    QLinear(hidden_units, hidden_units, num_deg, bit, para_dict=para_list[0][1],all_positive=True, 
                            uniform=uniform,init=init),
                    relu(),
                ),
                train_eps=True, 
                in_features=num_deg, out_features=1,
                bit=bit, para_dict=para_list[0][2],quant_fea=False,uniform=uniform
            )
        else:
            self.conv1 = GINConv(
                nn.Sequential(
                    nn.Linear(dataset.num_features, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, hidden_units),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_units),
                ),
                train_eps=True, 
            )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if(is_q):
                self.convs.append(
                    gin_layer(
                        ResettableSequential(
                            QLinear(hidden_units, hidden_units, num_deg,bit, para_dict=para_list[0][0],all_positive=False,
                                    uniform=uniform,init=init),
                            relu(),
                            QLinear(hidden_units, hidden_units, num_deg,bit, para_dict=para_list[0][1], all_positive=True,
                                    uniform=uniform,init=init),
                            relu(),
                        ),
                        train_eps=True,
                        in_features=num_deg, out_features=hidden_units,
                        bit=bit, para_dict=para_list[0][2], uniform=uniform,quant_fea=True
                    )
                )
            else:
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_units, hidden_units),
                            nn.ReLU(),
                            nn.Linear(hidden_units, hidden_units),
                            nn.ReLU(),
                            nn.BatchNorm1d(hidden_units),
                        ),
                        train_eps=True,
                    )
                )
        self.bn_list = torch.nn.ModuleList()
        for i in range(num_layers):
            self.bn_list.append(nn.BatchNorm1d(hidden_units))
        if(is_q):
            self.lin1 = QLinear(hidden_units, hidden_units, num_deg, bit, para_dict=para_list[-1][0], all_positive=False,
                                        uniform=uniform,init=init)
            self.lin2 = QLinear(hidden_units, dataset.num_classes, num_deg, bit, para_dict=para_list[-1][0], all_positive=True,
                                        uniform=uniform,init=init)
        else:
            self.lin1 = nn.Linear(hidden_units, hidden_units)
            self.lin2 = nn.Linear(hidden_units, dataset.num_classes)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        bit_sum=x.new_zeros(1)
        x,bit_sum = self.conv1(x, edge_index,bit_sum)
        x = self.bn_list[0](x)
        # x,_,bit_sum = self.embeding(x,edge_index,bit_sum)
        # x = F.relu(x)
        i = 1
        for conv in self.convs:
            x,bit_sum = conv(x,edge_index,bit_sum)
            x = self.bn_list[i](x)
            i=i+1
        x = global_mean_pool(x, batch)
        x,_,bit_sum = self.lin1(x,edge_index,bit_sum)
        x = F.relu(x)
        x,_,bit_sum = self.lin2(x,edge_index,bit_sum)
        return F.log_softmax(x, dim=-1),bit_sum

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader,a_loss, a_storage=1):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out,bit_sum = model(data)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss_store = a_loss*F.relu(bit_sum-a_storage)**2
        loss_store.backward(retain_graph=True)
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

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='GIN')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--dataset_name',type=str,default='REDDIT-BINARY')
    parser.add_argument('--num_deg',type=int,default=1000)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_units',type=int,default=64)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--bit',type=int,default=4)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--max_cycle',type=int,default=2000)
    parser.add_argument('--folds',type=int,default=10)
    parser.add_argument('--weight_decay',type=float,default=0)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--a_loss',type=float,default=0.001)
    parser.add_argument('--lr_quant_scale_fea',type=float,default=0.02)
    parser.add_argument('--lr_quant_scale_xw',type=float,default=1e-2)
    parser.add_argument('--lr_quant_scale_weight',type=float,default=0.02)
    parser.add_argument('--lr_quant_bit_fea',type=float,default=0.008)
    parser.add_argument('--lr_quant_bit_weight',type=float,default=0.0001)  
    parser.add_argument('--lr_step_size',type=int, default=50)
    parser.add_argument('--lr_decay_factor',type=float,default=0.5)
    parser.add_argument('--lr_schedule_patience',type=int,default=10)
    parser.add_argument('--is_naive',type=bool,default=False)
    ###############################################################
    parser.add_argument('--resume',type=bool,default=True)
    parser.add_argument('--store_ckpt',type=bool,default=True)
    parser.add_argument('--uniform',type=bool,default=True)
    parser.add_argument('--use_norm_quant',type=bool,default=True)
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
    print(args)
    ###############################################################
    model = args.model
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
    ###############################################################
    if(args.resume==True):
        file_name = path2result+'/'+args.model+'_'+str(hidden_units)+'_'+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.txt'
        if not os.path.exists(file_name):
                with open(file_name, 'w') as f:
                    for key, value in vars(args).items():
                        f.write('%s:%s\n'%(key, value))
    if(args.dataset_name=='REDDIT-BINARY'):
        dataset = TUDataset(args.path2dataset, args.dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    
    
    
    # writer = SummaryWriter(log_dir=path2log)
    max_acc = 0.79
    for i in range(1000):
        accu = []
        accu_max = []
        # model=qGIN(dataset, args.num_layers,hidden_units=args.hidden_units,bit=args.bit, quant_method=args.quant_method).to(device)
        # model = GIN(2,32).to(device)
        # setup_seed(12345)
        
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds))):
            print_max_acc=0
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            val_dataset = dataset[val_idx]
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,drop_last=False)
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,drop_last=False)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,drop_last=False)
            k=0
            model=qGIN(train_dataset, args.num_layers,hidden_units=args.hidden_units,bit=args.bit, is_q=True,
                    num_deg=args.num_deg,
                    uniform=args.uniform).to(device)
            weight_paras,quant_paras_bit_weight, quant_paras_bit_fea, quant_paras_scale_weight, quant_paras_scale_fea, quant_paras_scale_xw, quant_paras_bit_xw, other_paras = paras_group(model)
            # quant_paras_bit.requires_grad = False
            optimizer = torch.optim.Adam([{'params':weight_paras}, 
                                        {'params':quant_paras_scale_weight,'lr':args.lr_quant_scale_weight,'weight_decay':0},
                                        {'params':quant_paras_scale_fea,'lr':args.lr_quant_scale_fea,'weight_decay':0},
                                        {'params':quant_paras_scale_xw,'lr':args.lr_quant_scale_xw,'weight_decay':0},
                                        # {'params':quant_paras_bit_weight,'lr':args.lr_quant_bit_weight,'weight_decay':0},
                                        {'params':quant_paras_bit_fea,'lr':args.lr_quant_bit_fea,'weight_decay':0},
                                        {'params':other_paras}],
                                        lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay_factor)
            
            # if (os.path.exists(path2check)):
            #     model = load_checkpoint(model,path2check)
            
            for epoch in range(args.max_epoch):
                t = tqdm(epoch)
                train_loss=0
                train_loss = train(model,optimizer,train_loader,args.a_loss, args.a_storage)
                start = time.process_time()
                val_loss = eval_loss(model,val_loader)
                end = time.process_time()
                acc = eval_acc(model,test_loader)
                # for name,param in model.named_parameters():
                #     a=param.grad
                #     if(a!=None):
                #         writer.add_histogram(tag=name+'_grad', values=a, global_step=epoch)
                scheduler.step()
                t.set_postfix(
                            {
                                "Train_Loss": "{:05.3f}".format(train_loss),
                                "Val_Loss": "{:05.3f}".format(val_loss),
                                "Acc": "{:05.3f}".format(acc),
                                "Epoch":"{:05.1f}".format(epoch),
                                "Fold":"{:05.1f}".format(fold),
                            }
                        )
                accu.append(acc)
                if(acc>max_acc):
                    path = path2check+'/'+args.model+'_'+str(hidden_units)+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.pth.tar'
                    max_acc = acc
                    torch.save({'state_dict': model.state_dict(), 'best_accu': acc,}, path)
                if(acc>print_max_acc):
                    print_max_acc = acc
                if(args.resume==True):
                    f = open(file_name,'a')
                    f.write(str(acc))
                    f.write('\n')
            if(args.resume==True):
                f = open(file_name,'a')
                f.write('The max accu of the {} runs is:'.format(i))
                f.write(str(print_max_acc))
                f.write('\n')
        # pdb.set_trace()
        accu = torch.tensor(accu)
        accu = accu.view(args.folds,args.max_epoch)
        _, argmax = accu.max(dim=1)
        accu = accu[torch.arange(args.folds, dtype=torch.long), argmax]
        acc_mean = accu.mean().item()
        acc_std = accu.std().item()
        desc = "{:.3f} ± {:.3f}".format(acc_mean,acc_std)
        if(args.resume==True):
            f = open(file_name,'a')
            f.write('The result is:')
            f.write(desc)
            f.write('\n')
    print("finish")