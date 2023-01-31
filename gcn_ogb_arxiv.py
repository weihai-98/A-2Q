#*************************************************************************
#   > Filename    : gcn_ogb_arxiv.py
#   > Description : GCN for ogbn-arxiv
#*************************************************************************
import argparse

import torch
from torch import Tensor
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils.logger import Logger
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree,softmax,remove_self_loops
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

from quantize_function.u_quant_func_bit_debug import *
from utils.gcn_norm import gcn_norm
from utils.quant_utils import analysis_bit

def parameter_stastic(model,dataset,hidden_units):
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
        elif(('bit' in name)&(('fea' in name)|('xw' in name))):
            a_scale = hidden_units
            # a_scale = dataset.data.num_nodes
            # par = torch.floor(par)
            a_Byte = a_scale*par.sum()/8./1024.+a_Byte
    return w_Byte, a_Byte

def paras_group(model):
    all_params = model.parameters()
    weight_paras=[]
    quant_paras_bit_weight = []
    quant_paras_bit_fea = []
    quant_paras_bit_xw = []
    quant_paras_scale_weight = []
    quant_paras_scale_fea = []
    quant_paras_scale_xw = []
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
    return weight_paras,quant_paras_bit_weight,quant_paras_bit_fea,quant_paras_bit_xw,quant_paras_scale_weight,quant_paras_scale_fea,quant_paras_scale_xw,other_paras



class qGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_nodes, bit, all_positive=False,
                para_dict={'alpha_init':0.01,'alpha_std':0.02,'gama_init':0.1,'gama_std':0.2},
                quant_fea=True):
        super().__init__(aggr='add')  
        num_nodes = dataset.data.num_nodes
        self.lin = QLinear(in_channels, out_channels, num_nodes, bit, all_positive=all_positive, para_dict=para_dict, quant_fea=quant_fea)
        # Quant the result of XW
        self.q_xw = u_quant_fea(num_nodes,bit, gama_init=para_dict['gama_init'],gama_std=para_dict['gama_std'],quant_fea=False)
        self.num_nodes = num_nodes
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        # zeros(self.bias)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        x = self.q_xw(x)

        # Step 3: Compute normalization matrix.
        edge_index = gcn_norm(edge_index, None, x.size(0),)
        return self.propagate(edge_index, x=x,)
    
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class qGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout,is_q=True,bit=4):
        super(qGCN, self).__init__()
        num_nodes = dataset.data.num_nodes 
        self.convs = torch.nn.ModuleList()
        if(is_q):
            self.convs.append(
                qGCNConv(dataset.num_node_features, hidden_channels, num_nodes, bit,quant_fea=False,all_positive=False)
            ) 
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            if(is_q):
                self.convs.append(
                qGCNConv(hidden_channels, hidden_channels, num_nodes, bit,quant_fea=True,all_positive=True)
                ) 
            else:
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        if(is_q):
            self.convs.append(
                qGCNConv(hidden_channels, out_channels, num_nodes, bit,quant_fea=True,all_positive=True)
                ) 
        else:
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)



def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model',type=str,default='GCN')
    parser.add_argument('--dataset_name',type=str,default='ogbn-arxiv')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay',type=float,default=0)
    #############################################################################
    parser.add_argument('--resume',type=bool,default=True)
    parser.add_argument('--store_ckpt',type=bool,default=True)
    parser.add_argument('--bit',type=int,default=4)
    parser.add_argument('--is_q',type=bool,default=True)
    #############################################################################
    parser.add_argument('--a_loss',type=float,default=0.0001)
    parser.add_argument('--lr_quant_scale_fea',type=float,default=0.01)
    parser.add_argument('--lr_quant_scale_xw',type=float,default=0.05)
    parser.add_argument('--lr_quant_scale_weight',type=float,default=0.001)
    parser.add_argument('--lr_quant_bit_fea',type=float,default=0.04)
    parser.add_argument('--lr_quant_bit_xw',type=float,default=0.04)
    parser.add_argument('--lr_quant_bit_weight',type=float,default=0.01)  
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
    
    dataset_name = args.dataset_name
    num_layers = args.num_layers
    hidden_units=args.hidden_channels
    bit=args.bit
    max_epoch = args.epochs
    resume = args.resume
    path2result = args.result_folder+'/'+args.model+'_'+args.dataset_name
    path2check = args.check_folder+'/'+args.model+'_'+args.dataset_name
    if not os.path.exists(path2result):  
        os.makedirs(path2result)
    if not os.path.exists(path2check):  
        os.makedirs(path2check)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name=args.dataset_name,root=args.path2dataset,
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    
    
    if(args.resume==True):
        file_name = path2result+'/'+args.model+'_'+str(hidden_units)+'_'+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.txt'
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                for key, value in vars(args).items():
                    f.write('%s:%s\n'%(key, value))

    model = qGCN(data.num_features, args.hidden_channels,
                dataset.num_classes, args.num_layers,
                args.dropout,is_q=args.is_q,bit=args.bit).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)
    accu = []
    max_acc=0.5
    for run in range(args.runs):
        model.reset_parameters()
        weight_paras,quant_paras_bit_weight, quant_paras_bit_fea, quant_paras_bit_xw, quant_paras_scale_weight, \
            quant_paras_scale_fea, quant_paras_scale_xw, other_paras = paras_group(model)
        
        optimizer = torch.optim.Adam([{'params':weight_paras}, 
                                        {'params':quant_paras_scale_weight,'lr':args.lr_quant_scale_weight,'weight_decay':0},
                                        {'params':quant_paras_scale_fea,'lr':args.lr_quant_scale_fea,'weight_decay':0},
                                        {'params':quant_paras_scale_xw,'lr':args.lr_quant_scale_fea,'weight_decay':0},
                                        # {'params':quant_paras_bit_weight,'lr':args.lr_quant_bit_weight,'weight_decay':0},
                                        {'params':quant_paras_bit_fea,'lr':args.lr_quant_bit_fea,'weight_decay':0},
                                        # {'params':quant_paras_bit_xw,'lr':args.lr_quant_bit_fea,'weight_decay':0},
                                        {'params':other_paras}],
                                        lr=args.lr, weight_decay=args.weight_decay)
        print_max_acc=0
        for epoch in range(1, 1 + args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.adj_t)[train_idx]
            wByte, aByte = parameter_stastic(model,dataset,args.hidden_channels)
            loss_a = F.relu(aByte-args.a_storage)**2
            loss_store = args.a_loss*loss_a
            loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
            if(args.is_q==True):
                loss_store.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            # loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            accu.append(test_acc)
            if(test_acc>max_acc):
                max_acc = test_acc
                path = path2check+'/'+args.model+'_'+str(hidden_units)+'_on_'+dataset_name+'_'+str(bit)+'bit-'+str(max_epoch)+'.pth.tar'
                torch.save({'state_dict': model.state_dict(), 'best_accu': test_acc,}, path)
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            if(args.resume==True):
                f = open(file_name,'a')
                f.write(str(test_acc))
                f.write('\n')
        logger.print_statistics(run)
        if(args.resume==True):
            f = open(file_name,'a')
            f.write(str(print_max_acc))
            f.write('\n')
    
    accu = torch.tensor(accu)
    accu = accu.view(args.runs,args.epochs)
    _,indices = accu.max(dim=1)
    accu = accu[torch.arange(args.runs, dtype=torch.long),indices]
    acc_mean = accu.mean()
    acc_std = accu.std()
    desc = "{:.3f} ± {:.3f}".format(acc_mean,acc_std)
    print("Result - {}".format(desc))
    if(args.resume==True):
        f = open(file_name,'a')
        f.write(desc)
        f.write('\n')
    logger.print_statistics()
    
    state = torch.load(path)
    dict=state['state_dict']
    analysis_bit(data,dict,all_positive=True,name='ogbn-arxiv')



