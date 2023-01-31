#*************************************************************************
#   > Filename    : get_scale_index.py
#   > Description : The implement of Nearest Neighbor Strategy
#*************************************************************************
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree,remove_self_loops

def get_deg_index(fea,edge_index):
    """
    fea         : input data
    edge_index  : the edge connections in graphs
    """
    deg_interval = [5,2]
    with torch.no_grad():
        row,col = edge_index
        deg = degree(row,fea.size()[0])
        index = deg.new_zeros(deg.size())
        index[deg>=deg_interval[0]] = int(0)
        index[(deg<deg_interval[0])&(deg>=deg_interval[1])] = int(1)
        index[deg<deg_interval[1]] = int(2)
    return index.long()

def get_scale_index(fea,deg_index,scale,bit):
    interval=[10,100]
    with torch.no_grad():
        scale = scale.abs()
        scale = scale.transpose(0,1)
        bit = bit.round()
        bit = bit.transpose(0,1)
        x = fea.abs()
        if(x.size()[1]==1):
            x = x.reshape(-1)
        else:
            x = x.max(dim=1)[0]
        x_deg1 = x[deg_index==0].unsqueeze(1)
        x_deg2 = x[deg_index==1].unsqueeze(1)
        x_deg3 = x[deg_index==2].unsqueeze(1)
        qmax = (2**(bit-1))*scale
        if(len(x.size())==1):
            x = x.unsqueeze(1)
        scale_index1 = ((x_deg1-qmax[:,0:interval[0]])**2).argmin(dim=1)
        scale_index2 = ((x_deg2-qmax[:,interval[0]:interval[1]])**2).argmin(dim=1)+interval[0]
        scale_index3 = ((x_deg3-qmax[:,interval[1]:])**2).argmin(dim=1)+interval[1]
        scale_index = deg_index.new_zeros(deg_index.size())
        scale_index[deg_index==0] = scale_index1
        scale_index[deg_index==1] = scale_index2
        scale_index[deg_index==2] = scale_index3
    return scale_index

def get_scale_index_uniform(fea,deg_index,scale,bit):
    with torch.no_grad():
        # Ensure the step size is positive
        scale = scale.abs()
        scale = scale.transpose(0,1)
        # Ensure the bitwidth is non-positive and integer
        bit = bit.abs().round()
        bit = bit.transpose(0,1)
        x = fea.abs()
        if(x.size()[1]==1):
            x = x.reshape(-1)
        else:
            x = x.max(dim=1)[0].unsqueeze(1)
        # Cal the quantization max value
        qmax = (2**(bit-1))*scale
        if(len(x.size())==1):
            x = x.unsqueeze(1)
        # Find the nearest quantization parameters
        scale_index = ((x-qmax)**2).argmin(dim=1)
    return scale_index

# Assign the quantization only according the degrees of nodes
def get_scale_index_naive(fea,edge_index,num_deg):
    row,col = edge_index
    deg = degree(row,fea.size(0))
    scale_index = fea.new_zeros(fea.size(0))
    # The nodes whose degrees larger than num_deg use the same quantization parameters
    scale_index[deg>=(num_deg-1)] = num_deg - 1
    scale_index[deg<(num_deg-1)] = deg[deg<(num_deg-1)]
    scale_index = scale_index.type(torch.long)
    return scale_index

    


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

