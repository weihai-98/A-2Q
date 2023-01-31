#*************************************************************************
#   > Filename    : qGINConv.py
#   > Description : Quantized GIN
#*************************************************************************
from quantize_function.u_quant_func_bit_debug import *
from quantize_function.MessagePassing import GINConvMultiQuant
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GINConv


class GIN(nn.Module):
    def __init__(self, dataset, num_layers, hidden_units, bit, quant_method=2, is_q=False,
                drop_out=0):
        super(GIN, self).__init__()
        gin_layer = GINConvMultiQuant
        self.bit = bit
        self.drop_out = drop_out
        num_nodes = dataset.data.num_nodes
        para_list=[[{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'alpha_init':0.01,'gama_init':0.01,'alpha_std':0.1,'gama_std':0.1},{'gama_init':0.01,'gama_std':0.1}]]
        if(is_q):
            self.conv1 = gin_layer( 
                nn.Sequential(
                    QLinear(dataset.num_node_features, hidden_units, num_nodes, bit, all_positive=True, para_dict=para_list[0][0],
                            quant_fea=True),
                    nn.ReLU(),
                ),
                train_eps=True,
                in_features=num_nodes, bit=bit, para_dict=para_list[0][0],
                quant_fea=False,
            )
        else:
            self.conv1 = GINConv( 
                nn.Sequential(
                    nn.Linear(dataset.num_node_features, hidden_units),
                    nn.ReLU(),
                ),
                train_eps=True,
            )
        self.convs = torch.nn.ModuleList()
        if(is_q):
            for i in range(num_layers - 1):
                self.convs.append(
                    gin_layer(
                        nn.Sequential(
                            QLinear(hidden_units, dataset.num_classes, num_nodes, bit, para_dict=para_list[0][0],all_positive=True,
                                    quant_fea=True),
                            nn.ReLU(),
                        ),
                        train_eps=True,
                        in_features=num_nodes, bit=bit, para_dict=para_list[0][0],
                        quant_fea=True,out_features=hidden_units
                    )
                )
        else:
            for i in range(num_layers - 1):
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_units, hidden_units),
                            nn.ReLU(),
                        ),
                        train_eps=True,
                    )
                )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x= self.conv1(x, edge_index)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        for conv in self.convs:
            x= conv(x, edge_index)
        return F.log_softmax(x, dim=1)
