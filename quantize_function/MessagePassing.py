import inspect
from collections import OrderedDict

import torch
from torch.nn import Parameter, Module, ModuleDict
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (
    add_remaining_self_loops,
    remove_self_loops,
    add_self_loops,
    softmax,
)

from utils.quant_utils import (
    msg_special_args,
    aggr_special_args,
    update_special_args,
)
from utils.quant_utils import scatter_
from quantize_function.u_quant_func_bit_debug import *



class MessagePassingMultiQuant(Module):
    def __init__(
        self,
        aggr="add",
        flow="source_to_target",
        node_dim=0, in_features=1, bit=4, para_dict={'gama_int':0.07,'gama_std':0.05},
        quant_fea=True,
        out_features=16
    ):
        super(MessagePassingMultiQuant, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max"]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        # [index:,dim_size:,]
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        # [aggr_out:,x:,]
        self.__update_params__ = OrderedDict(self.__update_params__)
        # [x:,]
        self.__update_params__.popitem(last=False)
        # {'x_j'}-{'size_i', 'size_j', 'edge_index', 'size', 'edge_index_j', 'edge_index_i'} = {'x_j'}
        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__args__ = set().union(msg_args, aggr_args, update_args)
        if(quant_fea==False):
            self.q_xw = nn.Identity()
        else:
            # self.q_xw = u_quant_fea(in_features,bit, quant_method, gama_init=gama_init, gama_std=gama_std,fea_list=fea_list)
            self.q_xw = u_quant_xw(1,out_features,bit, alpha_init=0.01,alpha_std=0.01)

        
        

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (
                    f"Encountered node tensor with size "
                    f"{tensor.size(self.node_dim)} in dimension {self.node_dim}, "
                    f"but expected size {size[index]}."
                )
            )

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out["edge_index"] = edge_index
        out["edge_index_i"] = edge_index[i]
        out["edge_index_j"] = edge_index[j]
        out["size"] = size
        out["size_i"] = size[i]
        out["size_j"] = size[j]

        # Add special aggregate arguments.
        out["index"] = out["edge_index_i"]
        out["dim_size"] = out["size_i"]

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"Required parameter {key} is empty.")
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2
        # Quantize the input before aggregation
        x = kwargs['x'].T
        x = self.q_xw(x)
        x = x.T
        kwargs['x'] = x
        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        msg = self.message(**msg_kwargs)
        # msg = self.quant_fea(msg)
        

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        aggrs = self.aggregate(msg, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        updates = self.update(aggrs, **update_kwargs)

        return updates

    def message(self, x_j):  # pragma: no cover
        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def update(self, inputs):  # pragma: no cover
        return inputs

class GINConvMultiQuant(MessagePassingMultiQuant):
    def __init__(self, nn, eps=0, train_eps=False, in_features=1, bit=4, para_dict={'gama_int':0.1,'gama_std':0.1}, quant_fea=True, out_features=16, **kwargs):
        super(GINConvMultiQuant, self).__init__(
            aggr="add", in_features=in_features, bit=bit, para_dict=para_dict, 
            quant_fea=quant_fea,
            out_features=out_features,
            **kwargs
        )
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
            torch.nn.init.zeros_(self.eps)
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return self.nn((1 + self.eps) * x + aggr_out)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)