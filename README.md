# A^2Q method
Dear readers, we have provided the source code to reproduce the results of our paper. Due to time constraints, we first provide the code for the main experimental section (i.e., Section 4.2 and Section 4.3) in the main text to reproduce the results reported in our work. However, the code for the ablation experiments has not yet been organized. Therefore, the other code used in our paper will soon be provided. 
We first list the dependencies of our code and then provide the commands to run the experiments mentioned in our main text.

## Dependencies
The dependencies of our experiments are:

`Python : 3.6.13`

`Pytorch : 1.8.0`

`CUDA : 10.2`

`Torch Geometric : 2.0.3`

`Torch Cluster : 1.5.9`

`Torch Scatter : 2.0.8`

`Torch Sparse : 0.6.10`

`Torch Spline Conv : 1.2.1`

`ogb : 1.3.3`

`Numpy : 1.19.2`

`Scikit Learn : 0.24.2`

`Matplotlib : 3.3.4`

## Train command

The `--path2dataset` is the path to dataset directory, the `--check_folder` is the directory storing the checkpoint file, and the `--result_folder` is the directory storing the accuracy/loss 
results of the model during training process. You can add these arguments according to your needs.
The train commands are as follow.

### Node-level

---GCN-Cora---
```bash
python node_level_1.py --lr_quant_bit_fea 0.1 --lr_quant_scale_fea 0.04 --a_loss 2.5 --lr_quant_scale_weight 0.02 --lr_quant_scale_xw 0.008 --drop_out 0.35 --weight_decay 0.02 --dataset_name Cora --model GCN
```

---GCN-CiteSeer---
```bash
python node_level_1.py --lr_quant_bit_fea 0.1 --lr_quant_scale_fea 0.04 --a_loss 1.5 --lr_quant_scale_weight 0.008 --lr_quant_scale_xw 0.008 --drop_out 0.5 --weight_decay 0.015 --dataset_name CiteSeer --model GCN 
```

---GIN-Cora---
```bash
python node_level_1.py --lr_quant_bit_fea 0.02 --lr_quant_scale_fea 0.05 --a_loss 2 --lr_quant_scale_weight 0.005  --lr_quant_scale_xw 0.005 --dataset_name Cora --model GIN
```

---GIN-CiteSeer---
```bash
python node_level_1.py --lr_quant_bit_fea 0.02 --lr_quant_scale_fea 0.05 --a_loss 0.5 --lr_quant_scale_weight 0.005 --lr_quant_scale_xw 0.005 --dataset_name CiteSeer --model GIN
```

---GCN-PubMed---
```bash
python node_level_1.py --lr_quant_bit_fea 0.02 --lr_quant_scale_fea 0.005 --a_loss 0.1 --lr_quant_scale_weight 0.005 --dataset_name PubMed --model GCN
```

---GAT-Cora---
```bash
python gat_nc_lsb.py --weight_decay 1e-3 --lr 0.01 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.005 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.3 --drop_out 0.6 --drop_attn 0.6 --dataset_name Cora
```

---GAT-CiteSeer---
```bash
python gat_nc_lsb.py --weight_decay 1e-3 --lr 0.005 --lr_quant_scale_fea 0.05 --lr_quant_scale_weight 0.01 --lr_quant_scale_gat_fea 0.05 --lr_quant_scale_gat 0.005 --lr_quant_bit_fea 0.1 --a_loss 0.25 --drop_out 0.6 --drop_attn 0.6 --dataset_name CiteSeer
```

---GAT-PubMed---
```bash
python gat_nc_lsb.py --lr_quant_scale_fea 0.005 --lr_quant_scale_weight 0.001 --lr_quant_bit_fea 0.015 --a_loss 0.025 --is_q True --lr_quant_scale_gat_fea 0.005 --lr_quant_scale_gat 0.005 --drop_out 0.6 --drop_attn 0.3 --lr 0.005 --weight_decay 1e-3 --dataset_name PubMed 
```

---GCN-ogbn-arxiv---
```bash
python gcn_ogb_arxiv.py --lr_quant_scale_fea 1e-2 --lr_quant_scale_weight 1e-3 --lr_quant_bit_fea 1e-2 --a_loss 1e-4 --dataset_name ogbn-arxiv
```

### Graph-level

---GIN-REDDIT-BINARY---
```bash
python lr_gin_reddit_binary.py --a_loss 1e-3 --lr_quant_scale_fea 2e-2 --lr_quant_scale_xw 1e-2 --lr_quant_scale_weight 2e-2 --lr_quant_bit_fea 8e-3 
```

---GCN-MNIST---
```bash
python lr_gcn_mnist_bit.py --lr_quant_scale_fea 1e-3 --lr_quant_scale_weight 1e-3 --lr_quant_scale_xw 1e-2 --lr_quant_bit_fea 1e-3 --a_loss 0.001 --init norm --dataset_name MNIST
```

---GCN-CIFAR10---
```bash
python lr_gcn_mnist_bit.py --lr_quant_scale_fea 1e-2 --lr_quant_scale_xw 1e-2 --lr_quant_scale_weight 1e-3 --lr_quant_bit_fea 3e-4 --a_loss 2e-4 --init uniform --dataset_name CIFAR10
```

---GIN-MNIST---
```bash
python lr_gin_mnist_bit.py --lr_quant_scale_fea 5e-3 --lr_quant_scale_xw 5e-3 --lr_quant_scale_weight 5e-4 --lr_quant_bit_fea 1e-4 --init uniform --a_loss 5e-5 --dataset_name MNIST
```

---GIN-CIFAR10---
```bash
python lr_gin_mnist_bit.py --lr_quant_scale_fea 1e-2 --lr_quant_scale_weight 1e-3 --lr_quant_scale_xw 1e-2 --lr_quant_bit_fea 2e-4 --init norm --a_loss 2e-4 --dataset_name CIFAR10
```

---GAT-MNIST---
```bash 
python lr_gat_mnist.py --lr_quant_scale_fea 1e-2 --lr_quant_scale_weight 1e-4 --lr_quant_bit_fea 1e-2 --a_loss 2e-4 --lr_quant_scale_gat_fea 1e-2
--lr_quant_scale_gat_attn 1e-4 --lr_quant_scale_gat_edge 5e-2 --dataset_name MNIST
```

---GAT-CIFAR10---
```bash 
python lr_gat_mnist.py --lr_quant_scale_fea 1e-2 --lr_quant_scale_weight 1e-3 --lr_quant_bit_fea 5e-4 --a_loss 2e-5 --lr_quant_scale_gat_fea 1e-2
--lr_quant_scale_gat_attn 1e-3 --lr_quant_scale_gat_edge 0.1 --dataset_name CIFAR10
```

---GCN-ZINC---
```bash
python lr_gcn_zinc_bit.py --a_loss 1e-2 --lr_quant_scale_fea 1e-2 --lr_quant_scale_xw 5e-2 --lr_quant_scale_weight 1e-3 --lr_quant_bit_fea 1e-3 
``` 

## Citing this Work
