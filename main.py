import argparse, yaml, os
import torch
import torch_geometric.nn as nng
import train, metrics
from dataset import Dataset
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'The model you want to train, choose between GraphSAGE, GAT, PointNet, GKO, PointNet++, GUNet, MGKO', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-s', '--set', help = 'Set on which you want the scores and the global coefficients plot, choose between val and test (default: val)', default = 'val', type = str)
args = parser.parse_args()

if os.path.exists('datasets/train_dataset'):
    train_dataset = torch.load('datasets/train_dataset')
    val_dataset = torch.load('datasets/val_dataset')
    coef_norm = torch.load('datasets/normalization')
else:
    train_dataset, coef_norm = Dataset('datasets/', norm = True)
    torch.save(train_dataset, 'datasets/train_dataset')
    torch.save(coef_norm, 'datasets/normalization')
    val_dataset = Dataset('datasets/', set = 'val', coef_norm = coef_norm)
    torch.save(val_dataset, 'datasets/val_dataset')
    test_dataset = Dataset('datasets/', set = 'test', coef_norm = coef_norm)
    torch.save(test_dataset, 'datasets/test_dataset')

# Cuda
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

with open('params.yaml', 'r') as f: # hyperparameters of the model
    hparams = yaml.safe_load(f)[args.model]

for data in val_dataset:
    data.edge_index = nng.radius_graph(x = data.x[:, :2].to(device), r = hparams['r'], loop = True, max_num_neighbors = 512).cpu()
    x, edge_index = data.x, data.edge_index
    x_i, x_j = x[edge_index[0], 0:2], x[edge_index[1], 0:2]
    v_i, v_j = x[edge_index[0], 2:4], x[edge_index[1], 2:4]
    v_inf = x[edge_index[0], 2:3]
    p_i, p_j = x[edge_index[0], 4:5], x[edge_index[1], 4:5]
    sdf_i, sdf_j = x[edge_index[0], 5:6], x[edge_index[1], 5:6]

    data.edge_attr = torch.cat([x_i - x_j, v_i - v_j, p_i - p_j, sdf_i, sdf_j, v_inf], dim = 1)

    bool_surf = torch.isclose(torch.tensor(0.), data.x[:, 2]*coef_norm[1][2] + coef_norm[0][2], atol = 1e-3)
    data.surf = torch.nonzero(bool_surf).flatten()
    data.vol = torch.nonzero(~bool_surf).flatten()

del(x, edge_index, x_i, x_j, v_i, v_j, v_inf, p_i, p_j, sdf_i, sdf_j)   

from models.MLP import MLP
models = []
for i in range(args.nmodel):
    encoder = MLP(hparams['encoder'], batch_norm = False)
    decoder = MLP(hparams['decoder'], batch_norm = False)

    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE
        model = GraphSAGE(hparams, encoder, decoder)

    elif args.model == 'GAT':
        from models.GAT import GAT
        model = GAT(hparams, encoder, decoder)
    
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        model = PointNet(hparams, encoder, decoder)

    elif args.model == 'GKO':
        from models.GKO import Conv, GKO
        kernel = MLP(hparams['kernel'], batch_norm = False)
        conv = Conv(hparams, kernel)
        model = GKO(hparams, conv, encoder, decoder)

    elif args.model == 'PointNet++':
        from models.PointNetpp import PointNetpp
        model = PointNetpp(hparams, encoder, decoder)

    elif args.model == 'GUNet':
        from models.GUNet import GUNet
        model = GUNet(hparams, encoder, decoder)

    elif args.model == 'MGKO':
        from models.MGKO import Conv, MGKO
        list_kernel = nn.ModuleList()
        if hparams['local_nn'] is not None:
            list_local_nn = nn.ModuleList()
        list_conv = nn.ModuleList()
        for i in range(len(hparams['list_r'])):
            list_kernel.append(MLP(hparams['kernel'], batch_norm = False))
            if hparams['local_nn'] is not None:
                list_local_nn.append(MLP(hparams['local_nn'], batch_norm = False))
                list_conv.append(Conv(hparams, list_kernel[i], list_local_nn[i]))
            else:
                list_conv.append(Conv(hparams, list_kernel[i], None))
        model = MGKO(hparams, list_conv, encoder, decoder)
    

    path = 'metrics/' # path where you want to save log and figures
    model = train.main(device, train_dataset, val_dataset, model, hparams, path, criterion = 'MSE_weighted', val_iter = 10, reg = args.weight)
    models.append(model)

if len(models) == 1:
    metrics.Results_test(device, models, hparams['r'], set = args.set, std = False)
else:
    metrics.Results_test(device, models, hparams['r'], set = args.set)