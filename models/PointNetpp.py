import torch
import torch.nn as nn
import torch_geometric.nn as nng
from models.MLP import MLP

class SAModule(nn.Module):
    def __init__(self, ratio, r, local_nn, global_nn, max_neighbors = 64):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.max_neighbors = max_neighbors
        self.conv = nng.PointConv(local_nn = local_nn, global_nn = global_nn, add_self_loops = True)

    def forward(self, x, pos, training):
        idx = nng.fps(pos, ratio = self.ratio)
        if training:
            row, col = nng.radius(pos, pos[idx], r = self.r, max_num_neighbors = self.max_neighbors)
        else:
            row, col = nng.radius(pos, pos[idx], r = self.r, max_num_neighbors = 512)
        edge_index = torch.stack([col, row], dim = 0)

        x_dst = x[idx].clone()
        y = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        return y, pos[idx]

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, x_skip, pos_skip):
        x = nng.knn_interpolate(x, pos, pos_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip

class PointNetpp(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(PointNetpp, self).__init__()

        self.base_nb = hparams['base_nb']
        self.pool_ratio = hparams['pool_ratio']
        self.r = hparams['list_r']
        self.dim_enc = hparams['dim_enc']
        self.L = hparams['nb_scale']
        self.max_neighbors = hparams['max_neighbors']

        self.encoder = encoder
        self.decoder = decoder

        self.SAModule = nn.ModuleList()
        self.nn_list_local = nn.ModuleList()
        self.nn_list_local.append(MLP([self.dim_enc + 2, self.base_nb, self.base_nb*2]))
        self.nn_list_global = nn.ModuleList()
        self.nn_list_global.append(MLP([self.base_nb*2, self.base_nb*2, self.base_nb*2]))

        self.FPModule = nn.ModuleList()
        self.nn_list = nn.ModuleList()
        self.nn_list.append(MLP([self.base_nb*3, self.base_nb*2, self.dim_enc]))

        for l in range(self.L - 1):
            self.base_nb *= 2
            self.SAModule.append(SAModule(self.pool_ratio[l], self.r[l], self.nn_list_local[l], self.nn_list_global[l], max_neighbors = self.max_neighbors))
            if l != self.L - 2:
                self.nn_list_local.append(MLP([self.base_nb + 2, self.base_nb*2, self.base_nb*2]))
                self.nn_list_global.append(MLP([self.base_nb*2, self.base_nb*2, self.base_nb*2]))

            self.FPModule.append(FPModule(3, self.nn_list[l]))
            self.nn_list.append(MLP([self.base_nb*3, self.base_nb*2, self.base_nb]))
        
    def forward(self, data):
        x = data.x.float()
        pos = x[:, :2].clone()

        pos_list = []        
        pos_list.append(pos.clone())

        z = self.encoder(x)
        z_skip = []        
        z_skip.append(z.clone())        

        for l in range(self.L - 1):
            z, pos = self.SAModule[l](z, pos, self.training)
            z_skip.append(z.clone())
            pos_list.append(pos.clone())

        for l in range(self.L - 2, -1, -1):
            z, pos = self.FPModule[l](z, pos, z_skip[l], pos_list[l])
        
        out = self.decoder(z)       

        return out