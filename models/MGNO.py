import random

import torch
import torch.nn as nn
import torch_geometric.nn as nng

class Conv(nng.MessagePassing):
    def __init__(self, hparams, kernel, local_nn = None):
        super(Conv, self).__init__(aggr = 'mean')
        
        self.dim_rep = hparams['dim_rep'] 

        self.kernel = kernel
        self.local_nn = local_nn  

    def forward(self, x, edge_index, edge_attr):    
        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_j, edge_attr):
        conv = self.kernel(edge_attr).view(-1, self.dim_rep, self.dim_rep)
        out = torch.matmul(conv, x_j.unsqueeze(-1)).squeeze(-1)
        if self.local_nn is not None:
            out = self.local_nn(out)

        return out

def DownSample(id_sample, x, base_x, ratio, r, batch, training, max_neighbors, aggr = True):
    # Subsample carefully with batch information
    m = batch.max()
    id_batch = []
    n = int(len(x)*ratio)

    for i in range(m + 1):
        id = (batch == i).nonzero(as_tuple = True)[0]
        id = random.sample(list(id), n)
        id = torch.tensor(id, dtype = torch.long)
        id_batch.append(id)
    
    id_batch = torch.cat(id_batch).long()
    id_sample_new = id_sample + [id_batch] # Keep track of the subsampling

    # Find back the base features of the sampled points through U-Net (i.e. position, velocity, pressure...)
    base_x_new, batch_new = base_x.clone(), batch.clone()

    for id in id_sample_new[:-1]:
        base_x_new = base_x_new[id]
    
    x_sampled, batch_sampled = base_x_new[id_sample_new[-1]], batch_new[id_sample_new[-1]]

    # Create graph
    points = base_x_new[:, :2]
    points_sampled = x_sampled[:, :2]
    if aggr:        
        cluster = nng.nearest(points, points_sampled, batch_x = batch_new, batch_y = batch_sampled)
        y, batch_sampled = nng.avg_pool_x(cluster, x, batch_new)

    else:
        y = x[id_sample_new[-1]]

    if training:
        edge_index = nng.radius_graph(x = points_sampled.detach(), r = r, batch = batch_sampled.detach(), loop = True, max_num_neighbors = max_neighbors)
    else:
        edge_index = nng.radius_graph(x = points_sampled.detach(), r = r, batch = batch_sampled.detach(), loop = True, max_num_neighbors = 512)

    x_i, x_j = x_sampled[edge_index[0], :2], x_sampled[edge_index[1], :2]
    v_i, v_j = x_sampled[edge_index[0], 2:4], x_sampled[edge_index[1], 2:4]
    p_i, p_j = x_sampled[edge_index[0], 4:5], x_sampled[edge_index[1], 4:5]
    v_inf = x_sampled[edge_index[0], 2:3]
    sdf_i, sdf_j = x_sampled[edge_index[0], 5:6], x_sampled[edge_index[1], 5:6]

    edge_attr = torch.cat([x_i - x_j, v_i - v_j, p_i - p_j, sdf_i, sdf_j, v_inf], dim = 1)

    return id_sample_new, y, edge_index, edge_attr, batch_sampled

def UpSample(id_sample, x, x_upsampled, base_x, batch):
    # Upsample
    base_x_up, batch_up = base_x.clone(), batch.clone()
    base_n = base_x.size(0)
    n_sampled = x_upsampled.size(0)
    i = 0

    # Find back the base features of the sampled points through U-Net (i.e. position, velocity, pressure...)
    while (base_n != n_sampled):
        base_x_up, batch_up = base_x_up[id_sample[i]], batch_up[id_sample[i]]
        i = i + 1
        base_n = base_x_up.size(0)

    base_x_sub, batch_sub = base_x_up[id_sample[i]], batch_up[id_sample[i]]
    points = base_x_sub[:, :2]
    points_upsampled = base_x_up[:, :2]
    cluster = nng.nearest(points_upsampled, points, batch_x = batch_up, batch_y = batch_sub)

    y = x[cluster]

    return y

class MGNO(nn.Module):
    def __init__(self, hparams, list_conv, encoder, decoder):
        super(MGNO, self).__init__()
        self.L = len(list_conv)
        self.dim_rep = hparams['dim_rep']
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.nb_iter = hparams['nb_iter']
        self.aggr = hparams['aggr']
        self.res = hparams['res']
        self.max_neighbors = hparams['max_neighbors']
        self.pool_ratio = hparams['pool_ratio']
        self.r = hparams['list_r']
        self.bn_bool = hparams['bn_bool']

        self.list_conv = list_conv
        self.rep_in = encoder
        self.rep_out = decoder

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for i in range(self.nb_iter):
                self.bn.append(nng.BatchNorm(
                    in_channels = self.dim_rep,
                    track_running_stats = False
                ))

        self.activation = nn.Identity()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch.long()

        z = x.clone()      
        edge_attr_list = [edge_attr.clone()]
        edge_index_sub = edge_index.clone()

        y = self.rep_in(z)        

        for i in range(self.nb_iter - 1):   
            
            if self.res:
                y_res = y.clone()
            edge_attr_list.append(edge_attr_list[i].clone())
            batch_new = batch.clone()
            id_sample = []
            t = self.list_conv[0](y, edge_index, edge_attr_list[i])
            X = [t.clone()]

            # Downward
            for l in range(1, self.L):
                id_sample, y, edge_index_sub, edge_attr_sub, batch_new = DownSample(id_sample, y, z, self.pool_ratio[l - 1], self.r[l], batch_new,
                                                                                                self.training, self.max_neighbors, aggr = self.aggr)
                t = self.list_conv[l](y, edge_index_sub, edge_attr_sub)
                X.append(t.clone())

            # Upward
            y = X[-1]
            for l in range(self.L - 1, 0, -1):
                y = UpSample(id_sample, y, X[l - 1], z, batch) + X[l - 1]

            if self.bn_bool:
                y = self.bn[i](y)

            if self.res:
                y = self.activation(y)/self.L + y_res
            else:
                y = self.activation(y)
            out = self.rep_out(y) # compute the velocity at iteration t
            edge_attr_list[i + 1][:, 2:5] = out[edge_index[0], :3] - out[edge_index[1], :3]

        # Last iteration
        if self.res:
            y_res = y.clone()
        batch_new = batch.clone()
        id_sample = []
        t = self.list_conv[0](y, edge_index, edge_attr_list[-1])
        X = [t.clone()]

        # Downward
        for l in range(1, self.L):
            id_sample, y, edge_index_sub, edge_attr_sub, batch_new = DownSample(id_sample, y, z, self.pool_ratio[l - 1], self.r[l], batch_new,
                                                                                    self.training, self.max_neighbors, aggr = self.aggr)
            t = self.list_conv[l](y, edge_index_sub, edge_attr_sub)
            X.append(t.clone())

        # Upward
        y = X[-1]
        for l in range(self.L - 1, 0, -1):
            y = UpSample(id_sample, y, X[l - 1], z, batch) + X[l - 1]

        if self.bn_bool:
            y = self.bn[-1](y)

        if self.res:
            out = self.rep_out(y/self.L + y_res)
        else:
            out = self.rep_out(y)

        return out