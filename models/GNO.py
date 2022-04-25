import torch
import torch.nn as nn
import torch_geometric.nn as nng

class Conv(nng.MessagePassing):
    def __init__(self, hparams, kernel):
        super(Conv, self).__init__(aggr = 'mean')
        
        self.dim_rep = hparams['dim_rep']    

        self.kernel = kernel     

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_i, x_j, edge_attr):
        conv = self.kernel(edge_attr).view(-1, self.dim_rep, self.dim_rep)
        out = torch.matmul(conv, x_j.unsqueeze(-1)).squeeze(-1)
        return out

class GNO(nn.Module):
    def __init__(self, hparams, conv, encoder, decoder):
        super(GNO, self).__init__()

        self.dim_rep = hparams['dim_rep']
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.nb_iter = hparams['nb_iter']
        self.res = hparams['res']
        self.bn_bool = hparams['bn_bool']

        self.conv = conv

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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.rep_in(x)
        edge_attr_list = [edge_attr.clone()]

        for i in range(self.nb_iter - 1):
            if self.res:
                x_res = x.clone()

            edge_attr_list.append(edge_attr_list[i].clone())
            x = self.conv(x, edge_index, edge_attr_list[i])
            if self.bn_bool:
                x = self.bn[i](x)
            x = self.activation(x)

            if self.res:
                x = x + x_res
            y = self.rep_out(x) # compute the velocity at iteration t

            edge_attr_list[i + 1][:, 2:5] = y[edge_index[0], :3] - y[edge_index[1], :3]

        if self.res:
            x_res = x.clone()

        x = self.conv(x, edge_index, edge_attr_list[-1])
        if self.bn_bool:
            x = self.bn[-1](x)

        if self.res:   
            x = x + x_res
            
        out = self.rep_out(x)

        return out