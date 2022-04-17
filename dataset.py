import numpy as np
import pyvista as pv

import itertools
import json

import torch
from torch_geometric.data import Data

from tqdm import tqdm

def Dataset(path_in, set = 'train', norm = False, coef_norm = None):
    '''
    Create a list of simulation to input in a PyTorch Geometric DataLoader.

    Args:
        path_in (string): Global location of the simulations
        set (string, optional): Type of dataset generated, choose between 'train', 'val', 'test'. Default: ``'train'``
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned. 
            Moreover, the dataset will be normalized by these quantities (without the geometry indicator). Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None. 
            The dataset generated will be normalized by those quantites. Default: ``None``
    '''
    print('Loading the ' + set + 'set.', end = '\n')

    if norm and coef_norm is not None:
        raise ValueError('If coef_norm is not None and norm is True, the normalization will be done via coef_norm')

    with open(path_in + 'manifest.json', 'r') as f:
        manifest = json.load(f)[set]     
    path_in = path_in + set + '/'
    dataset = []

    for s in tqdm(manifest):

        # Get the 3D mesh, add the signed distance function and slice it to return in 2D
        mesh = pv.read(path_in + s + '/' + s + '_500.vtk')
        airfoil = pv.read(path_in + s + '/' + 'aerofoil/aerofoil_500.vtk')
        mesh.compute_implicit_distance(airfoil, inplace = True)
        mesh2d = mesh.slice(normal = [0, 0, 1])

        # Get the vertices of each faces
        faces = mesh2d.faces.reshape(-1, 4)
        faces = faces[:, 1:]

        # From faces, compute the undirected edges of the underlying graphs
        edges_index = []
        for i in range(faces.shape[0]):
            edges_index.append(list(itertools.permutations(faces[i], r = 2)))

        edges_index = np.unique(np.array(edges_index).reshape(-1, 2).T, axis = 1)

        # Get the initial mesh for inputs
        mesh_init = pv.read(path_in + s + '/' + s + '_0.vtk')
        mesh2d_init = mesh_init.slice(normal = [0, 0, 1]) # Slice to return 2D

        # Geometry information
        geom = mesh2d.point_data['implicit_distance'][:, None] # Signed distance function

        # Define the inputs and the targets
        init = np.hstack([mesh2d_init.points[:, :2], mesh2d_init.point_data['U'][:, :2], np.zeros_like(geom), geom])
        target = np.hstack([mesh2d.point_data['U'][:, :2], mesh2d.point_data['p'][:, None], mesh2d.point_data['nut'][:, None]]) # velocity (v_x, v_y), pressure and turbulent viscosity

        # Put everything in tensor 
        edge_index = torch.tensor(edges_index, dtype = torch.long)
        x = torch.tensor(init, dtype = torch.float)
        y = torch.tensor(target, dtype = torch.float)

        # Graph definition
        data = Data(x = x, edge_index = edge_index, y = y)

        dataset.append(data)
        
    results = dataset

    if norm and coef_norm is None:

        # Compute normalization
        mean_in = []
        mean_out = []
        mean_nut = []
        for data in dataset:
            bool_surf = (data.y[:, 0] == 0)
            mean_in.append(data.x)
            mean_out.append(data.y)
            mean_nut.append(data.y[bool_surf, 3:4])
        mean_in = np.vstack(mean_in)
        mean_out = np.vstack(mean_out)
        mean_nut = np.vstack(mean_nut)
        std_in = mean_in.std(axis = 0)
        std_out = mean_out.std(axis = 0)
        std_nut = mean_nut.std(axis = 0)
        mean_in =  mean_in.mean(axis = 0)
        mean_out = mean_out.mean(axis = 0)
        mean_nut = mean_nut.mean(axis = 0)

        # Normalize
        for data in dataset:
            bool_surf = (data.y[:, 0] == 0)
            data.x = (data.x - mean_in)/(std_in + 1e-8)
            data.y = (data.y - mean_out)/(std_out + 1e-8)
            data.y[bool_surf, 3] = (data.y[bool_surf, 3]*(std_out[3] + 1e-8) + mean_out[3] - mean_nut)/(std_nut + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out, mean_nut, std_nut)
        results = [dataset, coef_norm]
    
    elif coef_norm is not None:

        # Normalize
        for data in dataset:
            bool_surf = (data.y[:, 0] == 0)
            data.x = (data.x - coef_norm[0])/(coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2])/(coef_norm[3] + 1e-8)
            data.y[bool_surf, 3] = (data.y[bool_surf, 3]*(coef_norm[3][3] + 1e-8) + coef_norm[2][3] - coef_norm[4])/(coef_norm[5] + 1e-8)
        
        results = dataset
    
    return results