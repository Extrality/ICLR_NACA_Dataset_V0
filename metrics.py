import numpy as np
import torch
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader

import pyvista as pv
import json
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import train

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def rsquared(predict, true):
    '''
    Args:
        predict (tensor): Predicted values, shape (N, *)
        true (tensor): True values, shape (N, *)

    Out:
        rsquared (tensor): Coefficient of determination of the prediction, shape (*,)
    '''
    mean = true.mean(dim = 0)
    return 1 - ((true - predict)**2).sum(dim = 0)/((true - mean)**2).sum(dim = 0)

# 2D
def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    n = out_order_points.shape[0]
    idx = np.zeros(n)
    for i in range(n):
        cond = (out_order_points[i] == in_order_points)
        cond = cond[:, 0]*cond[:, 1]        
        idx[i] = np.argwhere(cond)[0][0]
    idx = idx.astype('int')

    assert (in_order_points[idx] == out_order_points).all()

    return quantity_to_reordered[idx]

def WallShearStress(Jacob_U, normals, nut):
    nu = 1e-5
    S = .5*(Jacob_U + Jacob_U.transpose(0, 2, 1))
    S = S - S.trace(axis1 = 1, axis2 = 2).reshape(-1, 1, 1)*np.eye(2)[None]/3
    ShearStress = 2*(nu + nut).reshape(-1, 1, 1)*S
    results = (ShearStress*normals.reshape(-1, 1, 2)).sum(axis = 2)

    return results

def WallPressure(p, normals):
    return p.reshape(-1, 1)*normals

def Compare_WSS_WP(model, dataset, path_in, device, set = 'val', coef_norm = None):
    '''
    Args:
        model (nn.Module): model that compute the predictions.
        dataset (list): list of the data in the dataset.
        path_in (str): path where the datasets and the manifest are.
        device (str): CPU or GPU device.
        set (str): type of the dataset, choose between 'train', 'val', 'test'. Default : 'train'.
        coef_norm (tuple): if not None, normalize the outputs of the model. Default: None.
    '''

    with open(path_in + 'manifest.json', 'r') as f:
        manifest = json.load(f)[set]    
    path_in = path_in + set + '/'

    Wss_real_list = []
    Wss_predict_list = []
    WP_real_list = []
    WP_predict_list = []
    for i, s in enumerate(tqdm(manifest)):
        # Prepare the data
        mesh = pv.read(path_in + s + '/' + s + '_500.vtk')
        surf = pv.read(path_in + s + '/aerofoil/aerofoil_500.vtk')

        surf = surf.compute_normals(flip_normals = False)

        mesh = mesh.slice(normal = [0, 0, 1])
        surf = surf.slice(normal = [0, 0, 1])

        surf = surf.compute_cell_sizes(area = False, volume = False)

        mesh = mesh.compute_derivative(scalars = 'U', gradient = 'real_grad', preference = 'point')

        bool_surf = (mesh.point_data['U'][:, 0] == 0)

        point_mesh = mesh.points[bool_surf, :2]
        point_surf = surf.points[:, :2]

        surf_grad = mesh.point_data['real_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2]
        surf_nut = mesh.point_data['nut'][bool_surf]
        surf_p = mesh.point_data['p'][bool_surf]
        surf_normals = surf.point_data['Normals'][:, :2]

        surf_grad = reorganize(point_mesh, point_surf, surf_grad)
        surf_nut = reorganize(point_mesh, point_surf, surf_nut)
        surf_p = reorganize(point_mesh, point_surf, surf_p)

        # Compute the real wall shear stress and Wall pressure
        Wss_real = WallShearStress(surf_grad, surf_normals, surf_nut)
        WP_real = WallPressure(surf_p, surf_normals)
        surf.point_data['Wss_real'] = Wss_real
        surf.point_data['WP_real'] = WP_real

        # Compute the predicted fields
        dataloader = DataLoader([dataset[i].clone()], batch_size = 1)
        with torch.no_grad():
            for data_load in dataloader:
                data = data_load.to(device)
                if coef_norm is not None:
                    out = (model(data).detach().cpu()*coef_norm[3] + coef_norm[2]).numpy()
                    out[bool_surf, 3] = (out[bool_surf, 3] - coef_norm[2][3])*coef_norm[5]/coef_norm[3][3] + coef_norm[4]
                else:
                    out = model(data).detach().cpu().numpy()

        # Compute the predicted wall shear stress and Wall pressure
        mesh.point_data['U'] = np.hstack([out[:, :2], np.zeros((out.shape[0], 1))])
        mesh = mesh.compute_derivative(scalars = 'U', gradient = 'predict_grad', preference = 'point')

        surf_grad = mesh.point_data['predict_grad'].reshape(-1, 3, 3)[bool_surf, :2, :2]
        surf_p = out[bool_surf, 2]
        surf_grad, surf_p = reorganize(point_mesh, point_surf, surf_grad), reorganize(point_mesh, point_surf, surf_p)
        surf_nut_predict = out[bool_surf, 3]
        surf_nut_predict = reorganize(point_mesh, point_surf, surf_nut_predict)
        
        Wss_predict = WallShearStress(surf_grad, surf_normals, surf_nut_predict)
        WP_predict = WallPressure(surf_p, surf_normals)
        surf.point_data['Wss_predict'] = Wss_predict
        surf.point_data['WP_predict'] = WP_predict
    
        # Integrate the surface quantities in order to compute global coefficients.
        surf = surf.ptc()
        global_Wss_real = (surf.cell_data['Wss_real']*surf.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        global_WP_real = (surf.cell_data['WP_real']*surf.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        global_Wss_predict = (surf.cell_data['Wss_predict']*surf.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)
        global_WP_predict = (surf.cell_data['WP_predict']*surf.cell_data['Length'].reshape(-1, 1)).sum(axis = 0)

        Wss_real_list.append(tuple([reorganize(point_surf, point_mesh, Wss_real), global_Wss_real]))
        WP_real_list.append(tuple([reorganize(point_surf, point_mesh, WP_real), global_WP_real])) 
        Wss_predict_list.append(tuple([reorganize(point_surf, point_mesh, Wss_predict), global_Wss_predict]))
        WP_predict_list.append(tuple([reorganize(point_surf, point_mesh, WP_predict), global_WP_predict]))      

    return Wss_real_list, Wss_predict_list, WP_real_list, WP_predict_list

def Plot_global_coef(global_coeffs, std = True):
    wss_global_coef_list_real = global_coeffs[:, :, 0].mean(axis = 0)
    wss_global_coef_list_predict = global_coeffs[:, :, 1].mean(axis = 0)
    WP_global_coef_list_real = global_coeffs[:, :, 2].mean(axis = 0)
    WP_global_coef_list_predict = global_coeffs[:, :, 3].mean(axis = 0)

    if std:
        wss_global_std_predict = global_coeffs[:, :, 1].std(axis = 0)
        WP_global_std_predict = global_coeffs[:, :, 3].std(axis = 0)

    wss_idx_sort = np.argsort(wss_global_coef_list_real[:, 0], axis = 0)
    wss_idy_sort = np.argsort(wss_global_coef_list_real[:, 1], axis = 0)
    WP_idx_sort = np.argsort(WP_global_coef_list_real[:, 0], axis = 0)
    WP_idy_sort = np.argsort(WP_global_coef_list_real[:, 1], axis = 0)

    sns.set()
    fig, ax = plt.subplots(2, 2, figsize = (20, 10))
    ax[0, 0].plot(wss_global_coef_list_real[wss_idx_sort, 0], label = 'Ground truth')        
    ax[0, 0].plot(wss_global_coef_list_predict[wss_idx_sort, 0], label = 'Predicted')
    if std:
        ax[0, 0].fill_between(
            range(len(wss_idx_sort)), 
            wss_global_coef_list_predict[wss_idx_sort, 0] - wss_global_std_predict[wss_idx_sort, 0],
            wss_global_coef_list_predict[wss_idx_sort, 0] + wss_global_std_predict[wss_idx_sort, 0],
            alpha = 0.8,
            color = 'y'
            )
    # ax[0, 0].set_yscale('log')
    ax2 = ax[0, 0].twinx()
    ratio = np.abs((wss_global_coef_list_real[:, 0] - wss_global_coef_list_predict[:, 0])/wss_global_coef_list_real[:, 0])     
    ax2.bar(np.arange(ratio.shape[0]), ratio[wss_idx_sort], width = 0.9, edgecolor = 'blue', alpha = 0.2, label = 'Relative error')
    ax2.grid(False)
    ax2.set_yscale('log')
    ax[0, 0].set_title('x-WallShearStress')
    ax[0, 0].set_xticks(range(len(wss_idx_sort)))
    ax[0, 0].set_xticklabels(wss_idx_sort)
    ax[0, 0].legend(loc = 'best')
    ax2.legend(loc = 'upper right');

    ax[0, 1].plot(wss_global_coef_list_real[wss_idy_sort, 1], label = 'Ground truth')        
    ax[0, 1].plot(wss_global_coef_list_predict[wss_idy_sort, 1], label = 'Predicted')
    if std:
        ax[0, 1].fill_between(
            range(len(wss_idy_sort)), 
            wss_global_coef_list_predict[wss_idy_sort, 1] - wss_global_std_predict[wss_idy_sort, 1],
            wss_global_coef_list_predict[wss_idy_sort, 1] + wss_global_std_predict[wss_idy_sort, 1],
            alpha = 0.8,
            color = 'y'
            )
    # ax[0, 1].set_yscale('log')
    ax2 = ax[0, 1].twinx()
    ratio = np.abs((wss_global_coef_list_real[:, 1] - wss_global_coef_list_predict[:, 1])/wss_global_coef_list_real[:, 1])        
    ax2.bar(np.arange(ratio.shape[0]), ratio[wss_idy_sort], width = 0.9, edgecolor = 'blue', alpha = 0.2, label = 'Relative error')
    ax2.grid(False)
    ax2.set_yscale('log')
    ax[0, 1].set_title('y-WallShearStress')
    ax[0, 1].set_xticks(range(len(wss_idy_sort)))
    ax[0, 1].set_xticklabels(wss_idy_sort)
    ax[0, 1].legend(loc = 'best')
    ax2.legend(loc = 'upper right');

    ax[1, 0].plot(WP_global_coef_list_real[WP_idx_sort, 0], label = 'Ground truth')
    ax[1, 0].plot(WP_global_coef_list_predict[WP_idx_sort, 0], label = 'Predicted')
    if std:
        ax[1, 0].fill_between(
            range(len(WP_idx_sort)), 
            WP_global_coef_list_predict[WP_idx_sort, 0] - WP_global_std_predict[WP_idx_sort, 0],
            WP_global_coef_list_predict[WP_idx_sort, 0] + WP_global_std_predict[WP_idx_sort, 0],
            alpha = 0.8,
            color = 'y'
            )
    # ax[1, 0].set_yscale('log')
    ax2 = ax[1, 0].twinx()
    ratio = np.abs((WP_global_coef_list_real[:, 0] - WP_global_coef_list_predict[:, 0])/WP_global_coef_list_real[:, 0])        
    ax2.bar(np.arange(ratio.shape[0]), ratio[WP_idx_sort], width = 0.9, edgecolor = 'blue', alpha = 0.2, label = 'Relative error')
    ax2.grid(False)
    ax2.set_yscale('log')
    ax[1, 0].set_title('x-WallPressure')
    ax[1, 0].set_xticks(range(len(WP_idx_sort)))
    ax[1, 0].set_xticklabels(WP_idx_sort)
    ax[1, 0].set_xlabel('Index of geometries')
    ax[1, 0].legend(loc = 'best')
    ax2.legend(loc = 'upper right');

    ax[1, 1].plot(WP_global_coef_list_real[WP_idy_sort, 1], label = 'Ground truth')
    ax[1, 1].plot(WP_global_coef_list_predict[WP_idy_sort, 1], label = 'Predicted')
    if std:
        ax[1, 1].fill_between(
            range(len(WP_idy_sort)), 
            WP_global_coef_list_predict[WP_idy_sort, 1] - WP_global_std_predict[WP_idy_sort, 1],
            WP_global_coef_list_predict[WP_idy_sort, 1] + WP_global_std_predict[WP_idy_sort, 1],
            alpha = 0.8,
            color = 'y'
            )
    # ax[1, 1].set_yscale('log')
    ax2 = ax[1, 1].twinx()
    ratio = np.abs((WP_global_coef_list_real[:, 1] - WP_global_coef_list_predict[:, 1])/WP_global_coef_list_real[:, 1])        
    ax2.bar(np.arange(ratio.shape[0]), ratio[WP_idy_sort], width = 0.9, edgecolor = 'blue', alpha = 0.2, label = 'Relative error')
    ax2.grid(False)
    ax2.set_yscale('log')
    ax[1, 1].set_title('y-WallPressure')
    ax[1, 1].set_xticks(range(len(WP_idy_sort)))
    ax[1, 1].set_xticklabels(WP_idy_sort)
    ax[1, 1].set_xlabel('Index of geometries')
    ax[1, 1].legend(loc = 'best')
    ax2.legend(loc = 'upper right');

    fig.savefig('metrics/global.png', dpi = 150, bbox_inches = 'tight')

def Results_test(device, models, r, set = 'val', path_in = 'datasets/', std = True):
    # To test
    coef_norm = torch.load('datasets/normalization')
    test_dataset = torch.load('datasets/' + set + '_dataset')

    for data in test_dataset:
        data.edge_index = nng.radius_graph(x = data.x[:, :2].to(device), r = r, loop = True, max_num_neighbors = 512).cpu()
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
    test_loader = DataLoader(test_dataset)   

    MSEs = []
    globs = []
    scores = []
    for model in models:
        score, score_var, score_surf_var, score_vol_var, score_surf, score_vol = train.test(device, model, test_loader, criterion = 'MSE')
        score = score_surf + score_vol
        scores.append([score_vol, score_surf])
        print('The global MSE score on the ' + set + ' set is {0:.3f}, the surface MSE is {1:.3f} and the volumetric MSE is {2:.3f}.'.format(*[score, score_surf, score_vol]))
        print('The MSE score per variables on the ' + set + ' set and on the surface is : vx :{0:.3f}, vy : {1:.3f}, p : {2:.3f}, nut : {3:.3f}.'.format(*score_surf_var))
        print('The MSE score per variables on the ' + set + ' set and on the volume is : vx :{0:.3f}, vy : {1:.3f}, p : {2:.3f}, nut : {3:.3f}.'.format(*score_vol_var))

        Wss_real, Wss_predict, WP_real, WP_predict = Compare_WSS_WP(model, test_dataset, path_in, device,
                                                                                set = set, coef_norm = coef_norm)

        MSE = []
        glob = []
        for i in range(len(Wss_real)):
            glob_Wss_real = Wss_real[i][1]
            glob_Wss_predict = Wss_predict[i][1]
            glob_WP_real = WP_real[i][1]
            glob_WP_predict = WP_predict[i][1]

            MSE.append([(glob_Wss_real - glob_Wss_predict)**2,  
                                (glob_WP_real - glob_WP_predict)**2])

            glob.append([glob_Wss_real, glob_Wss_predict, glob_WP_real, glob_WP_predict])
        

        MSE = np.array(MSE)
        glob = np.array(glob)
        # print('The mean MSE for the global coefficients is: \n', MSE.mean(axis = 0))

        MSEs.append(MSE)
        globs.append(glob)
    
    globs = np.array(globs)
    MSEs = np.array(MSEs)
    scores = np.array(scores)
    MSE = MSEs.mean(axis = 1)
    print('The mean MSE for the global coefficients is: \n', MSE.mean(axis = 0), end = '\n')
    print('The std MSE for the global coefficients is: \n', MSE.std(axis = 0), end = '\n')

    Plot_global_coef(globs, std = std)
    with open('metrics/score.json', 'w') as f:
        json.dump(
            {
                'mean_score': scores.mean(axis = 0),
                'std_score': scores.std(axis = 0),
                'mean_glob': MSE.mean(axis = 0),
                'std_glob': MSE.std(axis = 0),
            }, f, indent = 12, cls = NumpyEncoder
        )