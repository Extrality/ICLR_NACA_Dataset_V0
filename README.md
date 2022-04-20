# ICLR_NACA_Dataset_V0
In this repository, you will find the different python scripts to train the available models on the 2D incompressible steady-state RAND solutions over NACA airfoils proposed at the Geometrical and Topological Representation Learning Workshop at ICLR 2022. You can find the paper [here](https://openreview.net/forum?id=rqUUi4-kpeq&referrer=[the%20profile%20of%20Florent%20Bonnet](/profile?id=~Florent_Bonnet1)).

## Requirements
* Python 3.9.12
* PyTorch 1.11.0 with CUDA 11.3
* PyTorch Geometric 2.0.4
* PyVista 0.33.3
* Seaborn 0.11.2

## Training
To train a model, run main.py with the desired model architecture:

```
python main.py GraphSAGE -s val -n 10
```

Note that you must have the dataset in folder ```datasets/``` at the root of this repository, you can find the dataset [here](https://data.isir.upmc.fr/extrality/2D_RANS_NACA_Dataset.zip). You can change the parameters of the models and the training in the ```params.yaml``` file. You will find the different plots and scores at the root of the ```metrics``` folder.

## Usage
```usage: main.py [-h] [-n NMODEL] [-w WEIGHT] [-s SET] model

positional arguments:
  model                 The model you want to train, chose between GraphSAGE, GAT, PointNet, GKO, PointNet++, GUNet, MGKO.

optional arguments:
  -h, --help            show this help message and exit
  -n NMODEL, --nmodel NMODEL
                        Number of trained models for standard deviation estimation (default: 1)
  -w WEIGHT, --weight WEIGHT
                        Weight in front of the surface loss (default: 1)
  -s SET, --set SET     Set on which you want the scores and the global coefficients plot, choose between val and test (default: val)
 ```
 
 ## Results
The different results for each are given in the ```metrics``` folder and are displayed under the form of a table in the original paper.
