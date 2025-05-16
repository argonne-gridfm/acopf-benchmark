# Benchmark of Heterogeneous Graph Neural Networks for AC-OPF in Power Grids

This repository provides a benchmark of heterogenous graph neural networks (HGNNs) for the alternating current optimal power flow (AC-OPF) problem in power grids.


## Environment setup

```bash
conda create -n fm4g python=3.10
conda activate fm4g
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 pyg \
      ipykernel tqdm joblib \
      -c pytorch -c nvidia -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv scikit-learn pandapower matplotlib lightning ipykernel pytest \
      -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 

pip install -e .
```

<!-- 

pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric
pip install torch_cluster torch_scatter torch_sparse torch_spline_conv pyg_lib -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install joblib ipykernel tqdm

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 pyg scikit-learn pandapower -c pytorch -c nvidia -c conda-forge -c pyg
pip install matpowercaseframes
 -->


<!-- ## Data

* node (size, # of features)

| Grid    | Bus     | Gen     | Load    | Shunt  |
| ------- | ------- | ------- | ------- | ------ |
| case 14 | (14, 4) | (5, 11) | (11, 2) | (1, 2) |
| case 30 | (30, 4) | (6, 11) | (21, 2) | (2, 2) |

* edge (size, # of features)

| Grid    | (bus, ac_line, bus) | (bus, transformer, bus) | (gen, gen_link, bus) | (bus, gen_link, gen) | (load, load_link, bus) | (bus, load_link, load) | (shunt, shunt_link, bus) | (bus, shunt_link, shunt) |
| ------- | ------------------- | ----------------------- | -------------------- | -------------------- | ---------------------- | ---------------------- | ------------------------ | ------------------------ |
| case 14 | (17, 9)             | (3, 11)                 | (5, 0)               | (5, 0)               | (11, 0)                | (11, 0)                | (1, 0)                   | (1, 0)                   |
| case 30 | (34, 9)             | (7, 11)                 | (6, 0)               | (6, 0)               | (21, 0)                | (21, 0)                | (2, 0)                   | (2, 0)                   | --> |

## Data processing

- Changes:
  - Customized `OPFDataset` 
    - ![image](figures/GridFM.png)
    - download/process files in parallel - save time of processing
    - remove tmp folder to save disk space, similar to the size of raw data
    - remove split of data
    - split data after loading the dataset 
    - concatenate data from different cases by `+` operator, e.g., `ds1 + ds2`


### HeteroGNN

```
HeteroGNN(
  (lin_dict): ModuleDict(
    (bus): Linear(in_features=4, out_features=64, bias=True)
    (generator): Linear(in_features=11, out_features=64, bias=True)
    (load): Linear(in_features=2, out_features=64, bias=True)
    (shunt): Linear(in_features=2, out_features=64, bias=True)
  )
  (convs): ModuleList(
    (0-2): 3 x HGTConv(-1, 64, heads=4)
  )
  (out_dict): ModuleDict(
    (bus): Linear(in_features=64, out_features=1, bias=True)
    (generator): Linear(in_features=64, out_features=1, bias=True)
  )
)
```

## Training

* supervised training: loading `OPFDataset` with optimal variables (`VA, VM, PG, QG`) and training with MSE loss
```bash
$ python train.py --task supervised
```

* unsupervised training: loading `SynOPFDataset` without labels, and train with physics-informed loss
```bash
$ python train.py --task unsupervised
```

* `config.yaml` provides the detailed settings for models, 

### Models

We provide a set of heterogeneous graph neural networks (HGNNs) for the training. 
The models are implemented in `fm4g/models/hgnn.py`. 
Table below provides the list of models

| Model             | Description                                                        | Node Feature | Edge Feature |
| ----------------- | ------------------------------------------------------------------ | ------------ | ------------ |
| HeteroConv + GCN  | Heterogeneous graph convolutional network with GCNConv as backend  | :o:          | :x:          |
| HeteroConv + SAGE | Heterogeneous graph convolutional network with SAGEConv as backend | :o:          | :x:          |
| HeteroConv + GIN  | Heterogeneous graph convolutional network with GINConv as backend  | :o:          | :x:          |
| HeteroConv + GAT  | Heterogeneous graph convolutional network with GATConv as backend  | :o:          | :o:          |
| HGT               | Heterogeneous graph transformer network                            | :o:          | :o:          |
| HEAT              | Heterogeneous edge-enhanced graph attention networks               | :o:          | :o:          |
| RGAT              | Relational graph attention networks                                | :o:          | :o:          |

### Loss

* MSE loss (supervised)

```math
\ell(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i^{bus} - \hat{y}_i^{bus} \right)^2 + \lambda \frac{1}{M} \sum_{i=1}^{M} \left( y_i^{gen} - \hat{y}_i^{gen} \right)^2
```

where $\lambda$ is a hyperparameter, $N$ is the number of buses, and $M$ is the number of generators.

* Physics-informed loss (unsupervised)
```math
\ell(\hat{x}) = \ell_{bus} (\hat{x}) + \gamma_1 \ell_{gen cost} (\hat{x}) + \gamma_2 \ell_{line} (\hat{x})
```
where $\hat{x}$ is the predicted variables, $\gamma_1$ and $\gamma_2$ are weights for generator cost and line violations.




