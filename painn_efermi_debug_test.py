# %% [markdown]
# <a href="https://colab.research.google.com/github/Open-Catalyst-Project/ocp/blob/tutorials_01_11/tutorials/OCP_Tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import os
os.chdir("/l/users/elizaveta.starykh/OCP_project/ocp-git/")
import sys
import torch
import torch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import ase.io
from ase.io.trajectory import Trajectory
from ase.io import extxyz
from ase.calculators.emt import EMT
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.visualize.plot import plot_atoms
from ase import Atoms
from IPython.display import Image
import ocpmodels
import lmdb
import torch_geometric
from ocpmodels.datasets import LmdbDataset
### May 2024, python9_kernel, python9 env, python=3.9.18, CSCC
import e3nn
from ocpmodels.trainers import OCPTrainer
from ocpmodels.datasets import LmdbDataset
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging, setup_imports

while True:
    try:
        setup_imports()
        break
    except (ModuleNotFoundError, RuntimeError, TypeError, NameError):
        print('setup_imports() error raised, continue...')
        pass

setup_logging()
import copy
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch_geometric.loader  
import yaml
from tqdm.auto import tqdm
import pickle as pkl
import wandb
import random

# %% [markdown]
# ### Training

# %%
with open("./data/dataset_config.yaml", 'r') as file:
    dataset_info = yaml.safe_load(file)

# %%
train_dataset_config = dataset_info["datasets"][1]
val_dataset_config = dataset_info["datasets"][2]

train_src = train_dataset_config["path"]
val_src = val_dataset_config["path"]

# %%
# Model
# /configs/oc22/s2ef/painn

with open('configs/oc22/s2ef/painn/painn.yml', 'r') as file:
    model = yaml.safe_load(file)

model = model["model"]
model['name'] = 'ocpmodels.models.painn.painn.PaiNN'
model['hidden_channels'] = 512
model['efermi_length'] = 128
if train_dataset_config["efermi_available"]:    
    model["multiply_efermi"] = False
    model["concatenate_efermi"] = True
# model

# %%
# Task
task = {
    'dataset': 'lmdb', # dataset used for the S2EF task
    'description': 'Regressing to energies and forces for DFT trajectories from OCP',
    'type': 'regression',
    'metric': 'mae',
    'labels': ['potential energy'],
    'grad_input': 'atomic forces',
    'train_on_free_atoms': True,
    'eval_on_free_atoms': True
}

# Optimizer
optimizer = {
    'batch_size': 32,         # originally 32
    'eval_batch_size': 32,    # originally 32
    'num_workers': 2,
    'lr_initial': 5.e-4,
    'optimizer': 'AdamW',
    'optimizer_params': {"amsgrad": True},
    'scheduler': "ReduceLROnPlateau",
    'mode': "min",
    'factor': 0.8,
    'patience': 3,
    'max_epochs': 1,         # used for demonstration purposes
    'force_coefficient': 100,
    'ema_decay': 0.999,
    'clip_grad_norm': 10,
    'loss_energy': 'mae',
    'loss_force': 'l2mae',
    # 'eval_every': 500
}
# Dataset
dataset = [
{'src': train_src,
'normalize_labels': True,
#  "target_mean": mean,
"target_mean": train_dataset_config["mean"],
#  "target_std": stdev,
"target_std": train_dataset_config["stdev"],
"grad_target_mean": 0.0,
#  "grad_target_std": stdev
"grad_target_std": train_dataset_config["stdev"]
}, # train set
{'src': val_src}, # val set (optional)
]

# %%
# torch.manual_seed(0)

trainer = OCPTrainer(
    task=task,
    model=copy.deepcopy(model), # copied for later use, not necessary in practice.
    dataset=dataset,
    optimizer=optimizer,
    outputs={},
    loss_fns={},
    eval_metrics={},
    name="s2ef",
    identifier="S2EF-20k-train",
    run_dir=".", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
    is_debug=False, # if True, do not save checkpoint, logs, or results
    print_every=5,
    seed=0, # random seed to use
    logger="tensorboard", # logger of choice (tensorboard and wandb supported)
    local_rank=0,
    amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage),
)

# %%
trainer.train()

# %%
sys.exit("finishing trainer.train() function, exiting...")
