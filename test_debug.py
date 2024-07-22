# %% [markdown]
# <a href="https://colab.research.google.com/github/Open-Catalyst-Project/ocp/blob/tutorials_01_11/tutorials/OCP_Tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import os
os.chdir("/l/users/elizaveta.starykh/OCP_project/ocp-git/")

# %%
# !pwd

# %%
##### all imports collected


# %load_ext autoreload
# %autoreload 2

import os
# os.chdir("/l/users/elizaveta.starykh/OCP_project/ocp-git/")

import sys
print(sys.version)
import torch
print(torch.__version__)

import torch
import matplotlib
# matplotlib.use('Agg')
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

### April 2024, python9_kernel, python9 env, python=3.9.18, CSCC
import e3nn
from ocpmodels.trainers import OCPTrainer
from ocpmodels.datasets import LmdbDataset
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging, setup_imports
setup_imports()
setup_logging()
import copy

from torch.utils.data import DataLoader, SubsetRandomSampler
import torch_geometric.loader  

import yaml

from tqdm.auto import tqdm
import pickle as pkl

import wandb
import random


# %%
import wandb
import random

# wandb.login(key="1328f64c6e0b320e2120c10c420b1008f0e5be5d")
wandb.login()

# %%
from ocpmodels.common import logger

# %%
sweep_configuration = {
    # "logger" : "wandb",
    "method": "random",
    "name": "sweep-PaiNN-20K",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        # "batch_size": {"values": [16, 32, 64]},  ### given
        "batch_size": {"values": [2, 4, 8]},
        # "epochs": {"values": [5, 10, 15]},   ### given
        "epochs": {"value": 1},
        "lr" : {"value": 1}
        # "lr": {
        #     "distribution": "log_uniform_values",
        #     "max": 0.1, 
        #     "min": 0.001 },
    },
}

# %%
# wandb_logger = logger.WandBLogger(sweep_configuration)

# %%
sweep_configuration = {
    "method": "random",
    "name": "sweep-PaiNN-20K",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        # "batch_size": {"values": [16, 32, 64]},  ### given
        "batch_size": {"values": [2, 4, 8]},
        # "epochs": {"values": [5, 10, 15]},   ### given
        "epochs": {"value": 1},
        "lr" : {"value": 1}
        # "lr": {
        #     "distribution": "log_uniform_values",
        #     "max": 0.1, 
        #     "min": 0.001 },
    },
}

# %%
from pprint import pprint

pprint(sweep_configuration)

# %%
sweep_configuration["parameters"]['batch_size']['values']

# %%
sweep_id = wandb.sweep(sweep=sweep_configuration, project="PaiNN-DOS-efermi")

# %%


# %%
# LmdbDataset is our custom Dataset method to read the lmdbs as Data objects. Note that we need to give the path to the folder containing lmdbs for S2EF
# dataset = LmdbDataset({"src": "./data/s2ef/200k/train/"})
# dataset = LmdbDataset({"src": "./data/s2ef/200k/train/output_lmdb/normalized_efermi/train_20000_systems/"})
# dataset = LmdbDataset({"src": "./data/s2ef/200k/train/output_lmdb/normalized_efermi/"})
dataset = LmdbDataset({"src": "./data/s2ef/200k/train/output_lmdb/"})

# dataset = LmdbDataset({"src": "/home/elizaveta.starykh/OCP_project/ocp-git/data/s2ef/200k/train/output_lmdb/train_20000_systems"})


print("Size of the dataset created:", len(dataset))
print(dataset[0])

# %%
dataset[125106]["bulk_total_dos"]

# %%
dataset[0]['y']

# %%
# import math
# import pandas as pd
# for ii in range(len(dataset)):
#     system_tmp = pd.Series(dataset[ii])

#     if pd.isnull(system_tmp):
#     # for jj, key in enumerate(system_tmp.keys()):
#         # if math.isnan(system_tmp[key]):
#         print(jj, key) 


# %%
dataset[0].y

# %%
energies = torch.tensor([data.y for data in dataset])
energies

# %% [markdown]
# ### mean, std

# %%
# train_dataset = LmdbDataset({"src": "./data/s2ef/200k/train/output_lmdb"})

energies = []
for data in dataset:
    energies.append(data.y)

mean = np.mean(energies)
stdev = np.std(energies)

stdev, mean 
## == (2.8471219290033876, -0.7877437496095779) for /normalized_efermi/ 176k dataset
## == (2.8412495666979143, -0.78793442576679) for ??
## == (2.8392264933285123, -0.7946160068500009) for 20k dataset

# %%
# train_dataset = LmdbDataset({"src": "./data/s2ef/200k/train/output_lmdb"})

efermi_info = []
# for data in train_dataset:
for data in dataset:
    efermi_info.append(data.efermi)


# %%
def sigmoid(z):
  return 1.0 / (1 + np.exp(-z)) 

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


efermi_info = np.array(efermi_info)
efermi_tanh = tanh(efermi_info)


########
# efermi_normalized = (efermi_info - efermi_info.min()) / (efermi_info.max() - efermi_info.min()) 

# efermi_normalized.min(), efermi_normalized.max(), efermi_normalized.std()
# efermi_sigmoid = sigmoid(efermi_info)

# mean = np.mean(efermi_info)
# stdev = np.std(efermi_info)

# stdev, mean 
## == (2.8471219290033876, -0.7877437496095779) for /output_lmdb/ full dataset
## == (2.8412495666979143, -0.78793442576679)

# %%
efermi_tanh.min(), efermi_tanh.max(), efermi_tanh.std()
len(efermi_tanh)

# %%
plt.plot(efermi_info, efermi_tanh, 'bo')
# plt.yscale("log")
plt.xlabel("efermi")

plt.ylabel("efermi tanh")
plt.show()

# %% [markdown]
# ### Training

# %%
# train_src = "data/s2ef/train_100"

train_src = "./data/s2ef/200k/train/output_lmdb"
# train_src = "/home/elizaveta.starykh/OCP_project/ocp-git/data/s2ef/200k/train/output_lmdb/train_20000_systems"
# train_src = "./data/s2ef/200k/train/output_lmdb/normalized_efermi/"
val_src = "./data/s2ef/200k/train/output_lmdb/normalized_efermi/train_20000_systems/"

# %%
# Model
# /configs/oc22/s2ef/painn

with open('configs/oc22/s2ef/painn/painn.yml', 'r') as file:
    model = yaml.safe_load(file)

model = model["model"]
model['name'] = 'ocpmodels.models.painn.painn.PaiNN'
model['hidden_channels'] = 512

model['efermi_length'] = 128
# model['checkpoint_every'] = 1000

model

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
# stdev, mean = (2.8471219290033876, -0.7877437496095779)
## == (2.8471219290033876, -0.7877437496095779) for /normalized_efermi/ 176k dataset
## == (2.8392264933285123, -0.7946160068500009) for 20k dataset
dataset = [
{'src': train_src,
'normalize_labels': True,
#  "target_mean": mean,
"target_mean": -0.7877437496095779,
#  "target_std": stdev,
"target_std": 2.8471219290033876,
"grad_target_mean": 0.0,
#  "grad_target_std": stdev
"grad_target_std": 2.8471219290033876
}, # train set
{'src': val_src}, # val set (optional)
]


# %%
# def task_optim_dataset(batch_size_sweep):

#   # Task
#   task = {
#       'dataset': 'lmdb', # dataset used for the S2EF task
#       'description': 'Regressing to energies and forces for DFT trajectories from OCP',
#       'type': 'regression',
#       'metric': 'mae',
#       'labels': ['potential energy'],
#       'grad_input': 'atomic forces',
#       'train_on_free_atoms': True,
#       'eval_on_free_atoms': True
#   }

#   # Optimizer
#   optimizer = {
#       'batch_size': batch_size_sweep,         # originally 32
#       'eval_batch_size': batch_size_sweep,    # originally 32
#       'num_workers': 2,
#       'lr_initial': 5.e-4,
#       'optimizer': 'AdamW',
#       'optimizer_params': {"amsgrad": True},
#       'scheduler': "ReduceLROnPlateau",
#       'mode': "min",
#       'factor': 0.8,
#       'patience': 3,
#       'max_epochs': 1,         # used for demonstration purposes
#       'force_coefficient': 100,
#       'ema_decay': 0.999,
#       'clip_grad_norm': 10,
#       'loss_energy': 'mae',
#       'loss_force': 'l2mae',
#       'eval_every': 500
#   }
#   # Dataset
#   # stdev, mean = (2.8471219290033876, -0.7877437496095779)

#   dataset = [
#     {'src': train_src,
#     'normalize_labels': True,
#     #  "target_mean": mean,
#     "target_mean": -0.7946160068500009,
#     #  "target_std": stdev,
#     "target_std": 2.8392264933285123,
#     "grad_target_mean": 0.0,
#     #  "grad_target_std": stdev
#     "grad_target_std": 2.8392264933285123
#     }, # train set
#     # {'src': val_src}, # val set (optional)
#   ]

#   # (2.8392264933285123, -0.7946160068500009)

#   return task, optimizer, dataset

# %%
# wandb.config["batch_size"]

# %%
def train_wandb(config=None):
    with wandb.init(project="PaiNN-DOS-efermi", config = sweep_configuration):
        config = wandb.config

        for epoch in range(config.epochs):

            print(wandb.config["batch_size"])
            batch_size = config['batch_size']

            task, optimizer, dataset = task_optim_dataset(batch_size)


            trainer = OCPTrainer(
                task=task,
                model=copy.deepcopy(model), # copied for later use, not necessary in practice.
                dataset=dataset,
                optimizer=optimizer,
                outputs={},
                loss_fns={},
                eval_metrics={},
                name="s2ef",
                identifier="S2EF-example",
                run_dir=".", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
                is_debug=False, # if True, do not save checkpoint, logs, or results
                print_every=5,
                seed=0, # random seed to use
                logger="tensorboard", # logger of choice (tensorboard and wandb supported)
                local_rank=0,
                amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage),
            )



            trainer.train()
            avg_loss = trainer.final_loss
            wandb.log({"loss": avg_loss, "epoch": epoch}) 

# %%
# !pwd

# %%
torch.manual_seed(0)

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
