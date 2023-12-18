from pathlib import Path
import pickle
import pandas as pd
import networkx as nx
from allensdk.core import swc
import numpy as np
from tqdm import *
import os
import time
from pre_process import get_embedding
import csv
from enum import Enum
import logging
from typing import Callable, List, Optional, Tuple, Union
import pickle
import numpy as np
from pathlib import Path
from torch.multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from morphFM.data.datasets.utils import neighbors_to_adjacency, subsample_graph, rotate_graph, jitter_node_pos, translate_soma_pos, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, remap_neighbors, neighbors_to_adjacency_torch
from morphFM.data.datasets.data_utils import connect_graph, remove_axon, rotate_cell
import copy
import json
import seaborn as sns
from sklearn.manifold import TSNE
from morphFM.data.datasets.neuron_morpho import NeuronMorpho
from morphFM.train.utils_graph import plot_neuron, plot_tsne, neighbors_to_adjacency_torch, compute_eig_lapl_torch_batch
from morphFM.models import build_model_from_cfg
from morphFM.utils.config import setup
from morphFM.train.train import get_args_parser,build_optimizer,build_schedulers
from morphFM.train.ssl_meta_arch import SSLMetaArch
import os
from morphFM.fsdp import FSDPCheckpointer
from morphFM.models.graphdino import GraphTransformer
import torch.optim as optim
from torch.utils.data import random_split
import datetime
import math
from sklearn.neighbors import KNeighborsClassifier

all_dataset = ['allen_cell_type_processed', 'allen_region_processed', 'BBP_cell_type_processed', 'BIL_cell_type_processed', 'M1_EXC_cell_type_processed', 'M1_EXC_region_processed']
#all_dataset = ['M1_EXC_region_processed']
root_dir = '/mnt/data/aim/liyaxuan/projects/git_project2/benchmark_dataset/'
#checkpoint_path = '/mnt/data/aim/liyaxuan/projects/git_project2/ours_checkpoint.pth'
checkpoint_path = '/mnt/data/aim/liyaxuan/projects/git_project2/60423_student_checkpoint.pth'

def KNN(all_labels, all_embedding):
    
    acc_mean = 0.0

    all_labels = np.array(all_labels)
    all_data = np.squeeze(all_embedding, axis=1)

    num_all = len(all_labels)
    num_train = int( 0.6 * len(all_labels) )

    sum_val = 0.0
    num_repeat = 5

    for k in range(num_repeat):
        
        indices = np.arange(len(all_data))
        np.random.shuffle(indices)

        train_indices = indices[:int(0.6 * len(all_data))] 
        val_indices = indices[int(0.4 * len(all_data)):] 
        
        train_data = all_data[train_indices]
        train_labels = all_labels[train_indices]

        val_data = all_data[val_indices]
        val_labels = all_labels[val_indices]

        max_val = 0.0

        for now_n in range(1,20):

            k_nn = KNeighborsClassifier(n_neighbors=now_n)

            k_nn.fit(train_data, train_labels)

            predicted_labels = k_nn.predict(train_data)
            acc_num = np.sum((train_labels == predicted_labels) + 0)
            total_num = len(train_labels)

            training_acc = round(acc_num * 100.0 / total_num, 2)

            predicted_labels = k_nn.predict(val_data)
            acc_num = np.sum((val_labels == predicted_labels) + 0)
            total_num = len(val_labels)

            val_acc = round(acc_num * 100.0 / total_num, 2)

            if val_acc > max_val:
                max_val = val_acc

        sum_val += max_val
    
    sum_val = round(sum_val * 1.0 / num_repeat, 2)
    acc_mean += sum_val
    print(sum_val)


def build_model(cfg):

    backbonemodel = SSLMetaArch(cfg).to(torch.device("cuda"))
    backbonemodel.prepare_for_distributed_training()

    student_state_dict = torch.load(checkpoint_path)

    backbonemodel.student.load_state_dict(student_state_dict["student"])

    return backbonemodel

def run(now_dataset, cfg):

    embedding, labels = get_embedding(root_dir + now_dataset + '/', model, cfg)
    KNN(labels, embedding)


args = get_args_parser(add_help=True).parse_args()
args.config_file = '/mnt/data/aim/liyaxuan/projects/git_project2/configs/ours_final.yaml'
args.output_dir = '/mnt/data/aim/liyaxuan/projects/git_project2/neuron_org_embedding'
cfg = setup(args)
model = build_model(cfg)

for dataset in all_dataset:
    print('now test :' + dataset)
    run(dataset, cfg)







