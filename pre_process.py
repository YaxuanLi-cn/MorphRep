from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
from allensdk.core import swc
import numpy as np
from morphFM.data.datasets.data_utils import connect_graph, rotate_cell
from morphFM.data.datasets.utils import compute_eig_lapl_torch_batch, neighbors_to_adjacency_torch, neighbors_to_adjacency, plot_neuron, compute_node_distances, adjacency_to_neighbors, remap_neighbors, subsample_graph
import os
import numpy as np
import pandas as pd
from tqdm import *
import numpy as np
import networkx as nx
import logging
import os
import time
import torch
import math
import json
from pathlib import Path
import pickle
import pandas as pd
import networkx as nx
from allensdk.core import swc
import numpy as np
from tqdm import *
import os
import time
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

def remove_axon(neighbors, features, adj_matrix, soma_id):
    
    axon_mask = (features[:, 5] == 1)

    axon_idcs = list(np.where(axon_mask)[0])

    non_key_nodes = [k for k, v in neighbors.items() if len(v) in [1, 2] and k in axon_idcs]
    
    if soma_id in non_key_nodes:
        non_key_nodes.remove(soma_id)

    G = nx.Graph(adj_matrix)
    
    for node in non_key_nodes:
        neighs = list(G.neighbors(node))

        if node in neighs:
            neighs.remove(node)

        G.remove_node(node)
        axon_idcs.remove(node)
        
        if len(neighs) == 2:
            if not nx.has_path(G, neighs[0], neighs[1]):
                G.add_edge(neighs[0], neighs[1])

    adj_matrix = nx.to_numpy_array(G)
    
    neighbors = adjacency_to_neighbors(adj_matrix)
    
    mapping = {i: j for j, i in enumerate(sorted(set(range(features.shape[0])) - set(non_key_nodes)))}
    
    features = np.delete(features, non_key_nodes, axis=0)
    
    soma_id = mapping[soma_id]
    
    not_deleted = list(mapping.values())
    adj_matrix = neighbors_to_adjacency(neighbors, not_deleted)

    return neighbors, features, soma_id
    
def run(now_dir, cfg):

    if not os.path.exists(now_dir + 'processed'):
        os.makedirs(now_dir + 'processed')

    with os.scandir(now_dir) as entries:
        all_no = [entry.name[:] for entry in entries if entry.name[-4:]=='.swc' or entry.name[-4:]=='.SWC']

    cell_allids = []
    all_data = {}

    keep_node = cfg.crops.global_crops_size

    filename_to_phenotype = json.load(open(now_dir + 'label/filename_to_phenotype.json'))
    phenotype_to_label = json.load(open(now_dir + 'label/phenotype_to_label.json'))

    for i in tqdm(all_no[:]):
       
        now_swc = i
        now_id = i
        
        swc_path = now_dir  + str(i) 
        
        try:
            from io import StringIO

            with open(swc_path, 'r') as f:
                lines = [line for line in f if not line.lstrip().startswith('#') and line.strip() != '']

            virtual_file = StringIO('\n'.join(lines))

            morphology = pd.read_csv(
                virtual_file,
                delim_whitespace=True,
                skipinitialspace=True,
                names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                index_col=False
            )

            have_err_read = 0
            
            for i in range(morphology.shape[0]):
                item = morphology.iloc[i]
                for name in ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']:
                    if isinstance(item[name], str):
                        have_err_read = 1
                        break
                if have_err_read == 1:
                    break
            
            if have_err_read == 1:
                continue

        except:
            continue
        
        soma = morphology.iloc[0]
        
        norm_id = int(soma['id'])
        soma_pos = np.array([soma['x'], soma['y'], soma['z']])
        soma_id = int(soma['id'] - norm_id)

        # # Process graph.
        neighbors = {}
        idx2node = {}
        # hav_err = 0
        
        for i in range(morphology.shape[0]):
            # Get node features.
            
            item = morphology.iloc[i]
            sec_type = [0, 0, 0, 0, 0, 0, 0]

            if item['type'] > 7:
                item['type'] = 7
            sec_type[int(item['type']) - 1] = 1
            feat = tuple([item['x'], item['y'], item['z'], item['radius']]) + tuple(sec_type)
            
            now_item_id = int(int(item['id']) - norm_id)

            idx2node[now_item_id] = feat
            
            # Get neighbors.
            neighbors[now_item_id] = set(morphology[morphology['parent']==item['id']]['id'])
            neighbors[now_item_id] = set([int(i-1) for i in neighbors[now_item_id]])
            if item['parent'] != -1:
                neighbors[now_item_id].add(item['parent'] - norm_id)
        
        neighbors, _ = remap_neighbors(neighbors)
        
        features = np.array(list(idx2node.values()))
        
        if np.any(np.isnan(features)):
            continue
            
        # Normalize soma position to origin.
        norm_features = features.copy()
        norm_features[:, :3] = norm_features[:, :3] - soma_pos

        if len(neighbors) <= 20:
            continue

        adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))

        G = nx.Graph(adj_matrix)
        
        if nx.number_connected_components(G) > 1:
            continue
        
        assert len(neighbors) == len(adj_matrix)
        
        # Remove axons.
        neighbors, norm_features, soma_id = remove_axon(neighbors, norm_features, adj_matrix, int(soma_id))
        
        if len(neighbors) <= 20:
            continue

        adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))

        distances = compute_node_distances(soma_id, neighbors)

        keys = [ key for key in neighbors]
        
        assert len(neighbors) == len(norm_features)
        assert ~np.any(np.isnan(norm_features))

        neighbors, not_deleted = subsample_graph(neighbors=neighbors, not_deleted=set(range(len(neighbors))), keep_nodes=keep_node,  protected=[soma_id], filling=True)
        neighbors, subsampled2new = remap_neighbors(neighbors)
        soma_id = subsampled2new[soma_id]
        
        feat = features[list(subsampled2new.keys()), :]

        adj = neighbors_to_adjacency_torch(neighbors, list(neighbors.keys())).half().cuda()[None, ]
        lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=cfg.model.pos_dim).float().cuda()
        feat = torch.from_numpy(feat).half().cuda()[None, ]

        if now_swc not in filename_to_phenotype.keys():
            continue

        now_data = {}
        now_data['adj'] = adj.cpu().numpy().tolist()
        now_data['lapl'] = lapl.cpu().numpy().tolist()
        now_data['feat'] = feat.cpu().numpy().tolist()
        now_data['label'] = phenotype_to_label[filename_to_phenotype[now_swc]]

        all_data[str(i)] = now_data

    with open(now_dir + 'processed/processed_data.json', 'w') as f:
        json.dump(all_data, f)


all_dataset = ['allen_cell_type_processed', 'allen_region_processed', 'BBP_cell_type_processed', 'BIL_cell_type_processed', 'M1_EXC_cell_type_processed', 'M1_EXC_region_processed']
root_dir = '/mnt/data/aim/liyaxuan/git_project2/benchmark_datasets/'

args = get_args_parser(add_help=True).parse_args()
args.config_file = '/mnt/data/aim/liyaxuan/git_project2/configs/ours_final.yaml'
args.output_dir = '/mnt/data/aim/liyaxuan/git_project2/neuron_org_embedding'
cfg = setup(args)

for dataset in all_dataset:
    print('Processing:' + dataset)
    run(root_dir + dataset + '/', cfg)
    print(dataset + 'processed')








    