# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader

from .utils import neighbors_to_adjacency, subsample_graph, rotate_graph, jitter_node_pos, translate_soma_pos, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, remap_neighbors, neighbors_to_adjacency_torch
from .data_utils import connect_graph, remove_axon, rotate_cell
import json
import copy

logger = logging.getLogger("morphFM")
_Target = int

import numpy as np
import networkx as nx

def select_random_component(adj_matrix, neighbors, features):
    """
    Randomly select a connected component from the graph and return its features, adj_matrix, and neighbors.
    
    Args:
        adj_matrix: adjacency matrix of graph (N x N)
        neighbors: dict of neighbors per node
        features: features per node (N x D)
    
    Returns:
        component_adj_matrix: adjacency matrix of the selected connected component
        component_neighbors: neighbors dict of the selected connected component
        component_features: features of the nodes in the selected connected component
    """
    G = nx.Graph(adj_matrix)
    components = list(nx.connected_components(G))
    
    # Randomly select one connected component
    selected_component = list(np.random.choice(components))
    
    # Extract corresponding information from adj_matrix, neighbors, and features
    component_adj_matrix = adj_matrix[np.ix_(selected_component, selected_component)]
    component_neighbors = {k: [n for n in v if n in selected_component] for k, v in neighbors.items() if k in selected_component}
    component_features = features[selected_component]
    
    neighbors_new, _ = remap_neighbors(component_neighbors)
    
    return component_adj_matrix, neighbors_new, component_features

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 3,
            _Split.VAL: 2,
            _Split.TEST: 2,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index

class NeuronMorpho(Dataset):

    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "NeuronMorpho.Split",
        root: str,
        extra: str,
        keep_node: int,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mode: Optional[str] = 'all',
        config_path: Optional[str] = '/mnt/data/aim/liyaxuan/projects/project2/configs/config.json',
        inference: Optional[bool] = False,
        data_path: Optional[str] = None
    ) -> None:

        config = json.load(open(config_path))

        self._split = split

        self.config = config
        self.mode = mode
        self.inference = inference

        if data_path == None:
            data_path = config['data']['path']

        # Augmentation parameters.
        self.jitter_var = config['data']['jitter_var']
        self.rotation_axis = config['data']['rotation_axis']
        self.n_drop_branch = config['data']['n_drop_branch']
        self.translate_var = config['data']['translate_var']
        self.n_nodes = keep_node

        # Load cell ids.
        cell_ids = list(np.load(Path(data_path, f'{mode}_ids.npy')))

        print('neu:', data_path, mode, len(cell_ids))
        #print('mode:', mode)
        #print('cell_ids:', cell_ids)
        
        # Load graphs.
        self.manager = Manager()
        self.cells = self.manager.dict()
        #if mode == 'all':
        #    cell_ids = [cell_ids[953 + i] for i in range(4)]
        count = 0
        for cell_id in tqdm(cell_ids):
            # Adapt for datasets where this is not true.
            soma_id = 0

            #if not os.path.exists(Path(data_path, 'skeletons', str(cell_id), 'features.npy')):
            #    continue

            features = np.load(Path(data_path, 'skeletons', str(cell_id), 'features.npy'))
            #print('features:', features.shape)
            with open(Path(data_path, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                neighbors = pickle.load(f)
            '''
            if cell_id == 34965:
                print(neighbors)
                exit(0)
            '''
            assert len(features) == len(neighbors)

            if len(features) >= self.n_nodes or self.inference:
                
                
                # Subsample graphs for faster processing during training.
                neighbors, not_deleted = subsample_graph(neighbors=neighbors, 
                                                         not_deleted=set(range(len(neighbors))), 
                                                         keep_nodes=keep_node, 
                                                         protected=[soma_id])
                # Remap neighbor indices to 0..999.
                neighbors, subsampled2new = remap_neighbors(neighbors)
                soma_id = subsampled2new[soma_id]

                # Accumulate features of subsampled nodes.
                features = features[list(subsampled2new.keys()), :]

                '''
                if cell_id == 34965:
                    print(neighbors)
                '''

                leaf_branch_nodes = get_leaf_branch_nodes(neighbors)
                
                # Using the distances we can infer the direction of an edge.
                distances = compute_node_distances(soma_id, neighbors)

                item = {
                    'cell_id': cell_id,
                    'features': features, 
                    'neighbors': neighbors,
                    'distances': distances,
                    'soma_id': soma_id,
                    'leaf_branch_nodes': leaf_branch_nodes,
                }

                not_connect = 0
                for key in neighbors:
                    for to in neighbors[key]:
                        if to not in distances.keys():
                            not_connect = 1
                            break
                    if not_connect == 1:
                        break
                
                if not_connect == 1:
                    print('not_connect')
                    continue
                print('now_cellid:', cell_id)
                self.cells[count] = item
                count += 1

        #print('cells:',self.cells[748])
        #print('cells:',self.cells[749])
        self.num_samples = len(self.cells)

    def split(self) -> "NeuronMorpho.Split":
        return self._split

    def __len__(self):
        return self.num_samples

    def _delete_subbranch(self, neighbors, soma_id, distances, leaf_branch_nodes):

        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors))) 
        for i in range(self.n_drop_branch):
            neighbors, drop_nodes = drop_random_branch(leaf_branch_nodes, neighbors, distances, keep_nodes=self.n_nodes)
            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes
            
            if len(leaf_branch_nodes) == 0:
                break

        return not_deleted

    def _reduce_nodes(self, neighbors, soma_id, distances, leaf_branch_nodes):
        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        # Delete random branches.
        not_deleted = self._delete_subbranch(neighbors2, soma_id, distances, leaf_branch_nodes)

        # Subsample graphs to fixed number of nodes.
        neighbors2, not_deleted = subsample_graph(neighbors=neighbors2, not_deleted=not_deleted, keep_nodes=self.n_nodes, protected=soma_id)

        # Compute new adjacency matrix.
        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)
        
        assert adj_matrix.shape == (self.n_nodes, self.n_nodes), '{} {}'.format(adj_matrix.shape)
        
        return neighbors2, adj_matrix, not_deleted
    
    
    def _augment_node_position(self, features):
        # Extract positional features (xyz-position).
        pos = features[:, :3]

        # Rotate (random 3D rotation or rotation around specific axis).
        rot_pos = rotate_graph(pos, axis=self.rotation_axis)

        # Randomly jitter node position.
        jittered_pos = jitter_node_pos(rot_pos, scale=self.jitter_var)
        
        # Translate neuron position as a whole.
        jittered_pos = translate_soma_pos(jittered_pos, scale=self.translate_var)
        
        features[:, :3] = jittered_pos

        return features
    
    def local_crop(self, cell):

        features = copy.deepcopy(cell['features'])
        neighbors = copy.deepcopy(cell['neighbors'])
        distances = copy.deepcopy(cell['distances'])
        soma_id = copy.deepcopy(cell['soma_id'])

        keep_nodes = 300

        #print('feature_shape:', features.shape)
        neighbors, features, soma_id = remove_axon(neighbors, features, int(soma_id))
        adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))
        adj_matrix, neighbors, features = select_random_component(adj_matrix, neighbors, features)

        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        local_branch_nodes = {i for i in cell['leaf_branch_nodes'] if i in neighbors2.keys()}

        not_deleted = self._delete_subbranch(neighbors2, None, distances, local_branch_nodes)

        neighbors2, not_deleted = subsample_graph(neighbors=neighbors2, not_deleted = not_deleted, keep_nodes=keep_nodes, protected=[])

        # Compute new adjacency matrix.
        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)
        new_features = features[not_deleted].copy()

        return new_features, adj_matrix

    def _augment(self, cell):

        features = cell['features']
        neighbors = cell['neighbors']
        distances = cell['distances']
        
        #print('feature0:', features.shape)

        # Reduce nodes to N == n_nodes via subgraph deletion and subsampling.
        neighbors2, adj_matrix, not_deleted = self._reduce_nodes(neighbors, [int(cell['soma_id'])], distances, cell['leaf_branch_nodes'])

        # Extract features of remaining nodes.
        new_features = features[not_deleted].copy()
       
        # Augment node position via rotation and jittering.
        new_features = self._augment_node_position(new_features)

        return new_features, adj_matrix
    
    def __getsingleitem__(self, index): 
        cell = self.cells[index]
        return cell['features'], cell['neighbors']
    
    '''
    def geometric_augmentation_global(self):

    def global_transfo1
    '''
    
    def __getitem__(self, index): 

        #print('index:', index)
        cell = self.cells[index]
        n_local_crops = 5

        # Compute two different views through augmentations.
        features1_global, adj_matrix1_global = self._augment(cell)
        features2_global, adj_matrix2_global = self._augment(cell)

        output = {}
        output["global_crops_features"] = [features1_global, features2_global]
        output["global_crops_adj"] = [adj_matrix1_global, adj_matrix2_global]

        output["global_crops_teacher_features"] = [features1_global, features2_global]
        output["global_crops_teacher_adj"] = [adj_matrix1_global, adj_matrix2_global]
        
        output["local_crops_features"] = []
        output["local_crops_adj"] = []

        for i in range(n_local_crops):

            feature_local, adj_matrix_local = self.local_crop(cell)

            features_local = self._augment_node_position(copy.deepcopy(feature_local))

            adj_matrix_local = copy.deepcopy(adj_matrix_local)

            output["local_crops_features"].append(self._augment_node_position(copy.deepcopy(feature_local)))
            output["local_crops_adj"].append(copy.deepcopy(adj_matrix_local))
        
        output["offsets"] = ()

        return output
        
