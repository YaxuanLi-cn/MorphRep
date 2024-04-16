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
import datetime

_Target = int

nowdate = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
logger_bug = logging.getLogger('morphFM_bug')
logger_bug.setLevel(logging.DEBUG)
file_log = logging.FileHandler('/mnt/data/aim/liyaxuan/git_project2/idx_in_training/' + nowdate,'a',encoding='utf-8')
file_log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s ')
file_log.setFormatter(formatter)
logger_bug.handlers.clear()
logger_bug.addHandler(file_log)
logger_bug.propagate = False

import numpy as np
import networkx as nx

def fill_graph(neighbors=None, not_deleted=None, features=None, keep_nodes=200):

    #print('neighbors_before:', neighbors)
    k_nodes = len(neighbors)
    perm = torch.randperm(k_nodes).tolist()
    all_indices = np.array(list(not_deleted))[perm].tolist()
    now_choice = 3

    if k_nodes < keep_nodes:

        not_deleted = list(not_deleted)
        mx = np.max(not_deleted)

        for i in range(k_nodes, keep_nodes):
            
            now_id = mx + i - k_nodes + 1

            chose_a = -1
            chose_b = -1

            while True:
                
                if len(all_indices) == 0:
                    
                    now_choice += 1
                        
                    remaining = list(not_deleted)
                    perm = torch.randperm(len(remaining)).tolist()
                    all_indices = np.array(remaining)[perm].tolist()

                idx = all_indices.pop()

                for nei in neighbors[idx]:
                    if np.array_equal(features[nei,now_choice:],features[idx,now_choice:]):
                        chose_a = nei
                        chose_b = idx
                        break
                
                if chose_a != -1:
                    break
            
            #print('neighbors:', neighbors)

            neighbors[chose_a].remove(chose_b)
            neighbors[chose_b].remove(chose_a)
            
            neighbors[chose_a].add(now_id)
            neighbors[chose_b].add(now_id)
            
            neighbors[now_id] = set([])
            neighbors[now_id].add(chose_a)
            neighbors[now_id].add(chose_b)

            not_deleted.append(now_id)
            all_indices.append(now_id)

            features = np.vstack((features, np.hstack(((features[idx,:now_choice] + features[nei,:now_choice])/2, features[idx,now_choice:]))))
            
        return neighbors, set(not_deleted), features

    else:
        return neighbors, not_deleted, features

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
    
    component_debug = []
    len_max = 0
    max_component = 0
    for component_now in components:
        if len(component_now) > 150:
            component_debug.append(component_now)
        if len(component_now) > len_max:
            max_component = component_now
            len_max = len(component_now)

    # Randomly select one connected component
    if len(component_debug) > 0:
        selected_component = list(np.random.choice(component_debug))
    else:
        selected_component = list(max_component)

    #print('selected:', selected_component)
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
        cfg,
        split: "NeuronMorpho.Split",
        root: str,
        extra: str,
        keep_node: int,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mode: Optional[str] = 'all',
        inference: Optional[bool] = False,
        data_path: Optional[str] = None,
        n_local_crop: Optional[int] = 4
    ) -> None:

        self._split = split
        self.n_local_crop = n_local_crop
        self.save_cnt = 0
        self.mode = mode
        self.inference = inference

        if data_path == None:
            data_path = cfg.dataset.path
        
        # Augmentation parameters.
        self.jitter_var = cfg.dataset.jitter_var
        self.rotation_axis = cfg.dataset.rotation_axis
        self.n_drop_branch = cfg.dataset.n_drop_branch
        self.translate_var = cfg.dataset.translate_var

        self.n_nodes = cfg.crops.global_crops_size

        self.local_crop_nodes = cfg.crops.local_crops_size

        self.cells = []
        self.count = [0 for i in range(3)]
        self.data_path = data_path

        cell_cnt = 0

        if False:
            
            cell_ids = list(np.load(Path(data_path, f'{mode}_ids.npy')))
            self.manager = []
            
            for i in range(3):
                self.manager.append(Manager())
                self.cells.append(self.manager[i].dict())

            bug_cell = [136342, 110312, 136547, 110325]

            #if mode == 'all':
            #    cell_ids = [cell_ids[i] for i in range(100)]

            for cell_id in tqdm(cell_ids):

                if cell_id in bug_cell:
                    continue

                soma_id = 0

                features = np.load(Path(data_path, 'skeletons', str(cell_id), 'features.npy'))
                
                with open(Path(data_path, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                    neighbors = pickle.load(f)

                assert len(features) == len(neighbors)

                keep_node = -1
                cell_type = 2

                for i in range(3):
                    if len(features) >= self.n_nodes[i]:
                        keep_node = self.n_nodes[i]
                        cell_type = i

                if self.inference:
                    cell_type = 2
                    keep_node = self.n_nodes[2]

                if keep_node == -1:
                    print('node:', len(features))
                    continue

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
                    'type': cell_type,
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

                #print('now_cellid:', cell_id)
                
                self.cells[cell_type][self.count[cell_type]] = item
                
                self.count[cell_type] += 1

                cell_cnt += 1

                if cell_cnt % 20000 == 0:
                    self.save_processed_data()

            self.save_processed_data()
            with open(Path(data_path, 'processed_data/save_data_num.pkl'), 'wb') as f:
                pickle.dump((self.count, self.save_cnt), f)
        
        self.open_saved_data()

        self.num_samples = self.count[0] + self.count[1] + self.count[2]
        print('num_samples:', self.num_samples, self.count[0], self.count[1], self.count[2])
    
    def save_processed_data(self):

        normal_cells = [dict(cell) for cell in self.cells] 

        while(True):

            with open(Path(self.data_path, 'processed_data/processed_data' + str(self.save_cnt) + '.pkl'), 'wb') as f:
                pickle.dump(normal_cells, f)

            try:
                with open(Path(self.data_path, 'processed_data/processed_data' + str(self.save_cnt) + '.pkl'), 'rb') as file:
                    data1 = pickle.load(file)
                break
            except Exception as e:
                print('Error loading pickle file:', e)
                os.remove(Path(self.data_path, 'processed_data/processed_data' + str(self.save_cnt) + '.pkl'))
                continue

        self.save_cnt += 1

        for i in range(3):
            self.cells[i].clear()

    def open_saved_data(self):
       
        self.manager = []
        self.cells = []
        for i in range(3):
            self.manager.append(Manager())
            self.cells.append(self.manager[i].dict())

        with open(Path(self.data_path, 'processed_data/save_data_num.pkl'), 'rb') as f:
            self.count, self.save_cnt = pickle.load(f)

        for i in range(self.save_cnt):
            print('i:', i)
            with open(Path(self.data_path, 'processed_data/processed_data' + str(i) + '.pkl'), 'rb') as f:
                normal_cells = pickle.load(f)
            for j in range(3):
                self.cells[j].update(normal_cells[j])
        

    def split(self) -> "NeuronMorpho.Split":
        return self._split

    def get_count(self):
        return self.count

    def __len__(self):
        return self.num_samples

    def _delete_subbranch(self, neighbors, soma_id, distances, leaf_branch_nodes, cell_type):

        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors))) 
        for i in range(self.n_drop_branch):
            neighbors, drop_nodes = drop_random_branch(leaf_branch_nodes, neighbors, distances, keep_nodes=self.n_nodes[cell_type])
            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes
            
            if len(leaf_branch_nodes) == 0:
                break

        return not_deleted

    def _reduce_nodes(self, neighbors, soma_id, distances, leaf_branch_nodes, cell_type):
        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        # Delete random branches.
        not_deleted = self._delete_subbranch(neighbors2, soma_id, distances, leaf_branch_nodes, cell_type)

        # Subsample graphs to fixed number of nodes.
        neighbors2, not_deleted = subsample_graph(neighbors=neighbors2, not_deleted=not_deleted, keep_nodes=self.n_nodes[cell_type], protected=soma_id)

        # Compute new adjacency matrix.
        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)
        
        assert adj_matrix.shape == (self.n_nodes[cell_type], self.n_nodes[cell_type]), '{} {}'.format(adj_matrix.shape)
        
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

        keep_nodes = self.local_crop_nodes[cell['type']]

        #print('feature_shape:', features.shape)
        neighbors_test, features_test, soma_id_test = remove_axon(copy.deepcopy(neighbors), copy.deepcopy(features), int(soma_id))
        
        if len(neighbors_test) >= keep_nodes:
            neighbors, features, soma_id = neighbors_test, features_test, soma_id_test
            #print('YES')

        adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))

        #print('neighbors:', neighbors)

        adj_matrix, neighbors, features = select_random_component(adj_matrix, neighbors, features)

        #print('neighbors:', neighbors)
        #print('features:', features)

        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        local_branch_nodes = {i for i in cell['leaf_branch_nodes'] if i in neighbors2.keys()}

        not_deleted = self._delete_subbranch(neighbors2, None, distances, local_branch_nodes, cell['type'])

        new_features = features[list(not_deleted)].copy()

        neighbors2, not_deleted, new_features = fill_graph(neighbors=neighbors2, not_deleted=not_deleted, features=new_features, keep_nodes=keep_nodes)
        neighbors2, not_deleted = subsample_graph(neighbors=neighbors2, not_deleted = not_deleted, keep_nodes=keep_nodes, protected=[])

        # Compute new adjacency matrix.
        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)

        new_features = new_features[not_deleted].copy()

        return new_features, adj_matrix
    
    def _augment(self, cell):

        features = cell['features']
        neighbors = cell['neighbors']
        distances = cell['distances']
        
        #print('feature0:', features.shape)

        # Reduce nodes to N == n_nodes via subgraph deletion and subsampling.
        neighbors2, adj_matrix, not_deleted = self._reduce_nodes(neighbors, [int(cell['soma_id'])], distances, cell['leaf_branch_nodes'], cell['type'])

        # Extract features of remaining nodes.
        new_features = features[not_deleted].copy()
       
        # Augment node position via rotation and jittering.
        new_features = self._augment_node_position(new_features)

        return new_features, adj_matrix
    
    def get_cell_type(self, index):
        
        if index < self.count[2]:
            return 2, index
        elif index < self.count[1] + self.count[2]:
            return 1, index - self.count[2]
        else:
            return 0, index - self.count[2] - self.count[1]

    def __getsingleitem__(self, index): 

        cell_type, index = self.get_cell_type(index)
        cell = self.cells[cell_type][index]
        return cell['features'], cell['neighbors']
   
    def __getitem__(self, index): 
        
        #print('index:', index)

        output = {}

        output["index"] = index

        cell_type, index = self.get_cell_type(index)
        
        
        #print('index:', index)
        #print('cell_type:', cell_type)

        cell = self.cells[cell_type][index]

        logger_bug.info('self.cells_id: {}'.format(cell["cell_id"]))

        n_local_crops = self.n_local_crop

        # Compute two different views through augmentations.
        features1_global, adj_matrix1_global = self._augment(cell)
        features2_global, adj_matrix2_global = self._augment(cell)

        
        output["global_crops_features"] = [features1_global, features2_global]
        output["global_crops_adj"] = [adj_matrix1_global, adj_matrix2_global]

        output["global_crops_teacher_features"] = [features1_global, features2_global]
        output["global_crops_teacher_adj"] = [adj_matrix1_global, adj_matrix2_global]
        
        output["local_crops_features"] = []
        output["local_crops_adj"] = []

        output["cell_type"] = cell_type

        '''
        output["local_crops_features"] = [features1_global for i in range(5)]
        output["local_crops_adj"] = [adj_matrix1_global for i in range(5)]
        output["offsets"] = ()
        
        return output
        '''

        for i in range(n_local_crops):

            feature_local, adj_matrix_local = self.local_crop(cell)

            features_local = self._augment_node_position(copy.deepcopy(feature_local))

            adj_matrix_local = copy.deepcopy(adj_matrix_local)

            output["local_crops_features"].append(self._augment_node_position(copy.deepcopy(feature_local)))
            output["local_crops_adj"].append(copy.deepcopy(adj_matrix_local))
        
        output["offsets"] = ()
        #import math
        #if math.isnan(output["local_crops_features"]):
        #    logger.info('index: {}'.format(output))
        return output
        
