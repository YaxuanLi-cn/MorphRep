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
    
def get_embedding(now_dir, model, cfg):

    with os.scandir(now_dir) as entries:
        all_no = [entry.name[:] for entry in entries if entry.name[-4:]=='.swc' or entry.name[-4:]=='.SWC']

    cell_allids = []
    all_embedding = []
    all_labels = []
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

        embedding = model.student.backbone(feat, adj, lapl)["x_norm_clstoken"].detach()
        for embed in embedding[0]:
            assert not math.isnan(embed.item())

        all_embedding.append(embedding.cpu().numpy())  

        if now_swc not in filename_to_phenotype.keys():
            continue
            
        all_labels.append(phenotype_to_label[filename_to_phenotype[now_swc]])

        del feat, adj, lapl, embedding 
        torch.cuda.empty_cache()  

    return all_embedding, all_labels
    