from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from pathlib import Path
from torch.multiprocessing import Manager

from morphFM.data.datasets.utils import subsample_graph, rotate_graph, jitter_node_pos, translate_soma_pos, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, remap_neighbors, neighbors_to_adjacency_torch
from morphFM.data.datasets.utils import AverageMeter, compute_eig_lapl_torch_batch

import copy
import torch
import torch.nn as nn
from typing import Any
import math
from functools import partial

class Adapter(nn.Module):
    def __init__(self,
                 dropout=0.0,
                 n_embd = None):
        
        super().__init__()

        self.n_embd = 768 if n_embd is None else n_embd

        self.down_size = 64

        self.scale = nn.Parameter(torch.ones(1))

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        return up
    
class GraphAttention(nn.Module):
    """ Implements GraphAttention.

    Graph Attention interpolates global transformer attention
    (all nodes attend to all other nodes based on their
    dot product similarity) and message passing (nodes attend
    to their 1-order neighbour based on dot-product
    attention).

    Attributes:
        dim: Dimensionality of key, query and value vectors.
        num_heads: Number of parallel attention heads.
        bias: If set to `True`, use bias in input projection layers.
          Default is `False`.
        use_exp: If set to `True`, use the exponential of the predicted
          weights to trade-off global and local attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 bias: bool = False,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.use_exp = use_exp

        self.qkv_projection = nn.Linear(dim, dim * num_heads * 3, bias=bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        
        # Weigth to trade of local vs. global attention.
        self.predict_gamma = nn.Linear(dim, 2)
        # Initialize projection such that gamma is close to 1
        # in the beginning of training.
        self.predict_gamma.weight.data.uniform_(0.0, 0.01)

        
    @torch.jit.script
    def fused_mul_add(a, b, c, d):
        return (a * b) + (c * d)

    def forward(self, x, adj):
        B, N, C = x.shape # (batch x num_nodes x feat_dim)
        qkv = self.qkv_projection(x).view(B, N, 3, self.num_heads, self.dim).permute(0, 3, 1, 2, 4)
        query, key, value = qkv.unbind(dim=3) # (batch x num_heads x num_nodes x dim)

        attn = (query @ key.transpose(-2, -1)) * self.scale # (batch x num_heads x num_nodes x num_nodes)

        # Predict trade-off weight per node
        gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)
        if self.use_exp:
            # Parameterize gamma to always be positive
            gamma = torch.exp(gamma)

        adj = adj[:, None].repeat(1, self.num_heads, 1, 1)

        # Compute trade-off between local and global attention.
        attn = self.fused_mul_add(gamma[:, :, :, 0:1], attn, gamma[:, :, :, 1:2], adj)
        
        attn = attn.softmax(dim=-1).half()

        x = (attn @ value).transpose(1, 2).reshape(B, N, -1) # (batch_size x num_nodes x (num_heads * dim))
        return self.proj(x)
    
    
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> nn.Module:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """ Implements an attention block.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 bias: bool = False,
                 use_exp: bool = True,
                 norm_layer: Any = nn.LayerNorm) -> nn.Module:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GraphAttention(dim, num_heads=num_heads, bias=bias, use_exp=use_exp)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim * mlp_ratio)
        self.adapter = Adapter(n_embd=dim)

    def forward(self, x, a):

        lst_x = x
        x = self.norm1(x)
        x = self.attn(x, a) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        x = x + self.adapter(lst_x)
        return x
    
class GraphTransformer_withadapter(nn.Module):
    def __init__(self,
                 n_nodes: int = 200,
                 dim: int = 32,
                 depth: int = 5,
                 num_heads: int = 8,
                 mlp_ratio: int = 2,
                 feat_dim: int = 8,
                 num_classes: int = 1000,
                 pos_dim: int = 32,
                 proj_dim: int = 128,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #print('feat_dim:', feat_dim)
        self.enc_mask_token = nn.Parameter(torch.zeros(8))

        self.norm = norm_layer(dim)

        self.embed_dim = dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp)
            for i in range(depth)])

        self.to_pos_embedding = nn.Linear(pos_dim, dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, num_classes)
        )

        self.to_node_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(True),
            nn.Linear(dim * 2, dim)
        )

        
    def forward_single(self, node_feat, adj, lapl):
        B, N, _ = node_feat.shape

        #print('x_pre:', node_feat.shape)

        # Compute initial node embedding.
        x = self.to_node_embedding(node_feat)

        #print('x_lst:', x.shape)

        # Compute positional encoding
        pos_embedding_token = self.to_pos_embedding(lapl)

        # Add "classification" token
        cls_pos_enc = self.cls_pos_embedding.repeat(B, 1, 1)
        pos_embedding = torch.cat((cls_pos_enc, pos_embedding_token), dim=1)

        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add classification token entry to adjanceny matrix. 
        adj_cls = torch.zeros(B, N + 1, N + 1, device=node_feat.device)
        # TODO(test if useful)
        adj_cls[:, 0, 0] = 1.
        adj_cls[:, 1:, 1:] = adj

        x += pos_embedding
        for block in self.blocks:
            x = block(x, adj_cls)
        
        x_norm = self.norm(x)
        '''
        cls_token = x[:, 0]
        x_mlp = self.mlp_head(cls_token)
        x_projector = self.projector(x_mlp)
        '''
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            #"x_projector": x_projector,
        }

        '''
        tmp = x
        x = x[:, 0]
        x = self.mlp_head(x)
        '''

        #return tmp, self.projector(x)
    
    def forward_list(self, node_feat_list, adj_list, lapl_list, mask_list):

        output = []

        for node_feat, adj, lapl, mask in zip(node_feat_list, adj_list, lapl_list, mask_list):
                
            B, N, _ = node_feat.shape

            if mask is not None:

                position = torch.nonzero(mask)
                mask_num = position.shape[0]

                noise_index = torch.rand(mask_num) < 0.10
                selected_mask = position[noise_index].t()
                noise_num = selected_mask.shape[1]

                reshaped_tensor = node_feat.view(-1, 8)
                noise_feat_indices = torch.randperm(reshaped_tensor.shape[0])[:noise_num]
                selected_noise_feat = reshaped_tensor[noise_feat_indices]

                node_feat = torch.where(mask.unsqueeze(-1), self.enc_mask_token, node_feat)
                node_feat[selected_mask[0], selected_mask[1]] = selected_noise_feat


            # Compute initial node embedding.
            x = self.to_node_embedding(node_feat)

            # Compute positional encoding
            pos_embedding_token = self.to_pos_embedding(lapl)

            # Add "classification" token
            cls_pos_enc = self.cls_pos_embedding.repeat(B, 1, 1)
            pos_embedding = torch.cat((cls_pos_enc, pos_embedding_token), dim=1)

            cls_tokens = self.cls_token.repeat(B, 1, 1)
            x = torch.cat((cls_tokens, x), dim=1)

            # Add classification token entry to adjanceny matrix. 
            adj_cls = torch.zeros(B, N + 1, N + 1, device=node_feat.device)
            # TODO(test if useful)
            adj_cls[:, 0, 0] = 1.
            adj_cls[:, 1:, 1:] = adj

            x += pos_embedding
            for block in self.blocks:
                x = block(x, adj_cls)
            
            x_norm = self.norm(x)
            '''
            cls_token = x[:, 0]
            x_mlp = self.mlp_head(cls_token)
            x_projector = self.projector(x_mlp)
            '''
            return {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_patchtokens": x_norm[:, 1:],
                "x_prenorm": x,
                #"x_projector": x_projector,
            }
        
        return output

    def forward(self, node_feat, adj, lapl, mask=None):
        if isinstance(node_feat, list):
            return self.forward_list(node_feat, adj, lapl, mask)
        else:
            return self.forward_single(node_feat, adj, lapl)