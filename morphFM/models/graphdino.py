# Attention and Block adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# DINO adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/dino.py

import copy
import torch
import torch.nn as nn
from typing import Any
from functools import partial
import logging
logger = logging.getLogger("morphFM")

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
                 use_exp: bool = True,
                 test_flag: bool = False) -> nn.Module:
        super().__init__()
        self.test_flag = test_flag
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
       
        attn = attn.softmax(dim=-1)
        
        if not self.test_flag:
            attn = attn.half()

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
                 norm_layer: Any = nn.LayerNorm,
                 test_flag: bool = False) -> nn.Module:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GraphAttention(dim, num_heads=num_heads, bias=bias, use_exp=use_exp, test_flag=test_flag)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim * mlp_ratio)

    def forward(self, x, a):

        x = self.norm1(x)
        
        x = self.attn(x, a) + x
        
        x = self.norm2(x)
        
        x = self.mlp(x) + x
        
        return x
    
    
class GraphTransformer(nn.Module):
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
                 use_exp: bool = True,
                 test_flag: bool = False,
                 noise_replace_p: float = 0.10) -> nn.Module:
        super().__init__()

        self.noise_replace_p = noise_replace_p
        #print('test_flag', test_flag)
        self.test_flag = test_flag
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #print('feat_dim:', feat_dim)
        self.enc_mask_token = nn.Parameter(torch.zeros(11))

        self.norm = norm_layer(dim)

        self.embed_dim = dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp, test_flag=test_flag)
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

        self.feat_dim = feat_dim
        self.dim = dim
        
    def forward_single(self, node_feat, adj, lapl):

        B, N, _ = node_feat.shape

        '''
        for i in range(8):
            for j in range(1000):
                for k in range(11):
                    assert not torch.isnan(node_feat[i,j,k]), "flag node_feat"
        #print('x_pre:', node_feat.type())
        #print('value0:', self.to_node_embedding[0].weight)
        #print('value2:', self.to_node_embedding[2].weight)
        
        device = torch.device("cuda")
        self.to_node_embedding.to(device)
        node_feat = node_feat.to(device)
        with torch.no_grad():  # Ensure no gradient computations for this diagnostic code
            linear1 = nn.Linear(self.feat_dim, self.dim * 2).to(device)
            node_feat_temp = linear1(node_feat.float())
            assert not torch.isnan(node_feat_temp).any(), "Issue after first linear layer"
            
            relu = nn.ReLU(True).to(device)
            node_feat_temp = relu(node_feat_temp)
            assert not torch.isnan(node_feat_temp).any(), "Issue after ReLU"
            
            linear2 = nn.Linear(self.dim * 2, self.dim).to(device)
            node_feat_temp = linear2(node_feat_temp)
            assert not torch.isnan(node_feat_temp).any(), "Issue after second linear layer"
        '''

        x = self.to_node_embedding(node_feat)
        
        #print('lapl:', lapl.shape)
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
        
        #print('have_nan:', torch.any(torch.isnan(x)))
        
        #print('pos_embedding:', x)

        #print('x_in_graphdino:', x)

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
        
        #print('YES')
        
        output = []

        for node_feat, adj, lapl, mask in zip(node_feat_list, adj_list, lapl_list, mask_list):
            
            B, N, _ = node_feat.shape

            if mask is not None:

                position = torch.nonzero(mask)
                mask_num = position.shape[0]

                noise_index = torch.rand(mask_num) < self.noise_replace_p


                selected_mask = position[noise_index].t()
                noise_num = selected_mask.shape[1]

                reshaped_tensor = node_feat.view(-1, 11)
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
            i = 0
            for block in self.blocks:
                x = block(x, adj_cls)
            
            x_norm = self.norm(x)
            
            '''
            cls_token = x[:, 0]
            x_mlp = self.mlp_head(cls_token)
            x_projector = self.projector(x_mlp)
            '''
            
            output.append( {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_patchtokens": x_norm[:, 1:],
                "x_prenorm": x,
                #"x_projector": x_projector,
                }
            )
            
        return output

    def forward(self, node_feat, adj, lapl, mask=None):
        if isinstance(node_feat, list):
            return self.forward_list(node_feat, adj, lapl, mask)
        else:
            return self.forward_single(node_feat, adj, lapl)
