# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from .graphdino import GraphTransformer
from .graphdino_withadapter import GraphTransformer_withadapter
import json
import copy
import os
import torch

logger = logging.getLogger("morphFM")

'''
def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
'''

def build_model(cfg, only_teacher=False, resume_path=None, adapter=False, test_flag=False):
    
    num_classes = cfg.model.num_classes

    if adapter:
        transformer = GraphTransformer_withadapter(n_nodes=cfg.crops.global_crops_size,
                    dim=cfg.model.dim, 
                    depth=cfg.model.depth, 
                    num_heads=cfg.model.n_head,
                    feat_dim=cfg.model.feat_dim,
                    pos_dim=cfg.model.pos_dim,
                    num_classes=num_classes,
                    test_flag=test_flag)
    else:
        transformer = GraphTransformer(n_nodes=cfg.crops.global_crops_size,
                    dim=cfg.model.dim, 
                    depth=cfg.model.depth, 
                    num_heads=cfg.model.n_head,
                    feat_dim=cfg.model.feat_dim,
                    pos_dim=cfg.model.pos_dim,
                    num_classes=num_classes,
                    test_flag=test_flag,
                    noise_replace_p=cfg.ibot.noise_replace_p)
    
    teacher = transformer

    if only_teacher:
        if resume_path is not None:
            rank = int(os.environ['RANK'])
            state_dict = torch.load(resume_path + '.rank_{}.pth'.format(str(rank)))
            print('state_dict:', state_dict)
            teacher.load_state_dict(state_dict)
        return teacher, teacher.embed_dim
    
    student = copy.deepcopy(transformer)
    
    embed_dim = student.embed_dim

    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False, resume_path=None, adapter=False, test_flag=False):
    return build_model(cfg, only_teacher=only_teacher, resume_path=resume_path, adapter=adapter, test_flag=test_flag)
