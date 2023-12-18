# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):

    #print('samples_list:', samples_list[0])

    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0]["global_crops_features"])
    n_local_crops = len(samples_list[0]["local_crops_features"])

    #for i in samples_list:
    #    print(i["global_crops"][0])
    #    exit(0)

    #print('collate.py flag0')

    cell_type = [ s["cell_type"] for s in samples_list]
    #print('cell_type:', cell_type)
    
    tensors_to_stack = [torch.tensor(s["global_crops_features"][i]) for i in range(n_global_crops) for s in samples_list] 
    if not all(t.shape == tensors_to_stack[0].shape for t in tensors_to_stack):
        print('error', [t.shape for t in tensors_to_stack])
        assert 0, [torch.tensor(s["index"]) for s in samples_list] 
    stacked_tensor = torch.stack(tensors_to_stack)

    collated_global_crops_features = torch.stack([torch.tensor(s["global_crops_features"][i]).cuda() for i in range(n_global_crops) for s in samples_list])
    collated_global_crops_adj = torch.stack([torch.tensor(s["global_crops_adj"][i]).cuda() for i in range(n_global_crops) for s in samples_list])

    #print('collate.py flag1')
    
    if n_local_crops == 0:
        collated_local_crops_features = torch.tensor([])
        collated_local_crops_adj = torch.tensor([])
    else:
        collated_local_crops_features = torch.stack([torch.tensor(s["local_crops_features"][i]).cuda() for i in range(n_local_crops) for s in samples_list])
        collated_local_crops_adj = torch.stack([torch.tensor(s["local_crops_adj"][i]).cuda() for i in range(n_local_crops) for s in samples_list])
    
    #print('collate.py flag2')

    B = len(collated_global_crops_features)
    N = n_tokens[samples_list[0]["cell_type"]]
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(N, int(N * random.uniform(prob_min, prob_max)))))
        #masks_list.append(torch.BoolTensor([]))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(N, 0)))
        #masks_list.append(torch.BoolTensor([]))

    random.shuffle(masks_list)

    
    #collated_masks = torch.stack(masks_list).flatten(1)
    collated_masks = torch.stack(masks_list)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    #print('collated_masks:', collated_masks.shape)
    #print('mask_indices_list:',mask_indices_list)

    return {
        "collated_global_crops_features": collated_global_crops_features.to(dtype).to("cpu"),
        "collated_local_crops_features": collated_local_crops_features.to(dtype).to("cpu"),
        "collated_global_crops_adj": collated_global_crops_adj.to(dtype).to("cpu"),
        "collated_local_crops_adj": collated_local_crops_adj.to(dtype).to("cpu"),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
