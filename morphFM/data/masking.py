# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        num_masking_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        self.min_num_patches = 4

    def _mask(self, num_patches, mask, max_mask_patches):

        delta = 0
        
        for _ in range(10):

            target_area = round(random.uniform(self.min_num_patches, max_mask_patches))
            
            num_start = random.randint(0, num_patches - target_area)
            
            num_masked = mask[num_start : num_start + target_area].sum()
            
            if 0 < target_area - num_masked <= max_mask_patches:
                for i in range(num_start, num_start + target_area):
                    if mask[i] == 0:
                        mask[i] = 1
                        delta += 1

            if delta > 0:
                break

        return delta

    def __call__(self, num_patches, num_masking_patches=0):

        mask = np.zeros(shape=num_patches, dtype=bool)
        is_noise = np.zeros(shape=num_patches, dtype=bool)
        
        max_num_patches = 0.5 * num_patches

        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, max_num_patches)

            delta = self._mask(num_patches, mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        #print('masking produce:', mask.shape, mask_count)
        return mask
