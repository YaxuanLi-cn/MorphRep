import torch
import itertools
from torch.utils.data import Sampler
import torch.distributed as distributed

class GroupedShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        group_ranges,
        batch_size,
        shuffle=True,
        seed=0,
        start=None,
        step=None,
    ):
        self._group_ranges = group_ranges
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._generator = torch.Generator().manual_seed(seed)
        
    def __iter__(self):
        group_order = list(range(len(self._group_ranges)))
        if self._shuffle:
            group_order = torch.randperm(len(self._group_ranges), generator=self._generator).tolist()
        
        group_idx = self._start % len(self._group_ranges)
        data_idx = self._start // len(self._group_ranges)
        
        while True:
            current_group = group_order[group_idx]
            start = self._group_ranges[current_group][0] + data_idx
            end = min(start + self._batch_size - 1, self._group_ranges[current_group][1])

            if end - start + 1 < self._batch_size:
                # 当前组的剩余数据不足一个批次
                group_idx = (group_idx + 1) % len(self._group_ranges)
                data_idx = 0
                continue

            if self._shuffle:
                indices = torch.randperm(end - start + 1, generator=self._generator) + start
                yield from indices.tolist()
            else:
                yield from range(start, end + 1)

            data_idx += self._step * self._batch_size
            if data_idx > self._group_ranges[current_group][1] - self._group_ranges[current_group][0]:
                # 移动到下一个数据组
                group_idx = (group_idx + 1) % len(self._group_ranges)
                data_idx = 0

if __name__ == "__main__":
    # 测试
    group_ranges = [(0, 9), (10, 19), (20, 29)]  # 三个数据组，每个有10个样本
    batch_size = 4

    sampler = GroupedShardedInfiniteSampler(group_ranges, batch_size, shuffle=True, seed=42)
    for i, idx in enumerate(sampler):
        print(idx)
        if i > 50:  # 为了测试，我们只打印前50个样本
            break
