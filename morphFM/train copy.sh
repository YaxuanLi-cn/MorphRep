/mnt/data/aim/liyaxuan/.conda/envs/treedino/bin/torchrun --nproc_per_node=2 morphFM/train/train.py \
--config-file morphFM/configs/train/have_data_separate.yaml \
--output-dir /mnt/data/aim/liyaxuan/projects/project2/test/ \
train.dataset_path=NeuronMorpho:split=TRAIN:root=/mnt/data/aim/liyaxuan/projects/project2/sample_predata:extra=/mnt/data/aim/liyaxuan/projects/project2/sample_predata

