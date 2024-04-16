# MorphRep: Learning Meaningful Representation of Single-Neuron Morphology via Large-scale Pre-training

This repository holds the Pytorch implementation for MorphRep described in the paper 

# Preparation

CUDA Version: 11.3  
Pytorch-gpu: 2.0.1  
Python: 3.9.17  
GPU: NVIDIA 3090  

# Environment

requirement.txt


# Data Preparation

## Train Dataset
We collect neuron morphology reconstructions from a centrally curated inventory of digitally reconstructed
neurons and glia called NeuroMorpho.Org (https://neuromorpho.org/), which collect over 250000 neuron reconstructions along with their metadata including but not limited to species, brain region, cell types, and reconstruction softwares.

## Test Dataset
Following previous works in neuron morphology representation learning, we use seven commonly used datasets to benchmark the performance of MorphFM, namely ACT(Cell Type), ACT(Brain Region), BIL(Cell Type), BIL(Brain
Region), M1-EXC(Cell Type), M1-EXC(RNA family) and BBP. These datasets come from existing public available databases include M1-EXC, BBP, ACT and BIL. The labels of these datasets are either cell types or brain regions. Some datasets are with the same neurons but have different label annotation.

# Training

```
/mnt/data/aim/liyaxuan/.conda/envs/treedino/bin/torchrun --nproc_per_node=2 morphFM/train/train.py \
--config-file configs/ours_final.yaml \
--output-dir /mnt/data/aim/liyaxuan/projects/git_project2/ours_add_noise/ \
train.dataset_path=NeuronMorpho:split=TRAIN:root=/mnt/data/oss_beijing/liyaxuan/pre_data:extra=/mnt/data/oss_beijing/liyaxuan/pre_data

```

# Test
 - `pre_process.py`: Perform operations such as trimming redundant nodes and removing axons from neuron data
 - `KNN_classifier.py`: use unsupervised classification method ——KNN

'''

python pre_process.py
python KNN_classifier.py

'''