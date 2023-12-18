# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
import pickle
from fvcore.common.checkpoint import PeriodicCheckpointer
import torch
import json 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from pathlib import Path
from torch.multiprocessing import Manager
from tqdm import tqdm
from morphFM.data.datasets.neuron_morpho import NeuronMorpho
from morphFM.train.utils_graph import plot_neuron, plot_tsne, neighbors_to_adjacency_torch, compute_eig_lapl_torch_batch
from morphFM.data import SamplerType, make_data_loader, make_dataset
from morphFM.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import morphFM.distributed as distributed
from morphFM.fsdp import FSDPCheckpointer
from morphFM.logging import MetricLogger
from morphFM.utils.config import setup
from morphFM.utils.utils import CosineScheduler
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset, DataLoader
from morphFM.train.ssl_meta_arch import SSLMetaArch
import numpy as np
import shutil
from torch import nn
import copy
import datetime
from morphFM.data.datasets.utils import neighbors_to_adjacency, subsample_graph, rotate_graph, jitter_node_pos, translate_soma_pos, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, remap_neighbors, neighbors_to_adjacency_torch
import torch.optim as optim
from torch.utils.data import random_split

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("morphFM")

def process(dset, model, cfg):

    latents = np.zeros((dset.num_samples, cfg.model.dim))
    
    for i in tqdm(range(dset.num_samples)):
        feat, neigh = dset.__getsingleitem__(i)
        adj = neighbors_to_adjacency_torch(neigh, list(neigh.keys())).half().cuda()[None, ]
        lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=cfg.model.pos_dim).float().cuda()
        feat = torch.from_numpy(feat).half().cuda()[None, ]

        latents[i] = model.student.backbone(feat, adj, lapl)["x_norm_clstoken"].cpu().detach()
        #print('latent', latents[i])

    return latents

def fill_graph(neighbors=None, not_deleted=None, features=None, keep_nodes=200):

    k_nodes = len(neighbors)
    perm = torch.randperm(k_nodes).tolist()
    all_indices = np.array(list(not_deleted))[perm].tolist()

    if k_nodes < keep_nodes:

        not_deleted = list(not_deleted)
        mx = np.max(not_deleted)

        for i in range(k_nodes, keep_nodes):
            
            now_id = mx + 1 + i - k_nodes

            chose_a = -1
            chose_b = -1

            while True:
                
                if len(all_indices) == 0:
                    assert len(not_deleted) < keep_nodes
                    remaining = list(not_deleted)
                    perm = torch.randperm(len(remaining)).tolist()
                    all_indices = np.array(remaining)[perm].tolist()

                idx = all_indices.pop()

                for nei in neighbors[idx]:
                    if np.array_equal(features[nei,3:],features[idx,3:]):
                        chose_a = nei
                        chose_b = idx
                        break
                
                if chose_a != -1:
                    break
            
            neighbors[chose_a].remove(chose_b)
            neighbors[chose_b].remove(chose_a)
            
            neighbors[chose_a].add(now_id)
            neighbors[chose_b].add(now_id)
            
            neighbors[now_id] = set([])
            neighbors[now_id].add(chose_a)
            neighbors[now_id].add(chose_b)

            not_deleted.append(now_id)

            features = np.vstack((features, np.hstack(((features[idx,:3] + features[nei,:3])/2, features[idx,3:]))))
            
        return neighbors, set(not_deleted), features

    else:
        return neighbors, not_deleted, features

class BBP_dataset(Dataset):

    def __init__(
        self,
        data_path,
        keep_node,
        mode,
    ) -> None:
        
        cell_ids = list(np.load(Path(data_path, mode + '_ids.npy')))
        
        self.manager = Manager()
        self.cells = self.manager.dict()

        count = 0
        for cell_id in tqdm(cell_ids):
            
            soma_id = 0

            features = np.load(Path(data_path, 'skeletons', str(cell_id), 'features.npy'))
            
            with open(Path(data_path, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                neighbors = pickle.load(f)
            
            neighbors, not_deleted, features = fill_graph(neighbors=neighbors, not_deleted=set(range(len(neighbors))), features=features, keep_nodes=keep_node)
            neighbors, not_deleted = subsample_graph(neighbors=neighbors, not_deleted=not_deleted, keep_nodes=keep_node,  protected=[soma_id], filling=True)

            neighbors, subsampled2new = remap_neighbors(neighbors)
            soma_id = subsampled2new[soma_id]

            features = features[list(subsampled2new.keys()), :]
            item = {'features': features,  'neighbors': neighbors }
            
            self.cells[count] = item
            count += 1
        
        self.num_samples = count
    
    def __len__(self):
        return len(self.cells)

    def __getsingleitem__(self, index): 
        cell = self.cells[index]
        feat = cell['features']
        neigh = cell['neighbors']

        return feat, neigh

class data_dataset(Dataset):

    def __init__(
        self,
        data,
        mode,
        data_path,
    ) -> None:
        
        self.data = list(data)
        self.label = list(np.load(Path(data_path, mode + '_labels.npy')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        return self.data[index], self.label[index]

class Classifier(nn.Module):
    def __init__(self, num_classes, dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = self.classifier(x.float())
        return x

def do_test(args, cfg, model, logger_1, iteration):

    root_dir = '/mnt/data/aim/liyaxuan/projects/project2/'
    
    data_path = root_dir + 'condition_processed/10.1016_j.neuroscience.2018.01.031/'

    num_classes = 4
    test_keep_node = 500
    
    train_dset = BBP_dataset(data_path=data_path, keep_node=test_keep_node, mode='train')
    val_dset = BBP_dataset(data_path=data_path, keep_node=test_keep_node, mode='val')

    train_data = process(train_dset, model, cfg)

    #print('train_data:', train_data)

    val_data = process(val_dset, model, cfg)

    train_dataset = data_dataset(train_data, data_path=data_path, mode='train')
    val_dataset = data_dataset(val_data, data_path=data_path, mode='val')

    model_classifier = Classifier(num_classes=num_classes, dim=128).cuda()
    
    optimizer = optim.Adam(model_classifier.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_epochs = 30
    max_acc = 0.0

    for epoch in range(num_epochs):

        running_loss = 0.0
        ACC = 0
        sum_num = 0

        for embedding, label in train_loader:
            
            embedding = embedding.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            outputs = model_classifier(embedding)

            max_positions = torch.argmax(outputs, dim=1)
            corrects = (max_positions == label)

            answer = corrects.sum().item()
            ACC += answer

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            sum_num += len(label)

        training_acc = ACC * 100.0 / sum_num

        if distributed.is_main_process():
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)} Acc: { ACC * 100.0 / sum_num}%")
            logger_1.info('Iteration {} | Epoch {} | Train: Loss {:.4f} | Acc {:.2f}%'.format(iteration, epoch+1, running_loss/len(train_loader), ACC * 100.0 / sum_num))
        
        running_loss = 0.0
        ACC = 0
        sum_num = 0

        for embedding, label in val_loader:

            embedding = embedding.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            outputs = model_classifier(embedding)

            max_positions = torch.argmax(outputs, dim=1)
            corrects = (max_positions == label)
            answer = corrects.sum().item()
            ACC += answer

            loss = criterion(outputs, label)

            running_loss += loss.item()
            sum_num += len(label)

        val_acc = ACC * 100.0 / sum_num

        if training_acc - val_acc > -10.0:
            if val_acc > max_acc:
                max_acc = val_acc

        if distributed.is_main_process():
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(val_loader)} Acc: { ACC * 100.0 / sum_num}%")
            logger_1.info('Iteration {} | Epoch {} | Val: Loss {:.4f} | Acc {:.2f}%'.format(iteration, epoch+1, running_loss/len(val_loader), ACC * 100.0 / sum_num))

    print('Finished Training')
    return max_acc

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def save_checkpoint(cfg, model, loss, iteration):

    teacher_state_dict = model.teacher.state_dict()
    student_state_dict = model.student.state_dict()

    if distributed.is_main_process():
        iterstring = f"{loss:.6f}"  # format loss to 6 decimal places
        eval_base_dir = os.path.join(cfg.train.output_dir, "eval")
        eval_dir = os.path.join(eval_base_dir, iterstring)
        
        # Before saving the checkpoint, check and delete the one with highest loss if necessary
        if os.path.exists(eval_base_dir):
            subdirs = [d for d in os.listdir(eval_base_dir) if os.path.isdir(os.path.join(eval_base_dir, d))]
            if len(subdirs) > 5:
                highest_loss_dir = min(subdirs, key=lambda x: float(x))  # get the directory with highest loss
                shutil.rmtree(os.path.join(eval_base_dir, highest_loss_dir))  # delete the directory

        os.makedirs(eval_dir, exist_ok=True)

        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, str(iteration) + "_teacher_checkpoint.pth")
        student_ckp_path = os.path.join(eval_dir, str(iteration) + "_student_checkpoint.pth")
        
        torch.save({"teacher": teacher_state_dict}, teacher_ckp_path)
        torch.save({"student": student_state_dict}, student_ckp_path)
        

def do_knn(knn_dataset, cfg, root_dir, model, iteration, logger_1):

    dir_name = '/mnt/data/aim/liyaxuan/projects/project2/benchmark_datasets/'
    
    with os.scandir(dir_name) as entries:
        all_dataset_names = [entry.name for entry in entries if entry.is_dir()]
    #all_dataset_names = ['M1_EXC_cell_type_processed']
    sum_acc = 0.0

    for i, dataset_name in enumerate(all_dataset_names):

        max_acc = 0.0
        dataset_path = dir_name + dataset_name + '/'
        train_data = process(knn_dataset[i * 2], model, cfg)
        val_data = process(knn_dataset[i * 2 + 1], model, cfg)

        for now_n in range(1,21):

            #print('now_n:', now_n)

            train_labels = np.load(dataset_path + 'train_labels.npy')
            val_labels = np.load(dataset_path + 'val_labels.npy')

            k_nn = KNeighborsClassifier(n_neighbors=now_n)
            k_nn.fit(train_data, train_labels)

            predicted_labels = k_nn.predict(train_data)
            acc_num = np.sum((train_labels == predicted_labels) + 0)
            total_num = len(train_labels)

            training_acc = round(acc_num * 1.0 / total_num, 4)
            #print('Training Acc: ', training_acc)

            predicted_labels = k_nn.predict(val_data)
            acc_num = np.sum((val_labels == predicted_labels) + 0)
            total_num = len(val_labels)
            test_acc = round(acc_num * 1.0 / total_num, 4)
            #print('Test Acc:', test_acc)

            if training_acc - test_acc > -2:
                if test_acc > max_acc:
                    max_acc = test_acc

        if distributed.is_main_process():
        #print(dataset_name, ':', max_acc)
            logger_1.info('Iteration {} | Dataset: {} |Train: Acc {:.2f}%'.format(iteration, dataset_name, 100 * max_acc))
        
        sum_acc += max_acc

    if distributed.is_main_process():
        logger_1.info(' ')
    
    return sum_acc * 1.0 / len(all_dataset_names)



def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    
    
    nowdate = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logger_1 = logging.getLogger('class_logger')
    logger_1.setLevel(logging.DEBUG)
    file_log = logging.FileHandler('/mnt/data/aim/liyaxuan/projects/project2/test_in_training/' + nowdate,'a',encoding='utf-8')
    file_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s ')
    file_log.setFormatter(formatter)
    logger_1.handlers.clear()
    logger_1.addHandler(file_log)
    logger_1.propagate = False
    
    dir_name = '/mnt/data/aim/liyaxuan/projects/project2/benchmark_datasets/'

    with os.scandir(dir_name) as entries:
        all_dataset_names = [entry.name for entry in entries if entry.is_dir()]
    #all_dataset_names = ['M1_EXC_cell_type_processed']

    knn_dataset = {}
    
    for i, dataset_name in enumerate(all_dataset_names):
        dataset_path = dir_name + dataset_name + '/'
        knn_dataset[i * 2] = NeuronMorpho(cfg=cfg, root='', extra='', keep_node=cfg.dataset.node_num, split=NeuronMorpho.Split['TRAIN'], mode='train', inference=True, data_path = dataset_path)
        knn_dataset[i * 2 + 1] = NeuronMorpho(cfg=cfg, root='', extra='', keep_node=cfg.dataset.node_num, split=NeuronMorpho.Split['TRAIN'],mode='val', inference=True, data_path = dataset_path)
    
    node_num = cfg.crops.global_crops_size

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    
    #print('resume:', resume)
    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    #print('model_weights:', cfg.MODEL.WEIGHTS)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    print('node_num:', node_num)

    mask_generator = MaskingGenerator()

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
        n_tokens=node_num,
    )

    # setup data loader

    dataset = make_dataset(
        cfg=cfg,
        dataset_str=cfg.train.dataset_path,
        keep_node=node_num,
        n_local_crop=cfg.crops.local_crops_number,
        mode='all',
    )
    
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        #num_workers=cfg.train.num_workers,
        num_workers=0,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    acc_list = []
    knn_dir = '/mnt/data/aim/liyaxuan/projects/project2/benchmark_datasets/'

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops_features"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            print('loss:', loss_dict_reduced)
            print('iter:', iteration)
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing
        now_loss = losses_reduced

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            
            acc_mean = do_knn(knn_dataset, cfg, knn_dir, model, iteration, logger_1)
            torch.cuda.synchronize()

            save_checkpoint(cfg, model, acc_mean, iteration)

            torch.distributed.barrier()
            torch.cuda.synchronize()
            #print('flag4')
            max_acc = do_test(args, cfg, model, logger_1, iteration)
            torch.cuda.synchronize()

            logger.info('VAL | Iteration {} | KNN Acc Mean: {:.2f}% | Max Classification Acc {:.2f}%'.format(iteration, acc_mean * 100, max_acc))

        periodic_checkpointer.step(iteration)
        iteration = iteration + 1

    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
