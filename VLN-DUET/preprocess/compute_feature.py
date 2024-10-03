#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

import MatterSim

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60

def get_all_scanvp():
    with open("block_edge_list.json", "r") as f:
        block_edge_list = json.load(f)
    scans = list(block_edge_list.keys())
    scanvp_list = []
    for scan in scans:
        paths = list(block_edge_list[scan].keys())
        for path in paths:
            for edge in block_edge_list[scan][path]:
                scanvp_list.append((scan, path, edge[0], edge[1]))
    return scanvp_list

def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=(checkpoint_file is None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)

    for scan, path, edge_0, edge_1 in scanvp_list:
        # Loop all discretized views from this location
        images = []
        inpaint_dir = os.path.join('final_inpaint_results', scan, path, edge_0, edge_1)
        inpaint_indices = []
        inpaint_name = {}
        for x in os.listdir(inpaint_dir):
            indice = int(x.split('.')[0].split('_')[0])
            inpaint_indices.append(indice)
            inpaint_name[indice] = x
        for ix in range(VIEWPOINT_SIZE):
            if ix in inpaint_indices:
                image_path = os.path.join(inpaint_dir, inpaint_name[ix])
            else:
                image_path = os.path.join('views_img', scan, edge_0, '%d.jpg'%ix)
            image = Image.open(image_path).convert('RGB')
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        fts, logits = [], []
        for k in range(0, len(images), args.batch_size):
            b_fts = model.forward_features(images[k: k+args.batch_size])
            b_logits = model.head(b_fts)
            b_fts = b_fts.data.cpu().numpy()
            b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            logits.append(b_logits)
        fts = np.concatenate(fts, 0)
        logits = np.concatenate(logits, 0)

        out_queue.put((scan, path, edge_0, edge_1, fts, logits))

    out_queue.put(None)

def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = get_all_scanvp()

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan, path, edge_0, edge_1, fts, logits = res
                key = '%s_%s_%s_%s'%(scan, path, edge_0, edge_1)
                if args.out_image_logits:
                    data = np.hstack([fts, logits])
                else:
                    data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan
                outf[key].attrs['pathId'] = path
                outf[key].attrs['edge_0'] = edge_0
                outf[key].attrs['edge_1'] = edge_1
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file', default="./inpaint_features.hdf5")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)


