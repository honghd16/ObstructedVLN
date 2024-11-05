import os
import json
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nums', type=int, default=1)
args = parser.parse_args()

anno_dir = "../VLN-DUET/datasets/R2R/annotations"
connectivity_dir = "../VLN-DUET/datasets/R2R/connectivity"
splits = ["train", 'val_seen', 'val_unseen']

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
            + (pose1['pose'][7]-pose2['pose'][7])**2\
            + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

block_list = defaultdict(lambda: defaultdict(list))

paths = []
path_num = 0
path_block_num = 0
max_dist = 0
original_lengths = []
modified_lengths = []
for split in splits:
    with open(os.path.join(anno_dir, f"R2R_{split}_enc.json"), "r") as f:
        data = json.load(f)

    scans = set([x['scan'] for x in data])
    print('Loading navigation graphs for %d scans' % len(scans))
    graphs = load_nav_graphs(connectivity_dir, scans)

    for x in tqdm(data):
        scan = x['scan']
        path = x['path']
        path_id = x['path_id']
        
        if path_id in paths:
            print("Duplicate path_id: {}".format(path_id))              
            exit(0)
        paths.append(path_id)
        path_len = len(path)
        original_lengths.append(path_len)
        G = graphs[scan].copy()

        # draw graph G
        backup_G = G.copy()
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        combs = list(combinations(edges, args.nums))
        for comb in combs:
            flag = True
            for edge in comb:
                G.remove_edge(edge[0], edge[1])
                if not nx.has_path(G, edge[0], edge[1]):
                    flag = False

            if flag:
                cand_vps = set()
                new_path = []
                original_path = []
                detailed_path = path.copy()

                for vp in path:
                    if vp in new_path:
                        continue
                    
                    original_path.append(vp)
                    if len(new_path) == 0 or vp in cand_vps:
                        new_path.append(vp)
                        cand_vps.add(vp)
                        cand_vps.update(G.neighbors(vp))
                        continue
                    
                    bypass = nx.shortest_path(G, new_path[-1], vp)[1:]
                    real_bypass = []
                    for v in bypass[::-1]:
                        if v not in new_path:
                            real_bypass.append(v)
                        else:
                            break
                    
                    for v in real_bypass[::-1]:
                        if v == path[-1]:
                            break
                        cand_vps.add(v)
                        cand_vps.update(G.neighbors(v))
                    
                adjustment = 0
                for edge in comb:
                    bypass = nx.shortest_path(G, edge[0], edge[1])
                    edge_idx = path.index(edge[0]) + adjustment
                    detailed_path = detailed_path[:edge_idx] + bypass + detailed_path[edge_idx+2:]
                    adjustment += len(bypass) - 2
                    
                if len(new_path) <= (10 + args.nums*5):
                    block_list[scan][path_id].append((comb, new_path, original_path, detailed_path))
                    modified_lengths.append(len(detailed_path))

            G = backup_G.copy()
        
        if len(block_list[scan][path_id]) > 0:
            path_block_num += 1
        
    path_num += len(data)

    print("Finish split: {}".format(split))

print("path_num: {}, block_path_num: {}".format(path_num, path_block_num))    
print("max_dist: {}".format(max_dist))
print("original_lengths mean: {}, modified_lengths mean: {}".format(np.mean(original_lengths), np.mean(modified_lengths)))
print("original_lengths min: {}, modified_lengths min: {}".format(np.min(original_lengths), np.min(modified_lengths)))  
print("original_lengths max: {}, modified_lengths max: {}".format(np.max(original_lengths), np.max(modified_lengths)))
print("original_lengths len: {}, modified_lengths len: {}".format(len(original_lengths), len(modified_lengths)))
print("====================================================================")

with open(f'block_{args.nums}_edge_list.json', 'w') as f:
    json.dump(block_list, f, indent=4, sort_keys=True)
