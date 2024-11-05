import os
import json
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations

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
        for edge in edges:
            G.remove_edge(edge[0], edge[1])
            if nx.has_path(G, edge[0], edge[1]):
                block_list[scan][path_id].append(edge)
            G = backup_G.copy()

    print("Finish split: {}".format(split))

with open(f'block_edge_list.json', 'w') as f:
    json.dump(block_list, f, indent=4, sort_keys=True)
