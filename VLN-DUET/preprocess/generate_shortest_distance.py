import json
from tqdm import tqdm
import os
import numpy as np
import copy
import pickle
import networkx as nx

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data

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
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

block_jsons = []

for i in range(1, 4):
    with open(f"block_{i}_edge_list.json", "r") as f:
        block_json = json.load(f)
        block_jsons.append(block_json)

anno_dir = "../VLN-DUET/datasets/R2R/annotations"
connectivity_dir = "../VLN-DUET/datasets/R2R/connectivity"
splits = ['train', 'val_train_seen', 'val_seen', 'val_unseen']

for split in splits:
    shortest_distances = []
    instr_data = construct_instrs(
            anno_dir, 'r2r', [split], 
            tokenizer='bert', max_instr_len=200,
            is_test=False
    )

    scans = set([x['scan'] for x in instr_data])
    graphs = load_nav_graphs(connectivity_dir, scans)

    directory = f'shortest_distances/{split}'
    os.makedirs(directory, exist_ok=True)

    for item in tqdm(instr_data):
        item_0 = copy.deepcopy(item)
        item_0['block'] = None
        if (item_0['scan'], item_0['block']) not in shortest_distances:
            G = graphs[item_0['scan']].copy()
            value = dict(nx.all_pairs_dijkstra_path_length(G))
            key = (item_0['scan'], item_0['block'])
            shortest_distances.append(key)
            filename = f"{directory}/{key}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(value, f)

        for i in range(1, 4): 
            block_json = block_jsons[i-1]
            scan = item['scan']
            cands = block_json.get(scan, {}).get(str(item['path_id']), [])
            
            if len(cands) == 0:
                continue

            for j, cand in enumerate(cands):
                item_copy = copy.deepcopy(item)
                block = tuple(tuple(sublist) for sublist in cand[0].copy())
                item_copy['block'] = block
                
                if (item_copy['scan'], item_copy['block']) not in shortest_distances:
                    G = graphs[item_copy['scan']].copy()
                    for edge in item_copy['block']:
                        G.remove_edge(edge[0], edge[1])
                    value = dict(nx.all_pairs_dijkstra_path_length(G))
                    key = (item_copy['scan'], item_copy['block'])
                    shortest_distances.append(key)
                    filename = f"{directory}/{key}.pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(value, f)

        