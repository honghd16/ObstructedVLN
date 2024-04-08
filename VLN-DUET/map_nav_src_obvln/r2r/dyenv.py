''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy
from tqdm import tqdm
import pickle

import MatterSim

from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature

from r2r.eval_utils import cal_dtw, cal_cls

ERROR_MARGIN = 3.0

def calculate_new_position(x0, y0, z0, heading, elevation, distance=3.):
    x = x0 + distance * math.cos(elevation) * math.sin(heading)
    y = y0 + distance * math.cos(elevation) * math.cos(heading)
    z = z0 + distance * math.sin(elevation)

    return (x, y, z)

class DyEnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, batch_size=100, use_inpaint=False):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.use_inpaint = use_inpaint
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir:
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setBatchSize(1)
            sim.initialize()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self, blocks, paths):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]
            edges = blocks[i]
            feature = None
            if edges and self.use_inpaint:
                for edge in edges:
                    if state.location.viewpointId == edge[0]:
                        feature = self.feat_db.get_block_feature(state.scanId, paths[i], edge[0], edge[1])
                        break
                    elif state.location.viewpointId == edge[1]:
                        feature = self.feat_db.get_block_feature(state.scanId, paths[i], edge[1], edge[0])
                        break
            if feature is None:
                feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)
            feature_states.append((feature, state))
        return feature_states

    def getState(self, block, path, idx):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        sim = self.sims[idx]
        state = sim.getState()[0]
        edges = block
        feature = None
        if edges and self.use_inpaint:
            for edge in edges:
                if state.location.viewpointId == edge[0]:
                    feature = self.feat_db.get_block_feature(state.scanId, path, edge[0], edge[1])
                    break
                elif state.location.viewpointId == edge[1]:
                    feature = self.feat_db.get_block_feature(state.scanId, path, edge[1], edge[0])
                    break
        if feature is None:
            feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)
        return (feature, state)

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])

class DynamicBatch(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, view_db, instr_data, connectivity_dir, 
        batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None,
        block_num=0, use_inpaint=False
    ):
        self.env = DyEnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size, use_inpaint=use_inpaint)
        self.scans = set([x['scan'] for x in instr_data])
        self.connectivity_dir = connectivity_dir
        self._load_nav_graphs()
        self.name = name
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.block_num = block_num

        self.data = self._construct_dynamic_data(instr_data, block_num)
        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits 
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)
        
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _construct_dynamic_data(self, instr_data, block_num):    
        print('Start loading splits: %s' % (self.name))
        data = []
        block_jsons = []
        self.block_list = {}
        
        for i in range(1, block_num+1):
            with open(f"../datasets/R2R/annotations/block_{i}_edge_list.json", "r") as f:
                block_json = json.load(f)
                block_jsons.append(block_json)

        for item in tqdm(instr_data):
            item_0 = copy.deepcopy(item)
            item_0['instr_id'] = item_0['instr_id'] + '_0_0'
            item_0['original_path'] = item_0['path'].copy()
            item_0['r2r_path'] = item_0['path'].copy()
            item_0['detailed_path'] = item_0['path'].copy()
            item_0['block'] = None
            data.append(item_0)

            self.block_list[item_0['instr_id']] = None
            for i in range(1, block_num+1):
                
                block_json = block_jsons[i-1]
                scan = item['scan']
                cands = block_json.get(scan, {}).get(str(item['path_id']), [])
                
                if len(cands) == 0:
                    continue

                for j, cand in enumerate(cands):
                    item_copy = copy.deepcopy(item)
                    item_copy['instr_id'] = item_copy['instr_id'] + f'_{i}_{j}'
                    item_copy['r2r_path'] = item_copy['path'].copy()
                    block = tuple(tuple(sublist) for sublist in cand[0].copy())
                    item_copy['block'] = block
                    item_copy['path'] = cand[1].copy()
                    item_copy['original_path'] = cand[2].copy()
                    item_copy['detailed_path'] = cand[3].copy()
                    data.append(item_copy)
                    self.block_list[item_copy['instr_id']] = block
                    
        return data

    def _get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['detailed_path']) \
                for x in data if len(x['detailed_path']) > 1
        }
        return gt_trajs

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            batch += random.sample(self.data[:self.ix], batch_size - len(batch))
            self.ix = batch_size - len(batch)
            random.shuffle(self.data)
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def make_candidate(self, feature, scanId, viewpointId, viewId, block):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        seen_dict = {}
        for ix in range(36):
            if ix == 0:
                self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                self.sim.makeAction([0], [1.0], [1.0])
            else:
                self.sim.makeAction([0], [1.0], [0])

            state = self.sim.getState()[0]
            assert state.viewIndex == ix

            # Heading and elevation for the viewpoint center
            heading = state.heading - base_heading
            elevation = state.elevation - base_elevation

            visual_feat = feature[ix]

            # get adjacent locations
            for j, loc in enumerate(state.navigableLocations[1:]):
                distance = _loc_distance(loc)

                if block and any({state.location.viewpointId, loc.viewpointId} == set(edge) for edge in block):
                    if (loc.viewpointId not in seen_dict or distance < seen_dict[loc.viewpointId]['distance']):
                        angle_feat = angle_feature(heading, elevation, self.angle_feat_size)
                        seen_dict[loc.viewpointId] = {
                            'heading': heading,
                            'elevation': elevation,
                            "normalized_heading": state.heading,
                            "normalized_elevation": state.elevation,
                            'scanId': scanId,
                            'viewpointId': f"seen_node_{state.location.viewpointId}_{j+1}", # Next viewpoint id
                            'real_viewpointId': loc.viewpointId, # real viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            "real_position": (loc.x, loc.y, loc.z),
                        }
                else:
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    if (loc.viewpointId not in adj_dict or distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }
        candidate = list(adj_dict.values())
        seen = list(seen_dict.values())
        for seen_node in seen:
            seen_node['distance'] = 3
            seen_node['position'] = calculate_new_position(state.location.x, state.location.y, state.location.z, seen_node['normalized_heading'], seen_node['normalized_elevation'], distance=3)

        return candidate, seen

    def _get_obs(self):
        obs = []
        paths = [x["path_id"] for x in self.batch]
        blocks = [x['block'] for x in self.batch]
        for i, (feature, state) in enumerate(self.env.getStates(blocks, paths)):
            item = self.batch[i]
            base_view_id = state.viewIndex
           
            # Full features
            candidate, seen = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex, blocks[i])
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            if blocks[i]:
                NewNavigableLocations = [
                    x for x in state.navigableLocations
                    if not any({state.location.viewpointId, x.viewpointId} == set(edge) for edge in blocks[i])
                ]
            else:
                NewNavigableLocations = state.navigableLocations

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : NewNavigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': item['instr_encoding'],
                'gt_path' : item['path'],
                'endpoint': item['r2r_path'][-1],
                'path_id' : item['path_id'],
                'block': item['block'],
                'seen_node': seen,
                'original_path': item['original_path']
            }

            obs.append(ob)
        return obs

    def _get_ob(self, idx):
        path = self.batch[idx]["path_id"]
        block = self.batch[idx]['block']
        feature, state = self.env.getState(block, path, idx)
        item = self.batch[idx]
        base_view_id = state.viewIndex

        # Full features
        candidate, seen = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex, block)
        # [visual_feature, angle_feature] for views
        feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
        if block:
            NewNavigableLocations = [
                x for x in state.navigableLocations
                if not any({state.location.viewpointId, x.viewpointId} == set(edge) for edge in block)
            ]
        else:
            NewNavigableLocations = state.navigableLocations

        ob = {
            'instr_id' : item['instr_id'],
            'scan' : state.scanId,
            'viewpoint' : state.location.viewpointId,
            'viewIndex' : state.viewIndex,
            'position': (state.location.x, state.location.y, state.location.z),
            'heading' : state.heading,
            'elevation' : state.elevation,
            'feature' : feature,
            'candidate': candidate,
            'navigableLocations' : NewNavigableLocations,
            'instruction' : item['instruction'],
            'instr_encoding': item['instr_encoding'],
            'gt_path' : item['path'],
            'endpoint': item['r2r_path'][-1],
            'path_id' : item['path_id'],
            'block': item['block'],
            'seen_node': seen,
        }

        return ob

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)

        self.shortest_distances = {}
        for i in range(self.batch_size):
            scan = self.batch[i]['scan']
            block = self.batch[i]['block']
            shortest_distances = self.read_shortest_distances(scan, block)
            self.shortest_distances[(scan, block)] = shortest_distances

        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()


    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def read_shortest_distances(self, scan, block):
        distance_dir = f'../datasets/R2R/shortest_distances/{self.name}'
        filename = f"{distance_dir}/{(scan,block)}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                distances = pickle.load(f)
            return distances
        else:
            print(f"File {filename} not found!")
            return None

    def _eval_item(self, scan, instr_id, pred_path, gt_path):
        scores = {}
        shortest_distances = self.read_shortest_distances(scan, self.block_list[instr_id])

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'
        
        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(lambda: defaultdict(list))
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, instr_id, traj, gt_traj)
            block_num = instr_id.split('_')[2]
            for k, v in traj_scores.items():
                metrics[block_num][k].append(v)
            metrics[block_num]['instr_id'].append(instr_id)
        
        avg_metrics = {}
        for i in range(self.block_num+1):
            metric = metrics[str(i)]
            avg_metric = {
                'action_steps': np.mean(metric['action_steps']),
                'steps': np.mean(metric['trajectory_steps']),
                'lengths': np.mean(metric['trajectory_lengths']),
                'nav_error': np.mean(metric['nav_error']),
                'oracle_error': np.mean(metric['oracle_error']),
                'sr': np.mean(metric['success']) * 100,
                'oracle_sr': np.mean(metric['oracle_success']) * 100,
                'spl': np.mean(metric['spl']) * 100,
                'nDTW': np.mean(metric['nDTW']) * 100,
                'SDTW': np.mean(metric['SDTW']) * 100,
                'CLS': np.mean(metric['CLS']) * 100,
            }
            avg_metrics[str(i)] = avg_metric
            
        return avg_metrics, metrics
        
