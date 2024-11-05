import os
import cv2
import json
import math
import MatterSim
import numpy as np
from tqdm import tqdm

def get_sim():
    image_w = 640
    image_h = 480
    vfov = 60
    scan_data_dir = '~/Matterport3D/v1/scans'
    connectivity_dir = "../VLN-DUET/datasets/R2R/connectivity"
        
    sim = MatterSim.Simulator()
    sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setDepthEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def calculate_cor(state, i):
    WIDTH = 640
    HEIGHT = 480
    focal_length = HEIGHT / (2 * np.tan(np.radians(30)))

    heading = state.navigableLocations[i].rel_heading
    elevation = state.navigableLocations[i].rel_elevation

    u = np.tan(heading) * focal_length + WIDTH / 2  
    v = np.tan(elevation) * focal_length + HEIGHT / 2
    
    return int(u), int(v)

def calculate_views(scanId, viewpointId):
    def _loc_distance(loc):
        return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
    adj_dict = {}
    for ix in range(36):
        if ix == 0:
            sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        
        assert state.viewIndex == ix
        
        # get adjacent locations
        for j, loc in enumerate(state.navigableLocations[1:]):
            distance = _loc_distance(loc)
            if (loc.viewpointId not in adj_dict or
                    distance < adj_dict[loc.viewpointId]['distance']):
                adj_dict[loc.viewpointId] = {
                    'viewpointId': loc.viewpointId,
                    'pointId': ix,
                    'distance': distance,
                    'idx': j,
                    'pos': calculate_cor(state, j+1)
                }
    candidate = list(adj_dict.values())
    return candidate

def draw_candidate(scanID, viewpointID, candidate):
    path = f"views_img/{scanID}/{viewpointID}"
    for cand in candidate:
        pointID = cand["pointId"]
        pos = cand["pos"]

        img_path = os.path.join(path, str(pointID)+".jpg")
        save_path = f"./{pointID}.jpg"
        
        img = cv2.imread(img_path) if not os.path.exists(save_path) else cv2.imread(save_path)
        u, v = pos

        radius = 5  
        color = (0, 0, 255)
        thickness = -1
        cv2.circle(img, (u, v), radius, color, thickness)
        cv2.imwrite(save_path, img)

anno_dir = "../VLN-DUET/datasets/R2R/annotations"
splits = ["train", "val_seen", "val_unseen"]
headings = {}
for split in splits:
    with open(anno_dir + f"/R2R_{split}_enc.json", "r") as f:
        data = json.load(f)
    headings.update({p["path_id"]: p["heading"] for p in data})

with open("block_edge_list.json", "r") as f:
    block_edge_list = json.load(f)

scans = list(block_edge_list.keys())
print("scans:", len(scans))

sim = get_sim()
edge_info = {}
for scan in tqdm(scans):
    paths = list(block_edge_list[scan].keys())
    for path in paths:
        if path in edge_info.keys():
            print("Already in!")
            exit(0)
        edge_info[path] = {}
        for edge in block_edge_list[scan][path]:
            sim.newEpisode([scan], [edge[0]], [headings[int(path)]], [0])
            state = sim.getState()[0]
            assert state.scanId == scan and state.location.viewpointId == edge[0]
            candidate = calculate_views(state.scanId, state.location.viewpointId)
            
            if edge[1] not in [cand['viewpointId'] for cand in candidate]:
                print("Not in candidate!")
                print(scan, path, edge)
                exit(0)

            for cand in candidate:
                if cand['viewpointId'] == edge[1]:
                    edge_info[path][f"{edge[0]}_{edge[1]}"] = cand
                    break
            
with open("edge_info.json", "w") as f:
    json.dump(edge_info, f, indent=4)