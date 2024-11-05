import os
import cv2
import json
import random

def get_top3(prefix):
    with open(f"inpaint_score.json", "r") as f:
        data = json.load(f)
    scores = {}
    for obj in objects:
        k = prefix + obj + '.jpg'
        scores[obj] = data[k]
    top_3_obj = sorted(scores, key=scores.get, reverse=True)[:3]
    return [prefix + top_3_obj[i] + '.jpg' for i in range(3)]

def put_position():
    with open(f"final_list.json", "r") as f:
        data = json.load(f)
    for k, v in data.items():
        final_img = cv2.imread(os.path.join(k,v))
        new_path = os.path.join(k.replace("all_inpaint_results", "final_inpaint_results"), v)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        cv2.imwrite(new_path, final_img)

nope_num = 0
objects = ["chair", "table", "sofa", "potted plant", "basket", "exercise equipment", "vacuum cleaner", "suitcase", "toy", "dog"]

with open(f"qualified_candidat.json", "r") as f:
    data = json.load(f)

inpaint_dir = "all_inpaint_results"
with open(f"block_edge_list.json", "r") as f:
    block_edge_list = json.load(f)

with open(f"edge_info.json", "r") as f:
    edge_info = json.load(f)

image_paths = []
scans = list(block_edge_list.keys())
for scan in scans:
    paths = list(block_edge_list[scan].keys())
    for path in paths:
        for edge in block_edge_list[scan][path]:
            candidate = edge_info[path][f"{edge[0]}_{edge[1]}"]
            pointId = candidate["pointId"]
            image_path = os.path.join(inpaint_dir, scan, path, edge[0], edge[1])
            image_paths.append(image_path+f'/{pointId}_')

final_results = {}
for image_path in image_paths:
    candidate = []
    for obj in objects:
        k = image_path + obj + '.jpg'
        if k in data.keys():
            candidate.append(k)
    if len(candidate) == 0:
        nope_num += 1
        candidate = get_top3(image_path)
    
    chosen = random.choice(candidate)
    img_name = chosen.split('/')[-1]
    final_results[chosen.split(img_name)[0][:-1]] = img_name

with open(f"final_list.json", "w") as f:
    json.dump(final_results, f, indent=4)

print(nope_num)
print(len(image_paths))
put_position()
