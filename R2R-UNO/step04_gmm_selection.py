import os
import json
from transformers import CLIPProcessor, CLIPModel

import torch
from PIL import Image
from glob import glob
import numpy as np

from collections import defaultdict
from sklearn.mixture import GaussianMixture

import torch.multiprocessing as mp
from progressbar import ProgressBar

num_workers = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_feature_extractor():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

def zeroshot_classify(model, processor, image, category):
    inputs = processor(text=[category], images=image, return_tensors="pt", padding=True).to(device)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    similarity = float(logits_per_image.detach().cpu().numpy()[0])

    return similarity

def process_images(proc_id, out_queue, image_paths):
    print('start proc_id: %d' % proc_id)

    torch.set_grad_enabled(False)
    model, processor = build_feature_extractor()

    for image_path in image_paths:
        category = image_path.split('_')[-1].split('.')[0]
        mask_path = image_path.replace("all_inpaint_results", "masks").split(category)[0][:-1] + '.png'
        mask = Image.open(mask_path).convert('L')
        image = Image.open(image_path).convert('RGB')

        masked_image = Image.composite(image, Image.new('RGB', image.size, 'white'), mask)
        similarity = zeroshot_classify(model, processor, masked_image, category)

        out_queue.put((image_path, similarity))

    out_queue.put(None)

def detect_img():
    inpaint_dir = 'all_inpaint_results'
    image_paths = glob(os.path.join(inpaint_dir, '*', '*', '*', '*', '*_*.jpg'))

    num_data_per_worker = len(image_paths) // num_workers
    out_queue = mp.Queue()

    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_images,
            args=(proc_id, out_queue, image_paths[sidx: eidx])
        )
        process.start()
        processes.append(process)

    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(image_paths))
    progress_bar.start()

    results = {}
    with open(f"inpaint_score.json", 'w') as f:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                path, score = res
                results[path] = score
                num_finished_vps += 1
                progress_bar.update(num_finished_vps)
        progress_bar.finish()
        for process in processes:
            process.join()

        json.dump(results, f, indent=4)

def gmm_filter():
    with open('inpain_score.json', 'r') as f:
        inpain_score = json.load(f)

    scores = defaultdict(list)
    names = defaultdict(list)
    for k, v in inpain_score.items():
        category = k.split('_')[-1].split('.')[0]
        scores[category].append(v)
        names[category].append(k)

    for k, v in scores.items():
        data = np.array(v)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(data.reshape(-1, 1))
        labels = gmm.predict(data.reshape(-1, 1))
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        gt_label = int(means[0] < means[1])
        
        names[k] = np.array(names[k])[labels == gt_label]

    filtered_names = {}
    for k, v in inpain_score.items():
        category = k.split('_')[-1].split('.')[0]
        if k in names[category]:
            filtered_names[k] = v

    with open('qualified_candidates.json', 'w') as f:
        json.dump(filtered_names, f, indent=4)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    detect_img()
    gmm_filter()
