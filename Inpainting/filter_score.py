import os
import json
import numpy as np
from scipy.stats import norm

from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

def visualize_filter(split):
    with open('inpaint_score_filtered.json', 'r') as f:
        inpain_score = json.load(f)

    scores = defaultdict(int)
    for k, v in inpain_score.items():
        category = k.split('_')[-1].split('.')[0]
        scores[category] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(scores.keys(), scores.values())
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.savefig('filtered.png')
    plt.close()

def visualize_final():
    with open(f'final_list.json', 'r') as f:
        inpain_score = json.load(f)

    scores = defaultdict(int)
    for k, v in inpain_score.items():
        category = v.split('_')[-1].split('.')[0]
        scores[category] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(scores.keys(), scores.values())
    plt.xlabel('Category')
    plt.ylabel('Number of Images')
    plt.savefig(f'final.png')
    plt.close()

# visualize_filter()
# visualize_final()
# exit(0)

with open('inpaint_score.json', 'r') as f:
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

with open('inpaint_score_filtered.json', 'w') as f:
    json.dump(filtered_names, f, indent=4)




