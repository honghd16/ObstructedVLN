# DynamicVLN
This repository is the official implementation of the CVPR 2024 submission "Adapt or Fail: Vision-and-Language Navigation in Dynamic Environments."

Real-world navigation often involves dealing with ever-changing environments where doors might open or close, objects can be moved, and entities might traverse unpredictably. However, mainstream Vision-and-Language Navigation (VLN) tasks are trained and evaluated in unchanging environments with fixed and predefined navigation graphs, implicitly assuming that instructions perfectly match reality. Such a static paradigm overlooks potential discrepancies in the navigation graph and variances between instructions and real-world dynamic scenarios, which are prevalent for both indoor and outdoor agents. Therefore, we introduce \textbf{DyVEG} (\textbf{Dy}namic \textbf{V}LN \textbf{E}nvironments \textbf{G}eneration), an innovative approach that infuses real-world dynamics into VLN environments at both the navigation graph and visual levels, to 1) investigate the impact of this large gap on the agent's performance under changes 2) develop new strategies for agents to bridge this gap effectively. Applying DyVEG to the R2R dataset, we develop the Dynamic R2R (DY-R2R) dataset, which brings various environment changes by incorporating different numbers and types of path obstructions. Our comprehensive experiments on DY-R2R demonstrate that state-of-the-art VLN methods inevitably encounter significant challenges in dynamic environments. Subsequently, a novel method called DyVLN (Dynamic VLN) is proposed, which includes a curriculum training strategy and virtual graph construction to help agents effectively adapt to such dynamics. Empirical results show that DyVLN not only maintains robust performance in static scenarios but also achieves a substantial performance advantage when facing environment changes.

![model_arch](figures/teaser.png)

## Progress
- [X] Installation
- [X] Code for DyVEG
- [X] Code for DyVLN

## Installation
1. Please follow the instructions [here](https://github.com/peteanderson80/Matterport3DSimulator#building-using-docker) to install Matterport3D Simulator.
We use the latest version instead of v0.1.
Make sure the 'import Mattersim' will not raise ImportError.

2. Setup with Anaconda and pip to install prerequisites:
```
conda create --name DynamicVLN python==3.8
conda activate DynamicVLN
pip install -r requirements.txt
```

## DyVEG for generating DY-R2R
```
cd Inpainting
```
Download the views images from [here](https://github.com/airsplay/R2R-EnvDrop) and put it in the 'views_img' folder

### Step 1: obstruct redundant edges
change the 'x' to the number of edges you want to block
```
python block_edge.py --nums=x
```
Output: 'block_{x}_edge_list.json' file storing all the blocking information.

### Step 2: localize target node
```
python find_view.py
```
Output: 'edge_info.json' file storing all the locations of the node pairs

### Step 3: generate mask
```
python generate_mask.py
```
Output: 'masks' folder storing all the masks

### Step 4: inpainting
```
python inpaint_pipeline.py --batch_size bs
```
Output: 'all_inpaint_results' folder storing all the inpainting results for each obstruction

### Step 5: clip evaluation
```
python evaluate_score.py
```
Output: 'inpaint_score.json' file storing the compatibility score for each candidate

### Step 6: GMMs training and inference
```
python filter_score.py
```
Output: 'inpaint_score_filtered.json' file storing all the qualified candidates

### Step 7: final choice
```
python generate_final_list.py
```
Output: 
1. 'final_list.json' file storing the chosen inpainting names for each edge
2. 'final_inpaint_results' folder storing the chosen inpainting resutls

### Step 8: view projection
```
python project_views.py
```
Output: the project views in the 'final_inpaint_results'

## DyVLN
We provide the code for training DUET for R2R and DY-R2R here. HAMT and REVERIE will be added soon.

### Step 1: data download
Please follow the instructions [here](https://github.com/cshizhe/VLN-DUET) to download the annotations, connectivity files, pretrained models and features for R2R.

### Step 2: shortest path generation
```
cd Inpainting
python generate_shortest_distance.py
```
This will generate a "shortest_distances" folder to save the shortest distance dict of the modified graphs

### Step 3: dynamic environments feature extraction
```
cd Inpainting
python compute_feature.py
```
This will generate a 'inpaint_features.hdf5' file saving the extracted features by ViT-B/16

### Step 4: finetuning w/o DyVLN
```
cd VLN-DUET
mv ../Inpainting/block_*_edge_list.json ./datasets/R2R/annotations
mv ../Inpainting/inpaint_features.hdf5 ./datasets/R2R/features
cd map_nav_src
bash scripts/run_r2r.sh
```
You can modify the 'max_train_edge' and 'max_eval_edge' in 'run_r2r.sh' to choose which set of DY_R2R to train and evaluate

### Step 5: finetuning with DyVLN
```
cd VLN-DUET
cd map_nav_src_dyvln
bash scripts/run_r2r.sh
```
Also modify the 'max_train_edge' and 'max_eval_edge' in 'run_r2r.sh'