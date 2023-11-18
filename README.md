# DynamicVLN
This repository is the official implementation of the CVPR 2024 submission "Adapt or Fail: Vision-and-Language Navigation in Dynamic Environments."

Real-world navigation often involves dealing with ever-changing environments where doors might open or close, objects can be moved, and entities might traverse unpredictably. However, mainstream Vision-and-Language Navigation (VLN) tasks are trained and evaluated in unchanging environments with fixed and predefined navigation graphs, implicitly assuming that instructions perfectly match reality. Such a static paradigm overlooks potential discrepancies in the navigation graph and variances between instructions and real-world dynamic scenarios, which are prevalent for both indoor and outdoor agents. Therefore, we introduce \textbf{DyVEG} (\textbf{Dy}namic \textbf{V}LN \textbf{E}nvironments \textbf{G}eneration), an innovative approach that infuses real-world dynamics into VLN environments at both the navigation graph and visual levels, to 1) investigate the impact of this large gap on the agent's performance under changes 2) develop new strategies for agents to bridge this gap effectively. Applying DyVEG to the R2R dataset, we develop the Dynamic R2R (DY-R2R) dataset, which brings various environment changes by incorporating different numbers and types of path obstructions. Our comprehensive experiments on DY-R2R demonstrate that state-of-the-art VLN methods inevitably encounter significant challenges in dynamic environments. Subsequently, a novel method called DyVLN (Dynamic VLN) is proposed, which includes a curriculum training strategy and virtual graph construction to help agents effectively adapt to such dynamics. Empirical results show that DyVLN not only maintains robust performance in static scenarios but also achieves a substantial performance advantage when facing environment changes.

![model_arch](figures/teaser.png)

## Progress
- [X] Installation
- [X] Code for DyVEG
- [X] Code for DyVLN

## Installation
1. Please follow the instructions [(here)](https://github.com/peteanderson80/Matterport3DSimulator#building-using-docker) to install Matterport3D Simulator.
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

### Step 1: find redundant edges
```
bash run_clip.sh
```

### Step 2: localize target node

### Step 3: generate mask

### Step 4: inpainting

### Step 5: clip evaluation

### Step 6: GMM training

### Step 7: final choice


## DyVLN
```
cd VLN-DUET
```
### Step 1: data download


### Step 2: shortest path generation


### Step 3: finetuning w/o DyVLN


### Step 4: finetuning with DyVLN

