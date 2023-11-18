import os
import argparse
import numpy as np
from PIL import Image
import json
import torch
from diffusers import StableDiffusionInpaintPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

view_dir = "views_img"
mask_dir = "simple_masks"

with open("block_edge_list.json", "r") as f:
    block_edge_list = json.load(f)

with open("edge_info.json", "r") as f:
    edge_info = json.load(f)

objects = ["chair", "table", "sofa", "potted plant", "basket", "exercise equipment", "vacuum cleaner", "suitcase", "toy", "dog"]
neg_prompts = "human, unrealistic, duplicate"

for obj in objects:

    capt = f"a {obj} on the center, blocking the road"

    views = []
    masks = []
    scans = list(block_edge_list.keys())
    for scan in scans[args.start:args.end]:
        paths = list(block_edge_list[scan].keys())
        for path in paths:
            for edge in block_edge_list[scan][path]:
                candidate = edge_info[path][f"{edge[0]}_{edge[1]}"]
                pointId = candidate["pointId"]
                view_path = os.path.join(view_dir, scan, edge[0], f"{pointId}.jpg")
                mask_path = os.path.join(mask_dir, scan, path, edge[0], edge[1], f"{pointId}.png")
                views.append(view_path)
                masks.append(mask_path)

    width, height = Image.open(views[0]).size
    view_imgs = []
    mask_imgs = []
    prompts = []
    save_paths = []
    for i, (view, mask) in enumerate(zip(views, masks)):
        input_im = Image.open(view)
        input_ma = Image.open(mask)
        view_imgs.append(input_im)
        mask_imgs.append(input_ma)
        prompts.append(capt)
        save_paths.append(mask.replace("masks", "all_inpaint_results").replace(".png", f"_{obj}.jpg"))
        if len(prompts) == args.batch_size:
            with torch.autocast("cuda"):
                images = pipe(prompt=prompts, 
                            image=view_imgs, 
                            mask_image=mask_imgs, 
                            height=height, 
                            width=width,
                            strength=1.,
                            guidance_scale=30,
                            negative_prompt=[neg_prompts]*args.batch_size,
                            ).images
            for j in range(args.batch_size):
                os.makedirs(os.path.dirname(save_paths[j]), exist_ok=True)
                images[j].save(save_paths[j])
            view_imgs = []
            mask_imgs = []
            prompts = []
            save_paths = []
        elif i == len(views) - 1:
            with torch.autocast("cuda"):
                images = pipe(prompt=prompts, 
                            image=view_imgs, 
                            mask_image=mask_imgs, 
                            height=height, 
                            width=width,
                            strength=1.,
                            guidance_scale=30,
                            negative_prompt=[neg_prompts]*len(prompts),
                            ).images
            print(len(prompts))
            for j in range(len(prompts)):
                os.makedirs(os.path.dirname(save_paths[j]), exist_ok=True)
                images[j].save(save_paths[j])
            view_imgs = []
            mask_imgs = []
            prompts = []
            save_paths = []

        if i % 100 == 0:
            print(f"{obj}, Done {i}/{len(views)} images!")
