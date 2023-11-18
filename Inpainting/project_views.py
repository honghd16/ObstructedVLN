import os
import cv2
import numpy as np
import json
from tqdm import tqdm

def get_surrounding_indices(index):
    row = index // 12
    col = index % 12
    
    indices = []
    
    # Add left neighbor
    if col > 0:
        indices.append(index - 1)
    else:
        indices.append(index + 11)
    
    # Add right neighbor
    if col < 11:
        indices.append(index + 1)
    else:
        indices.append(index - 11)  # Wrap around to the first image in the row
    
    # Add top neighbor
    if row < 2:
        indices.append(index + 12)
    
    # Add bottom neighbor
    if row > 0:
        indices.append(index - 12)
    
    return indices

def apply_transform(img, M):
    # Apply the transformation to the image
    if M is not None:
        transformed_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    else:
        transformed_img = np.zeros_like(img)

    return transformed_img

def find_transform(img1, img2):
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use a Brute Force matcher to find matches between descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    thres = 0.75

    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < thres * n.distance:
            good.append(m)
    
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if len(good) < 4:
        M = None
    else:
        # Find the transformation matrix
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M

def get_surrounding_view(inpaint, view_path_1, view_path_2):

    inpaint_img = cv2.imread(inpaint)
    # Load the images
    imgA = cv2.imread(view_path_1)
    imgB = cv2.imread(view_path_2)

    # Find the transformation from imgA to imgB
    try:
        M = find_transform(imgA, imgB)
    except:
        M = None

    # Apply the same transformation to imgC
    final_view = apply_transform(inpaint_img, M)

    return final_view

def combine_images_with_mask(mask, img_A, img_B):
    assert mask.shape == img_A.shape and mask.shape == img_B.shape, "All images must have the same shape."

    mask_binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_binary, 1, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_binary = cv2.erode(mask_binary, kernel, iterations=1)

    mask_inv = cv2.bitwise_not(mask_binary)

    masked_region_A = cv2.bitwise_and(img_A, img_A, mask=mask_binary)
    masked_region_B = cv2.bitwise_and(img_B, img_B, mask=mask_inv)

    result_image = cv2.add(masked_region_A, masked_region_B)

    return result_image

view_dir = "views_img"
mask_dir = "masks"
inpa_dir = "final_inpaint_results"

with open("block_edge_list.json", "r") as f:
    block_edge_list = json.load(f)

with open("edge_info.json", "r") as f:
    edge_info = json.load(f)

scans = list(block_edge_list.keys())
for scan in tqdm(scans):
    paths = list(block_edge_list[scan].keys())
    for path in paths:
        for edge in block_edge_list[scan][path]:
            candidate = edge_info[path][f"{edge[0]}_{edge[1]}"]
            pointId = candidate["pointId"]
            
            inpaint_dir = os.path.join(inpa_dir, scan, path, edge[0], edge[1])
            inpaint_name = os.listdir(inpaint_dir)[0]
            inpaint_path = os.path.join(inpaint_dir, inpaint_name)
            
            surroundings = get_surrounding_indices(pointId)
            for sur in surroundings:
                sur_mask_path = os.path.join(mask_dir, scan, path, edge[0], edge[1], f"{sur}.png")
                view_path_1 = os.path.join(view_dir, scan, edge[0], f"{pointId}.jpg")
                view_path_2 = os.path.join(view_dir, scan, edge[0], f"{sur}.jpg")
                sur_view = get_surrounding_view(inpaint_path, view_path_1, view_path_2)

                sur_mask = cv2.imread(sur_mask_path)
                final_sur_view = combine_images_with_mask(sur_mask, sur_view, cv2.imread(view_path_2))

                save_path = os.path.join(inpaint_dir, f"{sur}.jpg")
                cv2.imwrite(save_path, final_sur_view)
                
