import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

skip_num = 0

mask_dir = "masks"
view_dir = "views_img"

def create_parallelogram_mask(point, image_shape, width, bias):
    mask = np.zeros(image_shape, dtype=np.uint8)

    if (point[1]+bias) < 0:
        bias = 0
    highest = point[1]+bias if point[1]+bias < image_shape[0] else image_shape[0]-1
    # Determine the four points of the parallelogram
    top_left = (point[0] - width // 2, highest)
    top_right = (point[0] + width // 2, highest)
    bottom_center = (image_shape[1] // 2, image_shape[0] - 1)
    bottom_left = (bottom_center[0] - width // 2, bottom_center[1])
    bottom_right = (bottom_center[0] + width // 2, bottom_center[1])
    if top_left[0] < bottom_left[0]:
        bottom_left = (top_left[0], bottom_left[1])
    elif top_right[0] > bottom_right[0]:
        bottom_right = (top_right[0], bottom_right[1])

    parallelogram = np.array([top_left, top_right, bottom_right, bottom_left])

    # Draw the filled parallelogram on the mask
    cv2.fillPoly(mask, [parallelogram], 255)

    return mask

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

def apply_transform(img, M):
    global skip_num
    # Apply the transformation to the image
    if M is not None:
        transformed_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    else:
        transformed_img = np.zeros_like(img)
        skip_num += 1

    return transformed_img

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
        indices.append(index - 11)
    
    # Add top neighbor
    if row < 2:
        indices.append(index + 12)
    
    # Add bottom neighbor
    if row > 0:
        indices.append(index - 12)
    
    # Add bottom left
    if row > 0 and col > 0:
        indices.append(index - 13)
    elif row > 0 and col == 0:
        indices.append(index - 1)
    
    # Add bottom right
    if row > 0 and col < 11:
        indices.append(index - 11)
    elif row > 0 and col == 11:
        indices.append(index - 23)
    
    # Add top left
    if row < 2 and col > 0:
        indices.append(index + 11)
    elif row < 2 and col == 0:
        indices.append(index + 24)

    # Add top right
    if row < 2 and col < 11:
        indices.append(index + 13)
    elif row < 2 and col == 11:
        indices.append(index + 1)
    
    return indices

def get_surrounding_mask(mask, view_path_1, view_path_2):
    imgA = cv2.imread(view_path_1)
    imgB = cv2.imread(view_path_2)

    try:
        M = find_transform(imgA, imgB)
    except:
        M = None

    final_mask = apply_transform(mask, M)

    return final_mask

def get_mask(pointId, pos):
    if pointId < 12:
        mask = create_parallelogram_mask(pos, (480,640), width=200, bias=-100)
    elif pointId < 24:
        mask = create_parallelogram_mask(pos, (480,640), 200, bias=0)
    else:
        mask = create_parallelogram_mask(pos, (480,640), 200, bias=100)
    cv2.imwrite("mask.png", mask)

    return mask

def process_scan(scan, block_edge_list, edge_info):
    paths = list(block_edge_list[scan].keys())
    for path in tqdm(paths, desc="Path", leave=False):
        for edge in block_edge_list[scan][path]:
            candidate = edge_info[path][f"{edge[0]}_{edge[1]}"]
            pointId = candidate["pointId"]
            pos = candidate["pos"]
            
            mask_path = os.path.join(mask_dir, scan, path, edge[0], edge[1])
            os.makedirs(mask_path, exist_ok=True)
            
            mask = get_mask(pointId, pos)
            cv2.imwrite(os.path.join(mask_path, f"{pointId}.png"), mask)
            # cv2.imwrite(f"mask_{x}.png", mask)

            surroundings = get_surrounding_indices(pointId)
            for sur in surroundings:
                view_path_1 = os.path.join(view_dir, scan, edge[0], f"{pointId}.jpg")
                view_path_2 = os.path.join(view_dir, scan, edge[0], f"{sur}.jpg")
                sur_mask = get_surrounding_mask(mask, view_path_1, view_path_2)
                cv2.imwrite(os.path.join(mask_path, f"{sur}.png"), sur_mask)

def main():
    with open("block_edge_list.json", "r") as f:
        block_edge_list = json.load(f)

    with open("edge_info.json", "r") as f:
        edge_info = json.load(f)

    scans = list(block_edge_list.keys())
    
    with Pool() as pool:
        pool_args = [(scan, block_edge_list, edge_info) for scan in scans]
        list(tqdm(pool.starmap(process_scan, pool_args), total=len(scans), desc="Scan"))

if __name__ == '__main__':
    main()    


