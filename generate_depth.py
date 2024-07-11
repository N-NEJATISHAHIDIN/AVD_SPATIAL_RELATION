import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import matplotlib
from scipy import stats
from sklearn.linear_model import LinearRegression

def align_depth_maps(generated_depth, ground_truth_depth):
    # Reshape the depth maps to be 1D arrays
    generated_depth_flat = generated_depth.flatten().reshape(-1, 1)
    ground_truth_depth_flat = ground_truth_depth.flatten().reshape(-1, 1)
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(generated_depth_flat, ground_truth_depth_flat)
    
    # Extract the scale and shift (slope and intercept)
    scale = model.coef_[0][0]
    shift = model.intercept_[0]
    
    return scale, shift

def get_sigma_mask(array):
    mean = np.mean(array)
    sigma = np.std(array)
    return (array >= mean - sigma) & (array <= mean + sigma)

def process_image(image_path, output_folder, model, model_name):

    image = Image.open(image_path).convert("RGB") 
    
    depth_path = os.path.join(image_path.split("/jpg_rgb")[0], "high_res_depth")
    img_name = os.path.basename(image_path)
    ground_truth_depth_path = f'{depth_path}/{img_name[:14]}3.png'
    depth_image = cv2.imread(ground_truth_depth_path, cv2.IMREAD_UNCHANGED)
    mask = (depth_image>0)
    values_gt = depth_image.flatten()[mask.flatten()]
    # mask = get_sigma_mask(depth_image)
    # values_gt = depth_image.flatten()[mask.flatten()]
    # import pdb; pdb.set_trace()

    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    if model_name == "zoe_depth":
        depth= model.infer_pil(image)
        # import pdb; pdb.set_trace()
        value_pred = depth.flatten()[mask.flatten()]

        # depth = 255 - ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0)
        # depth = depth.astype(np.uint8)
        # plt.imshow(depth, cmap= 'Spectral_r')
        # plt.show()


    elif model_name == "depth_anything":
        raw_img = cv2.imread(image_path)
        depth = model.infer_image(raw_img) # HxW raw depth map
        # depth = 255 - ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0)
        # depth = depth.astype(np.uint8)
        # plt.imshow(depth, cmap= 'Spectral_r')
        # plt.show()
        value_pred = depth.flatten()[mask.flatten()]

    else:
        return "NOT IMPLIMENTED ERROR"
    
    scale, shift = align_depth_maps(value_pred, values_gt)
    aligned_depth_map = scale * depth + shift


    npy_file_path = os.path.join(output_folder, os.path.basename(image_path)[:-4]+'.npy')
    np.save(npy_file_path, aligned_depth_map)

def process_images(input_folder, output_folder, model, model_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in tqdm(sorted(os.listdir(input_folder))):
        image_path = os.path.join(input_folder, image_name)
        process_image(image_path, output_folder, model, model_name)


if __name__ == "__main__":


    device = 'cuda'


    # # ZoeDepth
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
    # repo = "isl-org/ZoeDepth"
    # # Zoe_N
    # model = torch.hub.load(repo, "ZoeD_N", pretrained=True).to(device)
    # # Zoe_K
    # model_zoe_n = torch.hub.load(repo, "ZoeD_K", pretrained=True)
    # # Zoe_NK
    # model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

    import sys
    sys.path.insert(0, "./Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    # take depth-anything-v2-large as an example
    model = DepthAnythingV2(encoder='vitl', features=256).to(device)
    model.load_state_dict(torch.load('./Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth', map_location='cuda'))
    model.eval()

    output_folder = "./output/Home_001_1_depth_anything_aligned_depth_map"
    input_folder = '/home/negar/secondssd/AVD_spatial_relation/ActiveVisionDataset/Home_001_1/jpg_rgb'

    model_name = "depth_anything"
    # model_name = "zoe_depth"
    process_images(input_folder, output_folder, model, model_name)



