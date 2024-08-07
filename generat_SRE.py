# python generat_SRE.py ./output_labels ./ActiveVisionDataset/Home_001_1/high_res_depth ./ActiveVisionDataset/Home_001_1/jpg_rgb ./ActiveVisionDataset/ ./SRE_annotations 
import bz2
import os
import sys
import json
import pickle
import csv

import numpy as np
import argparse
from pathlib import Path
import _pickle as cPickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy import linalg as LA
from tqdm import tqdm
import gzip
from utils_AVD import generatePcdinCameraCoordinat, generatePcdinCameraCoordinat_per_object

sys.path.insert(0, '/home/negar/secondssd/semantic_abs/baseline')

from utils import draw_3D_pointcloud
spatial_relations = {'behind', 'left of', 'right of', 'in front of', 'on top of', 'inside'}
csv_file = 'SRE_annotations_fixed_label_ fixed_depth.csv'

# Define the headers and rows of data
headers = ["image_name", "is_behind", "is_in_front", "is_above", "is_below", "is_left", "is_right", "target_obj_id", "reference_obj_id"]



# Function to load and sort JSON by values
def load_and_sort_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Filter the dictionary to only include items with values more than 50
    filtered_data = {k: v for k, v in data.items() if v > 50}

    # Sort the filtered dictionary by values
    sorted_data = dict(sorted(filtered_data.items(), key=lambda item: item[1]))

    return sorted_data





def generate_SRE(label_path, depth_path, rgb_path, dataPath, output_labels):

    folder_path = Path(label_path)
    file_names = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix != '.jpg']

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        for img_name in tqdm(sorted(file_names)):

            depth_image_path = f'{depth_path}/{img_name[:14]}3.png'
            # depth_image_path = f'{depth_path}/{img_name[:15]}.npy'

            # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

            rgb_image_path = f'{rgb_path}/{img_name[:15]}.jpg'
            rgb_image = cv2.cvtColor(cv2.imread(rgb_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            
            with gzip.open(f'{label_path}/{img_name[:15]}.pkl.gz', 'rb') as f:
                labels = pickle.load(f)

            # # Display the image
            # fig, ax = plt.subplots()
            # ax.imshow(rgb_image)
            # ax.axis('off')  # Turn off axis

            # for target_obj_id in range(labels['num_instances']):
            #     for idx, detic_pred_boxes in enumerate(labels['pred_boxes']) :
            #         rect = patches.Rectangle((detic_pred_boxes[1], detic_pred_boxes[0]), 
            #                             detic_pred_boxes[3] - detic_pred_boxes[1], 
            #                             detic_pred_boxes[2] - detic_pred_boxes[0],
            #                             linewidth=1, edgecolor="red", facecolor='none')
            #         ax.add_patch(rect)
            #         ax.text(detic_pred_boxes[1], detic_pred_boxes[0], labels['pred_classes_names'][idx], color='white', fontsize=8, backgroundcolor='black')

            # plt.show()
            # plt.clf()


            # import pdb; pdb.set_trace()
            # target_obj_box = generatePcdinCameraCoordinat_per_object(dataPath, 'Home_001_1', rgb_image_path, depth_image_path, output_labels, labels, 1 )

            # index = 0
            # # continue

            # generatePcdinCameraCoordinat(dataPath, 'Home_001_1', rgb_image_path, depth_image_path, output_labels, labels ) #img_name[:15]+labels['pred_classes_names'][0], labels['pred_masks'][0] )
            for target_obj_id in range(labels['num_instances']):
                target_obj_box = generatePcdinCameraCoordinat_per_object(dataPath, 'Home_001_1', rgb_image_path, depth_image_path, output_labels, labels, target_obj_id )
                # if labels['pred_classes_names'][target_obj_id] not in CLASSES.keys():
                #     continue
                if target_obj_box is None :
                    continue
                for reference_obj_id in range(labels['num_instances']):
                    reference_obj_box = generatePcdinCameraCoordinat_per_object(dataPath, 'Home_001_1', rgb_image_path, depth_image_path, output_labels, labels, reference_obj_id )
                    captions = []

                    # if labels['pred_classes_names'][reference_obj_id] not in CLASSES.keys():
                    #     continue
                    if (labels['pred_classes'][target_obj_id] 
                        == labels['pred_classes'][reference_obj_id]) :
                        continue
                    elif reference_obj_box is None:
                        continue

                    # 2D info 
                    target_obj_2D_box = labels['pred_boxes'][target_obj_id]
                    reference_obj_2D_box = labels['pred_boxes'][reference_obj_id]

                    target_2D_center =  np.array([(target_obj_2D_box[1] + target_obj_2D_box[3]) /2 , (target_obj_2D_box[0] + target_obj_2D_box[2]) /2])
                    reference_2D_center =  np.array([(reference_obj_2D_box[1] + reference_obj_2D_box[3]) /2 , (reference_obj_2D_box[0] + reference_obj_2D_box[2]) /2 ])

                    target_2D_dim = [(target_obj_2D_box[3] - target_obj_2D_box[1])/2 , (target_obj_2D_box[2] - target_obj_2D_box[0]) /2] 
                    reference_2D_dim =  [(reference_obj_2D_box[3] - reference_obj_2D_box[1]) /2 , (reference_obj_2D_box[2] - reference_obj_2D_box[0]) /2 ]

                    pair_distance_2D = target_2D_center - reference_2D_center
                    dimension_sum_2D = (np.abs(target_2D_dim) + np.abs(reference_2D_dim))

                    # 3D info 
                    pair_distance = target_obj_box[0] - reference_obj_box[0]
                    dimension_sum = np.abs(target_obj_box[1]) + np.abs(reference_obj_box[1])

                    # import pdb; pdb.set_trace()
                    if np.abs(pair_distance_2D[0])/ np.abs(dimension_sum_2D[0]) > 4:
                        continue

                    pair_ratio = target_obj_box[1]/ reference_obj_box[1]
                    is_behind, is_in_front, is_above, is_below, is_left, is_right = False,False ,False ,False, False, False
                    if (np.abs(pair_distance_2D[0]) - dimension_sum_2D[0]) > -20 :
                        
                        # is_left = reference_obj_box[0][0] > target_obj_box[0][0]
                        # is_right = reference_obj_box[0][0] < target_obj_box[0][0]
                        is_left = reference_2D_center[0] > target_2D_center[0]
                        is_right = reference_2D_center[0] < target_2D_center[0]
                        if is_left :
                            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is left of '+ labels['pred_classes_names'][reference_obj_id])
                        elif is_right: 
                            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is right of '+ labels['pred_classes_names'][reference_obj_id])
                    # print(np.abs(pair_distance_2D[0]) - dimension_sum_2D[0])
                    # import pdb; pdb.set_trace()
                    # is_right = reference_obj_box[0][0] > target_obj_box[0][0]
                    # is_above = reference_obj_box[0][1] < target_obj_box[0][1] 
                    # is_above = (target_obj_box[0][1] -reference_obj_box[0][1])- target_obj_box[1][1] - reference_obj_box[1][1] >0
                    # print("np.abs(pair_distance_2D[1])- dimension_sum_2D[1]", np.abs(pair_distance_2D[1]), dimension_sum_2D[1])
                    if (np.abs(pair_distance_2D[1])- dimension_sum_2D[1]) > -20 :
                        if np.abs(pair_distance_2D[0]) - dimension_sum_2D[0]<10 :
                        # is_above = reference_obj_box[0][1] < target_obj_box[0][1] 
                        # is_below = reference_obj_box[0][1] > target_obj_box[0][1] 
                            is_above = reference_2D_center[1] > target_2D_center[1]
                            is_below = reference_2D_center[1] < target_2D_center[1]
                            if is_above :
                                captions.append(labels['pred_classes_names'][target_obj_id]+ ' is above of '+ labels['pred_classes_names'][reference_obj_id])
                            elif is_below: 
                                captions.append(labels['pred_classes_names'][target_obj_id]+ ' is below of '+ labels['pred_classes_names'][reference_obj_id])
                    
                    # is_below = reference_obj_box[0][1] < target_obj_box[0][1]   
                    # is_behind = reference_obj_box[0][2] > target_obj_box[0][2] 
                    # is_behind = (reference_obj_box[0][2] - target_obj_box[0][2] )- target_obj_box[1][2] - reference_obj_box[1][2] >0
                    if (np.abs(pair_distance[2])- dimension_sum[2]) > 0 :   
                        # import pdb; pdb.set_trace()
                        if np.abs(reference_2D_center[0] - target_2D_center[0]) > 2 * np.max([target_2D_dim[0], reference_2D_dim[0]]):
                            continue
                        elif np.abs(reference_2D_center[1] - target_2D_center[1]) > 2 *np.max([target_2D_dim[1] , reference_2D_dim[1]]):
                            continue
                        else:
                            is_behind = reference_obj_box[0][2] > target_obj_box[0][2]+0.02
                            is_in_front = reference_obj_box[0][2] < target_obj_box[0][2] 
                            if is_behind :
                                captions.append(labels['pred_classes_names'][target_obj_id]+ ' is behind of '+ labels['pred_classes_names'][reference_obj_id])
                            elif is_in_front: 
                                captions.append(labels['pred_classes_names'][target_obj_id]+ ' is in front of '+ labels['pred_classes_names'][reference_obj_id])

                    # if np.sum([is_behind, is_in_front, is_above, is_below, is_left, is_right]):
                    if np.sum([is_behind, is_in_front]):

                        # Display the image
                        fig, ax = plt.subplots()
                        ax.imshow(rgb_image)
                        ax.axis('off')  # Turn off axis

                        positions = [
                            (10, -30),   # Caption 1 position (x, y)
                            (10, 30),   # Caption 3 position
                            (10, 90),  # Caption 5 position
                        ]
                        # Add captions to the image
                        for caption, position in zip(captions, positions):
                            ax.text(position[0], position[1], caption, color='white', fontsize=8, backgroundcolor='black')
                        
                        
                        for detic_pred_boxes in [labels['pred_boxes'][target_obj_id],labels['pred_boxes'][reference_obj_id]] :
                            rect = patches.Rectangle((detic_pred_boxes[1], detic_pred_boxes[0]), 
                                                detic_pred_boxes[3] - detic_pred_boxes[1], 
                                                detic_pred_boxes[2] - detic_pred_boxes[0],
                                                linewidth=1, edgecolor="red", facecolor='none')
                            ax.add_patch(rect)
                        # plt.show()

                        plt.savefig('{}/{}_{}_{}_{}_{}.png'.format(output_labels, img_name[:15],target_obj_id, reference_obj_id, labels['pred_classes_names'][target_obj_id] , labels['pred_classes_names'][reference_obj_id]), bbox_inches='tight')
                        plt.clf()

                        row = [img_name[:15], is_behind, is_in_front, is_above, is_below, is_left, is_right, target_obj_id, reference_obj_id ]
                        writer.writerow(row) 


        # generatePcdinCameraCoordinat_per_object(dataPath, 'Home_001_1', rgb_image_path, depth_image_path, output_labels, obj_id) #img_name[:15]+labels['pred_classes_names'][0], labels['pred_masks'][0] )
        # draw_3D_pointcloud(depth_image, cam_inter, labels['pred_masks'][0] , labels['pred_classes_names'][0])
        # labels['num_instances'] 
        # labels['pred_masks'] 
        # labels['pred_boxes'] 
        # labels['pred_classes'] 
        # labels['pred_classes_names']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud from a .ply file")
    parser.add_argument("path_to_labels", type=str, help="Path to the labels")
    parser.add_argument("path_to_depth", type=str, help="Path to the depth images")
    parser.add_argument("path_to_rgb", type=str, help="Path to the rgb images")
    parser.add_argument("data_path",type=str, help="Path to the rgb images")
    parser.add_argument("path_to_output",type=str, help="Path to the rgb images")
    # parser.add_argument("path_to_detic_predictions",type=str, help="Path to the rgb images")

    args = parser.parse_args()


    if not os.path.exists(args.path_to_output):
        os.makedirs(args.path_to_output)
    # import pdb; pdb.set_trace()
    
    # CLASSES = load_and_sort_json('/home/negar/secondssd/AVD_spatial_relation/total_categories_count.json')
    # print(CLASSES)
    # read_labels(args.path_to_labels, args.path_to_depth, args.path_to_rgb)
    generate_SRE(args.path_to_labels, args.path_to_depth, args.path_to_rgb, args.data_path, args.path_to_output)
