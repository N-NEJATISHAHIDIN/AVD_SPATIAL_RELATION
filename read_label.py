# python read_label.py /home/negar/Downloads/AVD_spatial_relation/Home_001_1 ./ActiveVisionDataset/Home_001_1/high_res_depth ./ActiveVisionDataset/Home_001_1/jpg_rgb 
import bz2
import os
import sys
import json
import pickle

import numpy as np
import argparse
from pathlib import Path
import _pickle as cPickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import gzip

cmap = plt.get_cmap('tab20')
box_colors = [cmap(i % 20) for i in range(100)]

# Read the JSON file
with open('./all_labels.json', 'r') as f:
    data_labels = json.load(f)

def compute_encompassing_box(boxes):
    """
    Compute the bounding box that includes all input boxes.

    Parameters:
    boxes: List of arrays or lists, each containing four elements [x_min, y_min, x_max, y_max]
           representing the coordinates of each box.

    Returns:
    encompassing_box: An array of four elements [x_min, y_min, x_max, y_max]
                      representing the coordinates of the encompassing box.
    """
    if not boxes:
        raise ValueError("The list of boxes is empty")

    # Initialize the encompassing box with the first box
    x_min, y_min, x_max, y_max = boxes[0]

    # Iterate through the list of boxes and update the min and max coordinates
    for box in boxes[1:]:
        x_min = min(x_min, box[0])
        y_min = min(y_min, box[1])
        x_max = max(x_max, box[2])
        y_max = max(y_max, box[3])

    encompassing_box = np.array([x_min, y_min, x_max, y_max])
    return encompassing_box

# load lvis categories
f = open(f'datasets/lvis_categories.json', "r")
data = json.loads(f.read())

lvis_dict = {}
for idx in range(len(data)):
    cat_id = data[idx]['id']
    cat_name = data[idx]['name']
    lvis_dict[cat_id] = cat_name

def compute_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    
    iou = intersection / union if union != 0 else 0.0
    return iou

def dfs(matrix, i, j, visited, current_mask):

    # Define directions for horizontal, vertical, and diagonal moves
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    # Initialize bounding box coordinates
    min_row, max_row, min_col, max_col = i, i, j, j
    # Use a stack for DFS
    stack = [(i, j)]
    
    while stack:
        x, y = stack.pop()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and not visited[nx][ny] and matrix[nx][ny] == 1:
                visited[nx][ny] = True
                current_mask[nx][ny] = 1
                stack.append((nx, ny))
                # Update bounding box coordinates
                min_row, max_row = min(min_row, nx), max(max_row, nx)
                min_col, max_col = min(min_col, ny), max(max_col, ny)

    return np.array([min_row, min_col, max_row, max_col])

def find_islands(matrix):

    rows, cols = matrix.shape
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    islands = []

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and not visited[i][j]:
                # Start a new island
                visited[i][j] = True
                current_mask = [[0 for _ in range(cols)] for _ in range(rows)]
                current_mask[i][j] = 1
                bbox = dfs(matrix, i, j, visited, current_mask)
                islands.append((np.asarray(current_mask), bbox))
    return islands

def read_auto_labeling_labels(label_path, depth_path, rgb_path, output_labels, detic_path):
    
    folder_path = Path(label_path)
    file_names = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix != '.jpg']

    dict_total_categories_count = {}

    for img_name in sorted(file_names):

        labels = {}
        # labels['num_instances'] = []
        labels['pred_masks'] = []
        labels['pred_boxes'] = []
        labels['pred_classes'] = []
        labels['pred_classes_names'] = []
        print(img_name)

        depth_image_path = f'{depth_path}/{img_name[:14]}3.png'
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        rgb_image_path = f'{rgb_path}/{img_name[:15]}.jpg'
        rgb_image = cv2.cvtColor(cv2.imread(rgb_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        annotations_path = f'{label_path}/{img_name}'
        annotations_2D = cv2.imread(annotations_path, cv2.IMREAD_UNCHANGED)
        classes = set(annotations_2D.flatten())

        #detic annotations 
        # with bz2.BZ2File(f'{detic_path}/{img_name[:15]}.pbz2', 'rb') as fp:
        #     pred_dict = cPickle.load(fp)
        #     num_instances = pred_dict['num_instances']
        #     detic_pred_boxes = pred_dict['pred_boxes'].astype(np.int32)
        #     scores = pred_dict['scores']
        #     detic_pred_classes = pred_dict['pred_classes']
        #     predicted_masks = pred_dict['pred_masks']

        for class_id in classes:
            if data_labels[str(class_id)] not in dict_total_categories_count.keys():
                dict_total_categories_count[data_labels[str(class_id)]] = 0

            if class_id == 0 :
                continue
            islands = find_islands((annotations_2D==class_id).astype(int))
            
            if islands == []:
                continue
            # if len(islands) == 1:
            #     labels['pred_classes'].append(class_id)
            #     labels['pred_classes_names'].append(data_labels[str(class_id)])
            #     labels['pred_masks'].append(islands[0][0])
            #     labels['pred_boxes'].append(box)
            #     dict_total_categories_count[data_labels[str(class_id)]] +=1 
            #     continue

            # mask_dict = {}
            # for (island, box) in islands:
                
            #     if island.sum() < 1000:
            #         # print ("$$$", data_labels[str(class_id)], island.sum())
            #         continue

            #     # print ("!!!", data_labels[str(class_id)])

            #     mask_iou = np.array([compute_iou(island , predicted_mask) for predicted_mask in predicted_masks])
            #     posible_mask_idx = np.where( mask_iou > 0.05 )
            #     # print((detic_pred_classes[posible_mask_idx],mask_iou[posible_mask_idx], posible_mask_idx))
            #     posible_classes = np.dstack((detic_pred_classes[posible_mask_idx],mask_iou[posible_mask_idx], posible_mask_idx))

            #     for candidate in posible_classes[0]:
            #         if int(candidate[2]) not in mask_dict.keys():
            #             mask_dict[int(candidate[2])] = []
            #         mask_dict[int(candidate[2])].append((island, box, 
            #                                                  mask_iou[int(candidate[2])]))
                    
            # # import pdb; pdb.set_trace()
            # if len(mask_dict.keys()) == 1:
            #     dict_total_categories_count[data_labels[str(class_id)]] += 1
            #     island_final = np.zeros((island.shape))
            #     box_list_final = []
            #     labels['pred_classes'].append(class_id)
            #     labels['pred_classes_names'].append(data_labels[str(class_id)])
            #     for key in mask_dict.keys():
            #         for value in mask_dict[key]:
            #             island_final = np.logical_or(island_final ,value[0])
            #             box_list_final.append(value[1])
            #     labels['pred_masks'].append(island_final)
            #     labels['pred_boxes'].append(compute_encompassing_box(box_list_final))

            # else:
            for (island, box) in islands:
            
                if island.sum() < 1000:
                    # print ("$$$", data_labels[str(class_id)], island.sum())
                    continue
                dict_total_categories_count[data_labels[str(class_id)]] += 1 

                labels['pred_classes'].append(class_id)
                labels['pred_classes_names'].append(data_labels[str(class_id)])
                labels['pred_masks'].append(island.astype(bool))
                labels['pred_boxes'].append(box)


        print("###########################")
        # import pdb; pdb.set_trace()

        print( len(labels['pred_classes']))#dict_total_categories_count)
        labels['num_instances'] = len(labels['pred_classes'])
        # import pdb; pdb.set_trace()

        with gzip.open('{}/{}.pkl.gz'.format(output_labels, img_name[:15]), 'wb') as f:
            pickle.dump(labels, f)

        # with open('{}/{}.pkl'.format(output_labels, img_name[:15]), 'wb') as f:
        #     pickle.dump(labels, f)
    
    with open("total_categories_count.json", 'w') as file:
        json.dump(dict_total_categories_count, file, indent=4) 
    return 


def read_labels(label_path, depth_path, rgb_path):

    folder_path = Path(label_path)
    file_names = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix != '.jpg']

    for img_name in file_names:
        with bz2.BZ2File(f'{folder_path}/{img_name}', 'rb') as fp:
            pred_dict = cPickle.load(fp)
            num_instances = pred_dict['num_instances']
            detic_pred_boxes = pred_dict['pred_boxes'].astype(np.int32)
            scores = pred_dict['scores']
            detic_pred_classes = pred_dict['pred_classes']
        
        depth_image_path = f'{depth_path}/{img_name[:-6]}3.png'
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        rgb_image_path = f'{rgb_path}/{img_name[:-5]}.jpg'
        rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Create the subplot layout
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        # Plot the RGB image in the first subplot
        axs[0].imshow(rgb_image)
        axs[0].set_title('RGB Image')
        # Add rectangle to the first subplot
        for i in range(len(detic_pred_boxes)):
            rect = patches.Rectangle((detic_pred_boxes[i][0], detic_pred_boxes[i][1]), 
                                    detic_pred_boxes[i][2] - detic_pred_boxes[i][0], 
                                    detic_pred_boxes[i][3] - detic_pred_boxes[i][1], 
                                    linewidth=1, edgecolor=box_colors[i], facecolor='none')
            axs[0].add_patch(rect)
            x,y = detic_pred_boxes[i][0], detic_pred_boxes[i][1]
            axs[0].text(x, y ,  '{}'.format(lvis_dict[detic_pred_classes[i]+1]) , fontsize=10, verticalalignment='top', color=box_colors[i])
            print(data[(detic_pred_classes[i])])
            # axs[1, 0].text(detic_pred_boxes[i][0],detic_pred_boxes[i][1] , '{}, {0:.2f}'.format(data[str(detic_pred_classes[i])], scores[i]), 
            #         verticalalignment='top', fontsize=8, color=box_colors[i])
        axs[0].axis('off')
        # Plot the depth image in the second subplot
        axs[1].imshow(depth_image)
        axs[1].set_title('Depth Image')
        axs[1].axis('off')
        # Adjust layout
        plt.tight_layout()
        # Display the plot
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud from a .ply file")
    parser.add_argument("path_to_labels", type=str, help="Path to the labels")
    parser.add_argument("path_to_depth", type=str, help="Path to the depth images")
    parser.add_argument("path_to_rgb", type=str, help="Path to the rgb images")
    parser.add_argument("path_to_output",type=str, help="Path to the rgb images")
    parser.add_argument("path_to_detic_predictions",type=str, help="Path to the rgb images")

    args = parser.parse_args()


    if not os.path.exists(args.path_to_output):
        os.makedirs(args.path_to_output)

   
    # read_labels(args.path_to_labels, args.path_to_depth, args.path_to_rgb)
    read_auto_labeling_labels(args.path_to_labels, args.path_to_depth, args.path_to_rgb, args.path_to_output, args.path_to_detic_predictions)
