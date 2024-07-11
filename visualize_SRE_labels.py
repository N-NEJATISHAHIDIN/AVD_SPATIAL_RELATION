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
import pandas as pd

sys.path.insert(0, '/home/negar/secondssd/semantic_abs/baseline')

from utils import draw_3D_pointcloud
spatial_relations = {'behind', 'left of', 'right of', 'in front of', 'on top of', 'inside'}
csv_file = 'SRE_annotations_fixed_label_ fixed_depth.csv'

# Define the headers and rows of data
headers = ["image_name", "is_behind", "is_in_front", "is_above", "is_below", "is_left", "is_right", "target_obj_id", "reference_obj_id"]

# var_to_str = {is_behind: "is_behind", is_in_front:"is_in_front" , is_above:"is_above" , is_below:"is_below" , "is_left": , "is_right": }

def generate_SRE(label_path, rgb_path, dataPath, csv_file):


    data = pd.read_csv(csv_file)
    for index in tqdm(range(len(data))):

        img_name, is_behind, is_in_front, is_above, is_below, is_left, is_right, target_obj_id, reference_obj_id  = data.iloc[index]
        with gzip.open('{}/000{}.pkl.gz'.format(label_path, img_name), 'rb') as f:
            labels = pickle.load(f)
        # import pdb; pdb.set_trace()

        rgb_image_path = '{}/000{}.jpg'.format(rgb_path, img_name)
        rgb_image = cv2.cvtColor(cv2.imread(rgb_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
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
        captions = []
        if is_left :
            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is left of '+ labels['pred_classes_names'][reference_obj_id])
        elif is_right: 
            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is right of '+ labels['pred_classes_names'][reference_obj_id])
        if is_above :
            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is above of '+ labels['pred_classes_names'][reference_obj_id])
        elif is_below: 
            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is below of '+ labels['pred_classes_names'][reference_obj_id])
        if is_behind :
            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is behind of '+ labels['pred_classes_names'][reference_obj_id])
        elif is_in_front: 
            captions.append(labels['pred_classes_names'][target_obj_id]+ ' is in front of '+ labels['pred_classes_names'][reference_obj_id])


        for caption, position in zip(captions, positions):
            ax.text(position[0], position[1], caption, color='white', fontsize=8, backgroundcolor='black')
        
        
        for detic_pred_boxes in [labels['pred_boxes'][target_obj_id],labels['pred_boxes'][reference_obj_id]] :
            rect = patches.Rectangle((detic_pred_boxes[1], detic_pred_boxes[0]), 
                                detic_pred_boxes[3] - detic_pred_boxes[1], 
                                detic_pred_boxes[2] - detic_pred_boxes[0],
                                linewidth=1, edgecolor="red", facecolor='none')
            ax.add_patch(rect)

        plt.show()

        # plt.savefig('{}/{}_{}_{}_{}_{}.png'.format(output_labels, img_name[:15],target_obj_id, reference_obj_id, labels['pred_classes_names'][target_obj_id] , labels['pred_classes_names'][reference_obj_id]), bbox_inches='tight')
        # plt.clf()


       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud from a .ply file")
    parser.add_argument("path_to_labels", type=str, help="Path to the labels")
    parser.add_argument("path_to_rgb", type=str, help="Path to the rgb images")
    parser.add_argument("data_path",type=str, help="Path to the rgb images")
    parser.add_argument("path_to_csv_file",type=str, help="Path to the rgb images")
    # parser.add_argument("path_to_detic_predictions",type=str, help="Path to the rgb images")

    args = parser.parse_args()

    # import pdb; pdb.set_trace()
    generate_SRE(args.path_to_labels,  args.path_to_rgb, args.data_path, args.path_to_csv_file)
