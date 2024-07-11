# Automatic Labeling of Active Vision Dataset for Spatial Relations

## Overview
This project aims to automatically label the [Active Vision Dataset](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/) for spatial relations using semantic segmentation and ground truth depth information. The semantic segmentation is derived from the [Auto-Labeling paper](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/html/Li_Labeling_Indoor_Scenes_With_Fusion_of_Out-of-the-Box_Perception_Models_WACVW_2024_paper.html) by Yemeng Li. Inspired by the technique presented in the [SpatialVLM](https://arxiv.org/abs/2401.12168) we automaticly labeled spacial relation expresions in the dataset. The goal is to generate accurate spatial relation labels which is preresented in real indoor scenes for robotics applications.

## Data Paths
This experiment is done for The first scene of AVD with more than 15000 images. 

1. **Path to AVD Data:** In order to re-creat the experiment first download the RGB, depth, and camera information from:
    - [Path to AVD dataset](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/get_data.html)

2. **Path to semantics:** In order to do the experiments we need to have the annotations for semantics and classes of objects, here is the semantics for the first home from Auto-Labeling paper:
    - [Instance segmentation annotations](https://drive.google.com/drive/folders/1-xrACG1Gz4_3WLI7GFGuV-orN-WH4HUE?usp=drive_link)

3. **Path to extra info:** This is a json file which convert classes numbers into class names:
    - [class_is_to_names](https://drive.google.com/file/d/1oG3rp7Q7AAwjJU8q2Q8FZy6sDxHaajDG/view?usp=sharing)

4. **Path to extra info:** This is The saved annotations for avd using this repo:
    - [SRE_labels](https://drive.google.com/file/d/1MX9o3LFOKs9GTg_mjTmyqUxyNnUwas5Z/view?usp=sharing)
    - headers = ["image_name", "is_behind", "is_in_front", "is_above", "is_below", "is_left", "is_right", "target_obj_id", "reference_obj_id"]
    - to visualize the annotations run the following command:
        ```sh
        python visualize_SRE_labels.py path_to_semantic_annotations  path_to_rgb path_to_data_root path_to_SRE_csv_file 
        ```
<!-- - [Available Depth Estimation Models](#available-depth-estimation-models) -->

<!-- ## Data Paths
This project consists of three main data parts:
1. **Path to Main Data 1:** [Define what this data represents]
    ```plaintext
    /path/to/main/data1
    ```
2. **Path to Main Data 2:** [Define what this data represents]
    ```plaintext
    /path/to/main/data2
    ```
3. **Path to Main Data 3:** [Define what this data represents]
    ```plaintext
    /path/to/main/data3
    ``` -->

## Converting Semantics into Pickle Files

The gt semantic labels are coming in the form of png images. To convert them into pkl annotation dict you can run : 
```sh
python read_label.py path_to_labels path_to_depth path_to_rgb path_to_output 
```

Now that you have the required annotations you can run the following command and This program generates the annotations for spatial relation expressions and save the annotations as SRE_annotations.csv file and image with captions to path_to_output.  
```sh
python generat_SRE.py path_to_previous_command_output path_to_depth path_to_rgb path_to_data_root path_to_output 
```
