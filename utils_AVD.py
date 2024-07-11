import numpy as np
import os.path
import os
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getCamera(datapath):

    # Get camera information for AVD scene
    if os.path.isfile(datapath + "/cameras.txt"):
        f = open(datapath + "/cameras.txt")
    else :
        return "ERROR: NO CAMERA INFO AVALIBLE"
    data = f.readlines()
    tok = data[-1].split(" ")
    intr = []

    for i in range(4,len(tok)):
        intr.append(float(tok[i]))

    return np.asarray(intr, dtype=np.float32) # fx, fy, cx, cy, distortion params


def generatePcdinCameraCoordinat(dataPath, scene, imgPath, depthpath, outFolder, labels): #name, target_obj_mask ):

    intr = getCamera(dataPath)
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_params.set_intrinsics(1920, 1080, intr[0], intr[1], intr[2], intr[3])

    color_img = o3d.io.read_image(imgPath)
    depth_img = o3d.io.read_image(depthpath)

    # visualize the whole image 
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic_params)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])


    # generate masked depth of all objs 
    for obj_id in range(labels['num_instances']):
        numpy_depth = o3d.geometry.Image(np.array(labels['pred_masks'][obj_id].astype(np.float32) * np.array(depth_img)))

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, numpy_depth, convert_rgb_to_intensity=False)
        pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic_params)
        if  len(np.asarray(pcd2.points)) > 500 :
            pcd2.colors = o3d.utility.Vector3dVector( np.repeat(np.array([[1,0,0]], dtype=np.float32), len(np.asarray(pcd2.points)), axis=0))
        # import pdb; pdb.set_trace()
        pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if  len(np.asarray(pcd2.points)) > 50 :
            o3d.visualization.draw_geometries([pcd, pcd2], window_name= labels['pred_classes_names'][obj_id])

            # obb = pcd2.get_oriented_bounding_box()
            # obb_center = obb.center
            # obb_extent = obb.extent
            # obb_rotation = obb.R
            # obb_center, obb_extent, obb_rotation

        else:
            print( labels['pred_classes_names'][obj_id] , "Not enough points!")
            continue

# def cluster_and_get_largest(point_cloud, eps=0.05, min_points=10):
#     # Apply DBSCAN clustering
#     labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
#     max_label = labels.max()

#     if max_label < 0:
#         print("No clusters found!")
#         return None

#     # Select the largest cluster
#     largest_cluster_indices = np.where(labels == np.bincount(labels[labels >= 0]).argmax())[0]
#     largest_cluster = point_cloud.select_by_index(largest_cluster_indices)
#     return largest_cluster

def cluster_and_get_largest(point_cloud, eps=0.05, min_points=10):
    # Apply DBSCAN clustering
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()

    if max_label < 0:
        print("No clusters found!")
        return None, None

    # Get the size of each cluster
    cluster_sizes = np.bincount(labels[labels >= 0])

    # Sort the clusters by size
    sorted_clusters = np.argsort(cluster_sizes)[::-1]

    # If there is only one cluster, return it
    if len(sorted_clusters) == 1:
        largest_cluster_indices = np.where(labels == sorted_clusters[0])[0]
        largest_cluster = point_cloud.select_by_index(largest_cluster_indices)
        return largest_cluster

    # If the largest cluster is significantly larger than the next one, return it
    if cluster_sizes[sorted_clusters[0]] > cluster_sizes[sorted_clusters[1]] * 2:
        largest_cluster_indices = np.where(labels == sorted_clusters[0])[0]
        largest_cluster = point_cloud.select_by_index(largest_cluster_indices)
        return largest_cluster
    else:
        print("The largest cluster is not significantly larger than the next one. Skipping.")
        return point_cloud
from scipy.spatial.transform import Rotation as R
  
def set_pitch_to_zero(RM):
    """
    Sets the pitch (elevation) component of the given rotation matrix to zero while retaining the azimuth (yaw) and roll.

    Parameters:
        R (numpy.ndarray): The original 3x3 rotation matrix.

    Returns:
        numpy.ndarray: The new 3x3 rotation matrix with the pitch set to zero.
    """
    # Ensure the input is a valid 3x3 rotation matrix
    assert RM.shape == (3, 3), "The input rotation matrix must be a 3x3 matrix."

    # Decompose the rotation matrix into roll, pitch, and yaw angles
    # import pdb; pdb.set_trace()
    RM = np.copy(RM)
    r = R.from_matrix(RM)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    # Set the pitch angle to zero
    # roll = 0
    # yaw = 0
    # Construct the new rotation matrix with the adjusted angles
    new_r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    new_R = new_r.as_matrix()

    return new_R
def compute_azimuth_angle(aabb):
    # Project the AABB onto the xy-plane
    aabb_proj_xy = aabb.rotate(aabb.get_rotation_matrix_from_xyz((np.pi/2, 0, 0)), center=aabb.get_center())
    # Get the min and max points of the projected AABB
    min_point, max_point = np.asarray(aabb_proj_xy.get_min_bound()), np.asarray(aabb_proj_xy.get_max_bound())
    # Compute the azimuth angle (rotation around the z-axis) of the AABB in the xy-plane
    azimuth_angle = np.arctan2(max_point[1] - min_point[1], max_point[0] - min_point[0])
    return azimuth_angle

def rotate_aabb(aabb, azimuth_angle):
    # Rotate the AABB back to its original orientation using the computed azimuth angle
    aabb_rotated = aabb.rotate([0, 0, azimuth_angle], center=aabb.get_center())
    return aabb_rotated

def random_downsample_point_cloud(point_cloud, sample_ratio):
    # Convert Open3D point cloud to numpy array
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    
    # Calculate the number of points to sample
    num_points = points.shape[0]
    num_sampled_points = int(num_points * sample_ratio)
    
    # Generate random indices to sample
    sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
    
    # Create a new downsampled point cloud
    downsampled_points = points[sampled_indices, :]
    downsampled_colors = colors[sampled_indices, :]

    downsampled_point_cloud = o3d.geometry.PointCloud()
    downsampled_point_cloud.points = o3d.utility.Vector3dVector(downsampled_points)
    downsampled_point_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors)

    
    return downsampled_point_cloud
from PIL import Image

def generatePcdinCameraCoordinat_per_object(dataPath, scene, imgPath, depthpath, outFolder, labels, obj_id): #name, target_obj_mask ):

    intr = getCamera(dataPath)
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_params.set_intrinsics(1920, 1080,intr[0], intr[1], intr[2], intr[3])

    color_img = o3d.io.read_image(imgPath)

    if depthpath[-4:]==".png":
        # image = Image.open(depthpath)
        # plt.imshow(image)
        # plt.show()
        # depth_img = o3d.geometry.Image(image)

        depth_img = o3d.io.read_image(depthpath)
    elif depthpath[-4:]==".npy":
        depth_array = np.load(depthpath)
        # depth = 255 - ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0)
        # depth = depth.astype(np.uint8)
        # plt.imshow(depth_array, cmap='plasma')
        # plt.show()
        # plt.savefig("jana/{}.png".format(os.path.basename(depthpath)[:-4]))
        # import pdb; pdb.set_trace()
        depth_img = o3d.geometry.Image(depth_array)
    else:
        import pdb; pdb.set_trace()
        print("NOT IMPLIMENTED ERROR")

    # visualize the whole image 
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img,1000, 10, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic_params)
    pcd = random_downsample_point_cloud(pcd, 0.1)
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud("./samples/{}.ply".format(os.path.basename(depthpath)[:-4]), pcd)        



    # generate masked depth of all objs 
    # for obj_id in range(labels['num_instances']):
    numpy_depth = o3d.geometry.Image(np.array(labels['pred_masks'][obj_id].astype(np.float32) * np.array(depth_img)))

    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, numpy_depth, 1000, 10, convert_rgb_to_intensity=False)
    pcd2_noise = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic_params)
    pcd2_noise = random_downsample_point_cloud(pcd2_noise, 0.1)

    cl, ind  = pcd2_noise.remove_statistical_outlier(nb_neighbors=3000, std_ratio=2.0)
    pcd2 = pcd2_noise.select_by_index(ind)
    if  len(np.asarray(pcd2.points)) < 500 :
        return None
    # pcd2 = cluster_and_get_largest(pcd2)
    if  len(np.asarray(pcd2.points)) > 500 :
        pcd2.colors = o3d.utility.Vector3dVector( np.repeat(np.array([[1,0,0]], dtype=np.float32), len(np.asarray(pcd2.points)), axis=0))
    # import pdb; pdb.set_trace()
    pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if  len(np.asarray(pcd2.points)) > 500 :
        # o3d.visualization.draw_geometries([obb, pcd2], window_name= labels['pred_classes_names'][obj_id])

        obb = pcd2.get_axis_aligned_bounding_box()
        # obb = pcd2.get_oriented_bounding_box()
        # obb_new.center = obb.get_center()
        # obb_new.extent = obb.get_extent()

        # azimuth_angle = compute_azimuth_angle(obb_new)

        # Step 4: Rotate the AABB back to its original orientation using the computed azimuth angle
        # obb_rotated = rotate_aabb(obb_new, azimuth_angle)
        center = obb.get_center()
        dimensions = obb.get_extent()

        # import pdb; pdb.set_trace()
        # obb_center = obb.center
        # obb_extent = obb.extent
        # new_rotation = set_pitch_to_zero(obb.R)
        # rotation = obb.R 
        # o3d.visualization.draw_geometries([pcd, obb, pcd2], window_name= labels['pred_classes_names'][obj_id])
        # o3d.visualization.draw_geometries([pcd], window_name= labels['pred_classes_names'][obj_id])
        return [center, dimensions] #rotation]

    else:
        print( labels['pred_classes_names'][obj_id] , "Not enough points!")
        return None
        
    # labels['num_instances'] 
    # labels['pred_masks'] 
    # labels['pred_boxes'] 
    # labels['pred_classes'] 
    # labels['pred_classes_names']

    # depth anything to get the depth 

    # from transformers import pipeline
    # from PIL import Image
    # import requests

    # # load pipe
    # pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    # image = Image.open(imgPath)
    # depth = pipe(image)["depth"]
    # depth_img = o3d.geometry.Image(np.array(depth))
    # # import pdb; pdb.set_trace()


    # rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, numpy_depth, convert_rgb_to_intensity=False)
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic_params)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # try: 
    #     o3d.visualization.draw_geometries([pcd])
    # except:
    #     return
    # o3d.io.write_point_cloud("{}/{}.ply".format(outFolder, name), pcd)
    # print(name+".ply is saved.")

