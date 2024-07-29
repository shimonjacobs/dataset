"""
In this version, we will keep it in lidar coordinate system, but convert to 03d just for visualization
"""

import pandas as pd
import open3d as o3d
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R

def convert_orientation(vector, source, target):
    """
    Convert orientation vector between different standards.

    :param vector: Orientation vector as a list or numpy array [x, y, z]
    :param source: Source orientation standard ('FLU', 'NUE', 'RUB')
    :param target: Target orientation standard ('FLU', 'NUE', 'RUB')
    :return: Converted orientation vector as a numpy array
    """
    # Define transformation matrices for each standard
    transformations = {
        'RUB': np.array([[1, 0, 0], #o3d
                         [0, 1, 0],
                         [0, 0, 1]]),
        'NUE': np.array([[0, 0, -1], #opti 
                         [0, 1, 0],
                         [1, 0, 0]]),
        'FLU': np.array([[0, 0, -1], #lidar
                         [-1, 0, 0],
                         [0, 1, 0]])
    }


    # Ensure the source and target are valid
    if source not in transformations or target not in transformations:
        raise ValueError("Invalid source or target orientation standard")

    # Get the transformation matrices
    source_matrix = transformations[source]
    target_matrix = transformations[target]

    # Compute the transformation matrix from source to target
    transform_matrix = np.linalg.inv(target_matrix) @ np.linalg.inv(source_matrix)

    # Convert the orientation vector
    vector = np.array(vector)
    converted_vector = transform_matrix @ vector

    return converted_vector

def quart_to_mat(q):
    q = np.array(q)
    q = q / np.linalg.norm(q)
    x, y, z, w = q[0], q[1], q[2], q[3]
    #a=w b=x c=y d=z
    return np.array([[1-2*(y**2+z**2),2*(x*y-w*z),2*(x*z+w*y)],\
                    [2*(x*y+w*z),1-2*(x**2+z**2),2*(y*z-w*x)],\
                    [2*(x*z-w*y),2*(y*z+w*x),1-2*(x**2+y**2)]])

def create_point_cloud(points, color):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    return point_cloud

def get_markers_from_tracking(file_name):
    human_description = pd.read_csv(file_name, skiprows=list(range(6))).dropna()
    human_description.drop(human_description.columns[list(range(2))], axis=1, inplace=True)
    human_description = human_description.mean()

    human_piv_rot = human_description[["X", "Y", "Z", "W"]].to_numpy()

    human_piv_rot = R.from_quat(human_piv_rot).as_matrix()

    human_piv_loc = human_description[["X.1", "Y.1", "Z.1"]].to_numpy()


    marker_count = int(list(human_description.keys())[-1].split(".")[-1])
    human_markers = np.stack((np.identity(4, dtype=human_piv_rot.dtype), ) * (marker_count - 1))

    for idx, marker_i in enumerate(range(2, marker_count + 1)):
        marker_i_loc = human_description[["X.%i" % marker_i, "Y.%i" % marker_i, "Z.%i" % marker_i]].to_numpy()
        human_markers[idx, :3, 3] = marker_i_loc - human_piv_loc

    return human_markers, marker_count

def create_point_cloud(points, color):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    return point_cloud

def dict2transMat(in_dict):
    position = np.array([in_dict['loc_x'], in_dict['loc_y'], in_dict['loc_z']], dtype=np.float64)
    orientation = np.array([in_dict['rot_x'], in_dict['rot_y'], in_dict['rot_z'], in_dict['rot_w']], dtype=np.float64) #scipy uses xyzw

    rotation = R.from_quat(orientation).as_matrix()
    transMat = np.identity(4, dtype=rotation.dtype)
    transMat[:3, :3] = rotation
    transMat[:3, 3] = position
    return transMat

def row2transMat(row):
    human_data = {key: row[value] for key, value in zip(["loc_x", "loc_y", "loc_z", "rot_w", "rot_x", "rot_y", "rot_z"], ["H_x", "H_y", "H_z", "H_qw", "H_qx", "H_qy", "H_qz"])}
    # print(human_data)
    human_transformation = dict2transMat(human_data)
    
    lidar_data = {key: row[value] for key, value in zip(["loc_x", "loc_y", "loc_z", "rot_w", "rot_x", "rot_y", "rot_z"], ["S_x", "S_y", "S_z", "S_qw", "S_qx", "S_qy", "S_qz"])}
    lidar_transformation = dict2transMat(lidar_data)
    
    return human_transformation, lidar_transformation

def transMat2Dict(transMat):
    rotMat = transMat[:3, :3]
    orientation = R.from_matrix(rotMat).as_quat()

    xyz = transMat[:3, 3]
    out_dict = {key: value for key, value in zip(["loc_x", "loc_y", "loc_z"], xyz)}
    for key, value in zip(["rot_x", "rot_y", "rot_z", "rot_w"], orientation): #scipy uses xyzw
        out_dict[key] = value
    return out_dict

def draw_bounding_box(points: np.ndarray) -> o3d.geometry.OrientedBoundingBox:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    obb = point_cloud.get_oriented_bounding_box()
    aabb = pcd.get_axis_aligned_bounding_box()
    obb.color = (1, 0, 0)  # Red color
    
    return obb

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def draw_bounding_box(points: np.ndarray) -> o3d.geometry.OrientedBoundingBox:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    obb = point_cloud.get_oriented_bounding_box()
    aabb = pcd.get_axis_aligned_bounding_box()
    obb.color = (1, 0, 0)  # Red color
    
    return obb

def read_bounding_box_from_file(file_path):
    """
    Read bounding box information from a txt file and return an OrientedBoundingBox.
    """
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        parts = line.split()

        # Extract the relevant values
        object_type = parts[0]
        height = float(parts[8])
        width = float(parts[9])
        length = float(parts[10])
        x = float(parts[11])
        y = float(parts[12])
        z = float(parts[13])
        rotation_y = float(parts[14])
        print(rotation_y)
        # Create the bounding box
        center = [x, y, z]
        extent = [width, height, length]

        # Create the rotation matrix from rotation_y
        rotation = R.from_euler('Y', rotation_y, degrees=False).as_matrix()
        # rotation = np.eye(3)
        # rotation = np.linalg.inv(rotation)  # Invert the rotation matrix
        

        # Create and return the OrientedBoundingBox
        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        return obb
    
def write_to_file(row, set_name, data) :
    h, w, l, x, y, z, r = data
    # x += 0.1 # Decrease length by 10 cm for better results
    line = f"human 0 0 0 0 0 0 0 {h} {l} {w} {x} {y} {z} {r}"
    label_name = row["point_cloud_fn"].replace('.pcd', '.txt')
    label_folder = os.path.join(set_name, 'labels')
    label_file = os.path.join(label_folder, label_name)
    
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    with open(label_file, 'w') as file:
        file.write(line)

def extract_bounding_box_info(bbox):
    """
    Extract bounding box information: height, width, length, x, y, z, Rotation_y
    """
    if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
        # For AABB
        center = bbox.get_center()
        extent = bbox.extent
        rotation_y = 0.0  # No rotation needed for AABB

        height, width, length = extent
        x, y, z = center

    elif isinstance(bbox, o3d.geometry.OrientedBoundingBox):
        # For OBB
        center = bbox.get_center()
        extent = bbox.extent

        # Rotation matrix to euler angles (rotation around y-axis)
        rotation_matrix = bbox.R.copy()
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=False)
        print(euler_angles)
        rotation_y = euler_angles[0]  # Yaw angle
        height, width, length = np.asarray(extent) 
        length = length * 2
        x, y, z = center

    else:
        raise TypeError("Bounding box type not recognized. Must be AxisAlignedBoundingBox or OrientedBoundingBox.")

    return height, width, length, x, y, z, rotation_y

# load in the human markers
human_markers, marker_count = get_markers_from_tracking("human_description.csv")

human_markers_in_o3d = np.array([convert_orientation(marker, 'NUE', 'RUB') for marker in human_markers[:, :3, 3]])

#load in the lidar data
set_name = "take_2"
lidar_folder = os.path.join(set_name, 'point_clouds')
csv_file = f"{set_name}_filtered.csv"
data = pd.read_csv(csv_file).dropna()
data_out = pd.DataFrame(columns=["timestamp", "point_cloud_fn", "img_fn", "loc_x", "loc_y", "loc_z", "rot_x", "rot_y", "rot_z", "rot_w"])

# initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# data_out contains the relative positions of the human markers with respect to the lidar
for row in data.to_dict(orient="records"):
    human_transformation, lidar_transformation = row2transMat(row)
    human_rel_posMat = np.matmul(np.linalg.inv(lidar_transformation), human_transformation)
    
    relDict = transMat2Dict(human_rel_posMat)
    relDict["timestamp"] = row['Time Elapsed']
    relDict["img_fn"] = row['Closest Image']
    relDict["point_cloud_fn"] = row['Closest Image'].replace('.png', '.pcd')

    rel_df = pd.DataFrame([relDict])
    data_out = pd.concat([data_out, rel_df], ignore_index=True)

geometries = []
# main loop
for idx, row in enumerate(data_out.to_dict(orient="records")):
    
    # pause at frame (comment out to run continuously)
    if idx not in [500]: #use this to run only specific frames (comment out this line and next line to run all)
        continue
    if idx != 0:
    # remove all the geometry objects from the previous frame
        for item in geometries:
            vis.remove_geometry(item)

    # create the lidar point cloud
    lidar_file = os.path.join(lidar_folder, row["point_cloud_fn"])

    human_TMat = dict2transMat(row) #in optitrack frame (NUE)
    human_loc = convert_orientation(human_TMat[:3, 3], 'NUE', 'RUB')
    
    human_rot = human_TMat[:3, :3]
    human_rot = np.array([convert_orientation(rot, 'NUE', 'RUB') for rot in human_rot])


    #visualize in o3d 
    # lidar point cloud
    pcd = o3d.io.read_point_cloud(lidar_file)
    pcd.points = o3d.utility.Vector3dVector([convert_orientation(point, 'FLU', 'RUB') for point in np.asarray(pcd.points)])
    #multiply the x and z by -1
    pcd.points = o3d.utility.Vector3dVector([[-1 * point[0], point[1], -1 * point[2]] for point in np.asarray(pcd.points)])
    pcd.paint_uniform_color([0, 0, 1]) ## blue
    vis.add_geometry(pcd)

    #origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis.add_geometry(origin)
    
    # human markers (template)
    human_pc_red = create_point_cloud(human_markers_in_o3d, [1, 0, 0]) # point cloud of human template; color red
    

  
    # human markers (actual)
    human_markers_loc = human_markers_in_o3d + human_loc

    #multiply the x and z by -1
    # human_markers_loc[:, 0] = -1 * human_markers_loc[:, 0]
    # human_markers_loc[:, 2] = -1 * human_markers_loc[:, 2]

    #rotate point cloud by 180 degrees around the third point
    # human_markers_loc = np.matmul(human_markers_loc, rotation_matrix_from_vectors(np.array([0, 0, 1]), np.array([0, 0, -1])))
    
    human_pc_green = create_point_cloud(human_markers_loc, [0, 1, 0]) # point cloud of human template; color green
    human_pc_green.rotate(human_rot, center=human_pc_green.points[3])
    

  
    # human anchor point (yellow)
    human_anchor = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    human_anchor.translate(human_loc)
    human_anchor.paint_uniform_color([1, 1, 0]) ## yellow
    vis.add_geometry(human_anchor)

    #human coordinate frame
    human_relative_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    human_relative_pose.translate(human_loc)
    human_relative_pose.rotate(human_rot, center=human_loc)
    vis.add_geometry(human_relative_pose)

    
      #align human point cloud with axis
    v1 = np.array([-1, 0,0]) # negative x axis
    v2 = human_pc_red.points[0] # across human shoulders
    r = rotation_matrix_from_vectors(v2, v1)
    human_pc_red.rotate(r, center=human_pc_red.points[3])
    human_pc_green.rotate(r, center=human_pc_green.points[3])
    vis.add_geometry(human_pc_green)
    vis.add_geometry(human_pc_red)
    
    # human bounding box
    human_bbox = draw_bounding_box(human_pc_green.points)
    vis.add_geometry(human_bbox)
    geometries.extend([pcd, human_pc_green, human_bbox, human_anchor])

    # read bounding box from file
    bb_data = (extract_bounding_box_info(human_bbox))
    write_to_file(row, set_name, bb_data)

    bb2 = read_bounding_box_from_file(os.path.join(set_name, 'labels', row["point_cloud_fn"].replace('.pcd', '.txt')))
    bb2.color = (0, 1, 0)  # Green color
    # print(bb2.get_center())
    # bb2.rotate(human_rot, center=bb2.get_center())
    vis.add_geometry(bb2)
    

    #crop for visualization purposes
    position = human_loc
    # print(position)
    pcd_points = np.asarray(pcd.points)
    
    size = 1.5 #2
    pcd_points = pcd_points[np.where(pcd_points[:, 0] > position[0]-size)]
    pcd_points = pcd_points[np.where(pcd_points[:, 0] < position[0]+size)]
    pcd_points = pcd_points[np.where(pcd_points[:, 2] > position[1]-size)]
    pcd_points = pcd_points[np.where(pcd_points[:, 2] < position[1]+size)]
   
    # pcd.points = o3d.utility.Vector3dVector(pcd_points) # uncomment this line to crop into subject
  
    pcd.paint_uniform_color([0, 0, 1]) ## blue 
   

    vis.poll_events()
    vis.update_renderer()

    time.sleep(1)

vis.run()
vis.destroy_window()
