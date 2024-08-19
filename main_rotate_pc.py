"""
In this version, we will keep it in lidar coordinate system, but convert to 03d just for visualization
"""

import pandas as pd
import open3d as o3d
import numpy as np
import os
from pathlib import Path
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

## below are code taken from: https://github.com/google-research-datasets/Objectron/tree/master
from objectron import iou as IoU 
from objectron import box as Box 

from IoU3d import box3d_iou

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

WRITE_FILE = False

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

'''
    points is array type : [N x 3]
    color is rgb for example [0,0,1] is blue
'''
def create_point_cloud(points, color):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    return point_cloud

'''
    modified to return the points as [N x 3] array like object
    also now returns the pivot point's rotation for fixing later  
    also now has a bolean that allows us to use the floor ar the reference for the base of the box
'''
def get_markers_from_tracking(file_name, usefloor = False):
    human_description = pd.read_csv(file_name, skiprows=list(range(6))).dropna()
    human_description.drop(human_description.columns[list(range(2))], axis=1, inplace=True)
    human_description = human_description.mean()

    human_piv_rot = human_description[["X", "Y", "Z", "W"]].to_numpy()

    human_piv_rot = R.from_quat(human_piv_rot).as_matrix()

    human_piv_loc = human_description[["X.1", "Y.1", "Z.1"]].to_numpy()

    marker_count = int(list(human_description.keys())[-1].split(".")[-1])
    human_markers = np.zeros(((marker_count), 3), dtype=human_piv_loc.dtype) if usefloor else np.zeros(((marker_count - 1), 3), dtype=human_piv_loc.dtype)

    for idx, marker_i in enumerate(range(2, marker_count + 1)):
        marker_i_loc = human_description[["X.%i" % marker_i, "Y.%i" % marker_i, "Z.%i" % marker_i]].to_numpy()
        
        human_markers[idx] = marker_i_loc - human_piv_loc ## align all to (0,0,0)

    ## add an virtual marker below the pivot point at the floor height 
    # print(human_markers.shape)
    if usefloor:
        marker_count += 1
        human_markers[-1] = np.array([0, -1*human_piv_loc[1], 0], dtype=human_markers.dtype)

    return human_markers, marker_count, human_piv_rot

def create_point_cloud(points, color):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    return point_cloud

def dict2transMat(in_dict):
    position = np.array([in_dict['loc_x'], in_dict['loc_y'], in_dict['loc_z']], dtype=np.float32)
    orientation = np.array([in_dict['rot_x'], in_dict['rot_y'], in_dict['rot_z'], in_dict['rot_w']], dtype=np.float32) #scipy uses xyzw

    rotation = R.from_quat(orientation).as_matrix()
    transMat = np.identity(4, dtype=rotation.dtype)
    transMat[:3, :3] = rotation
    transMat[:3, 3] = position
    return transMat

def row2transMat(row):
    
    if "H_x" in row:
        human_data = {key: row[value] for key, value in zip(["loc_x", "loc_y", "loc_z", "rot_w", "rot_x", "rot_y", "rot_z"], ["H_x", "H_y", "H_z", "H_qw", "H_qx", "H_qy", "H_qz"])}
        human_transformation = dict2transMat(human_data)
    else: 
        human_transformation = None

    if "B_x" in row:
        box_data = {key: row[value] for key, value in zip(["loc_x", "loc_y", "loc_z", "rot_w", "rot_x", "rot_y", "rot_z"], ["B_x", "B_y", "B_z", "B_qw", "B_qx", "B_qy", "B_qz"])}
        box_transformation = dict2transMat(box_data)   
    else: 
        box_transformation = None

    ## for the processing there should alays be lidar tracking 
    lidar_data = {key: row[value] for key, value in zip(["loc_x", "loc_y", "loc_z", "rot_w", "rot_x", "rot_y", "rot_z"], ["S_x", "S_y", "S_z", "S_qw", "S_qx", "S_qy", "S_qz"])}
    lidar_transformation = dict2transMat(lidar_data)
    
    return human_transformation, box_transformation, lidar_transformation

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
    # aabb = pcd.get_axis_aligned_bounding_box()
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
        # print(rotation_y)
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
    
def write_to_file(row, set_name, data, data_box):
    h, w, l, x, y, z, r = data
    h2, w2, l2, x2, y2, z2, r2 = data_box
    # x += 0.1 # Decrease length by 10 cm for better results
    line = f"human 0 0 0 0 0 0 0 {h} {l} {w} {x} {y} {z} {r} \n"
    line2 = f"box 0 0 0 0 0 0 0 {h2} {l2} {w2} {x2} {y2} {z2} {r2} \n"
    label_name = row["point_cloud_fn"].replace('.pcd', '.txt')
    label_folder = os.path.join(set_name, 'labels')
    label_file = os.path.join(label_folder, label_name)
    
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    with open(label_file, 'w') as file:
        file.write(line)
        file.write(line2)
    ## Daniel's Writing to debug
    print(f"Written to file: {label_file}")


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
        # print(euler_angles)
        rotation_y = euler_angles[0]  # Yaw angle
        height, width, length = np.asarray(extent) 
        length = length * 2
        x, y, z = center

    else:
        raise TypeError("Bounding box type not recognized. Must be AxisAlignedBoundingBox or OrientedBoundingBox.")

    return height, width, length, x, y, z, rotation_y


"""
    Caluclates dimensions of a box from points

    args:
            points : np.ndarray containing the points represnting the box - or optitrack markers
            scale : float between multiplier to increase or decrease dimensions
            ordering : str order of the dimension in decreasing legth - relative to the direction the object is faceing.
                       The front face is WxH, side dim is depth, width is the dimensions accross - default is for the our human object 
                       this can also be set to false or none to just return the dimensions as a numpy array 

    return:
            dict with the bounding box dims 
"""
def box_from_points(points : np.ndarray, scale : float = 1, ordering = "HWD"):

    w_max, l_max, d_max = points.max(axis=0)
    w_min, l_min, d_min = points.min(axis=0)

    dims = np.array([(w_max - w_min) * scale, (l_max - l_min) * scale, (d_max - d_min) * scale])
    dim_order = dims.argsort()[::-1]
    if ordering:
        return {o : d for o, d in zip(ordering, dims[dim_order])}
    else: 
        return dims

def reorientation_matrix(open3dBox, ordering = "HWD"):
    # print(type(open3dBox))

    ## make a copy of the box locally for calulation 
    open3dBox_local = o3d.geometry.OrientedBoundingBox(open3dBox)
    ## align with lidar pose 
    open3dBox_local.rotate(np.linalg.inv(np.asarray(open3dBox.R)), center = open3dBox.get_center())
    # open3dBox_local.rotate(r_o2l, center = open3dBox.get_center())

    dims = box_from_points(np.asarray(open3dBox_local.get_box_points()),1,False)

    ## work out current orientation 
    print("T:", ordering)
    current_ordering = ''.join("DHW"[i] for i in dims.argsort()[::-1])

    print("in new func",dims)
    print(dims.argsort()[::-1])
    print("T:", ordering, "C:", current_ordering)


    return np.linalg.inv(np.asarray(open3dBox.R)), open3dBox_local

def draw_centre(vis, geometries, rot = None):
    #origin - relative to lidar
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.7, origin=[0, 0, 0])
    origin.rotate(rot, center = (0,0,0))
    geometries.append(origin)
    vis.add_geometry(origin)

    ## paint 3 points just to see relative to open3d
    ## point on x axes 
    x_sphere = o3d.geometry.TriangleMesh.create_sphere(.1)
    x_sphere.translate([1,0,0])
    x_sphere.paint_uniform_color([1, 0, 0]) ## red
    geometries.append(x_sphere)
    vis.add_geometry(x_sphere)

    ## point on y axes 
    y_sphere = o3d.geometry.TriangleMesh.create_sphere(.1)
    y_sphere.translate([0,1,0])
    y_sphere.paint_uniform_color([0, 1, 0]) ## green
    geometries.append(y_sphere)
    vis.add_geometry(y_sphere)

    ## point on z axis 
    z_sphere = o3d.geometry.TriangleMesh.create_sphere(.1)
    z_sphere.translate([0,0,1])
    z_sphere.paint_uniform_color([0, 0, 1]) ## blue
    geometries.append(z_sphere)
    vis.add_geometry(z_sphere)

    return geometries

# load in the human markers
human_markers, marker_count, human_pivot_rot = get_markers_from_tracking(Path(os.path.realpath(__file__)).parent / "human_description.csv")


# box_markers = get_box_descriptions("box_description.csv")
box_markers, box_marker_count, box_pivot_rot = get_markers_from_tracking(Path(os.path.realpath(__file__)).parent / "box_description.csv", usefloor=True)


## Daniel's Task 2 Functions
def convert_pcd_to_bin(pcd):
    points = np.asarray(pcd.points)
    bin_points = points.astype(np.float32).tobytes()
    return bin_points

## Daniel's Task 2 Functions
def save_bin_file(bin_data, filepath):
    with open(filepath, 'wb') as f:
        f.write(bin_data)


#load in the lidar data and define output path 
base_path = Path("/media/maxwell/DUMPSTERFIR1/sapience")
set_name = "take_1"
lidar_folder = base_path / os.path.join(set_name, 'point_clouds')
csv_file = base_path / f"optitrack_processedData/{set_name}_filtered.csv"
data = pd.read_csv(csv_file, float_precision='round_trip').dropna()

### Daniel Task 1: modify this to create the correct filestructure 
# output_path = Path(os.getcwd()) / "output_files"
# if not output_path.exists():
#     os.mkdir(output_path)

## Step 1: Going to initialize the output path to "training" 
# output_path = Path(os.getcwd()) / "training"
output_path = Path("/media/maxwell/DUMPSTERFIR1/sapience/generated_data/train")
label_path = output_path / "label_2"
velodyne_path = output_path / "velodyne"
if not output_path.exists():
    output_path.mkdir(parents=True)
if not label_path.exists():
    os.mkdir(label_path)
if not velodyne_path.exists():
    os.mkdir(velodyne_path)


data_rel = pd.DataFrame(columns=["timestamp", "point_cloud_fn", "img_fn", "loc_x", "loc_y", "loc_z", "rot_x", "rot_y", "rot_z", "rot_w", "class", "yaw", "L_Rot_Mat"])


## below is the translation matrix from the tracked pivot point of the lidar to the centre of the sensor which is its frame of reference 
##  (33.375, 52.84, 135.875) 
lidar_opti_pv_2_centre = np.array(
    [
        [1, 0, 0,  -33.375e-3],
        [0, 1, 0,  52.840e-3],
        [0, 0, 1, 135.875e-3],
        [0, 0, 0,  1]
    ]
)

# preprocessing step to fill data_rel with object position and and tracking information 
for row in data.to_dict(orient="records"):
    human_transformation, box_transformation, lidar_transformation = row2transMat(row)

    lidar_transformation = np.matmul(lidar_transformation, lidar_opti_pv_2_centre)

    ## if human in dataframe process 
    if human_transformation is not None: 
        ## calculate relative pose 
        human_rel_posMat = np.matmul(np.linalg.inv(lidar_transformation), human_transformation)

        relDict = transMat2Dict(human_rel_posMat)
        relDict["class"] = 0
        relDict["timestamp"] = row['Time Elapsed']
        relDict["img_fn"] = row['Closest Image']
        relDict["point_cloud_fn"] = row['Closest Image'].replace('.png', '.pcd')

    ## if box is in dataframe, process - same steps as the human
    if box_transformation is not None:
        box_rel_posMat = np.matmul(np.linalg.inv(lidar_transformation), box_transformation)

        relDict = transMat2Dict(box_rel_posMat)
        relDict["class"] = 1 ## class is 1 for box 
        relDict["timestamp"] = row['Time Elapsed']
        relDict["img_fn"] = row['Closest Image']
        relDict["point_cloud_fn"] = row['Closest Image'].replace('.png', '.pcd')

    relDict["L_Rot_Mat"] = np.linalg.inv(lidar_transformation[:3, :3])

    rel_df = pd.DataFrame([relDict])
    data_rel= pd.concat([data_rel, rel_df], ignore_index=True)

## buffer to store plotted objects 
geometries = []
# main loop
pc_filenames = data_rel["point_cloud_fn"].unique()
print("Number of pcd files:",{len(pc_filenames)})

# initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

## new approach to iterate through all filenames 
for idx, pc_filename in enumerate(pc_filenames):

    ## optitrack to lidar rotation: 
    r_o2l = np.array(
        [
            [-1, 0, 0],
            [ 0, 0, 1],
            [ 0, 1, 0]
        ]
    )

    # create the lidar point cloud
    lidar_file = os.path.join(lidar_folder, pc_filename)
    if not os.path.exists(lidar_file): 
        print("Could not find", idx, lidar_file)
        continue
    
    # pause at frame (comment out to run continuously) 680
    if idx <= 150: #use this to run only specific frames (comment out this line and next line to run all)
        continue
    if idx != 0:
    # remove all the geometry objects from the previous frame
        for item in geometries:
            vis.remove_geometry(item)

    ## plot the point cloud 
    # pcd.points = o3d.utility.Vector3dVector([convert_orientation(point, 'FLU', 'RUB') for point in np.asarray(pcd.points)])
    pcd = o3d.io.read_point_cloud(lidar_file)
    pcd.paint_uniform_color([0, 0, 1]) ## blue
    
    geometries.append(pcd)
    vis.add_geometry(pcd)

    ## get the rows associated with this file:
    pc_rows = data_rel.loc[data_rel['point_cloud_fn'] == pc_filename]

    L_Rot_Mat = pc_rows.iloc[0]["L_Rot_Mat"]

    r_o2l = np.matmul(L_Rot_Mat, r_o2l)
    pcd.rotate(L_Rot_Mat, center = (0,0,0))

    geometries = draw_centre(vis, geometries, r_o2l)    
    
    ## lazy write
    if WRITE_FILE:
        output_filepath = label_path / (pc_filename.replace('.pcd', "{}.txt".format(set_name)))
        f = open(output_filepath, "w")

    for _, row in pc_rows.iterrows():
        # print(row)
        cls_no = row["class"]

        ## get trandformation matrix - this is in optritack frame of reference 
        object_TMat = dict2transMat(row) 
        object_rot = object_TMat[:3, :3]
        object_loc = object_TMat[:3,  3]

        ## location of pivot point of object 
        object_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=(0,0,0))
        object_origin.rotate(object_rot, center=(0,0,0))
        object_origin.translate(object_loc)
        object_origin.rotate(r_o2l, center = (0,0,0)) ## rotate to match lidar coordinates 

        geometries.append(object_origin)
        vis.add_geometry(object_origin)
        
        if cls_no == 0:
            cls_name = "human"

            object_pc = create_point_cloud(human_markers,[1, 0, 0])
            object_pc.rotate(np.linalg.inv(human_pivot_rot), center = (0,0,0)) ## correct for the offset when aquiring points
            
            ### grab the points of the box in this perspective - can be used to get the whd 
            object_pc_bounds = object_pc.get_axis_aligned_bounding_box().get_box_points()
            dims_dict = box_from_points(np.asarray(object_pc_bounds), scale=1.)

        elif cls_no == 1:
            cls_name = "box"
            object_pc = create_point_cloud(box_markers,[1, 0.647, 0])
            object_pc.rotate(np.linalg.inv(box_pivot_rot), center = (0,0,0)) ## correct for the offset when aquring points
            
            ### grab the points of the box in this perspective - can be used to get the wld
            object_pc_bounds = object_pc.get_axis_aligned_bounding_box().get_box_points()
            dims_dict = box_from_points(np.asarray(object_pc_bounds), scale=1., ordering="DWH")

        else:
            print("Warning class not recognised")

        ### append these points to the pc for better oriented bbox estimation  
        object_pc.points.extend(object_pc_bounds)
        
        object_pc.rotate(object_rot, center=(0,0,0))
        object_pc.translate(object_loc)
        object_pc.rotate(r_o2l, center = (0,0,0))
        vis.add_geometry(object_pc)
        geometries.append(object_pc)

        ## apply open3d to get box of best fit - this has 6d pose 
        object_bb_6D = object_pc.get_minimal_oriented_bounding_box(robust=True)

        object_bb_6D.color = [135/255,31/255,120/255]
        vis.add_geometry(object_bb_6D)
        geometries.append(object_bb_6D)
        
        ## using 6D pose box fine its centre - this will determine it's location, remember is it in open3d coords
        bb_xyz = object_bb_6D.get_center().tolist()
        
        ## the boxes dims are calculate above 
        # bb_width(x), bb_height(y), bb_depth (z)
        bb_width, bb_height, bb_depth = dims_dict["W"], dims_dict["H"], dims_dict["D"]

        ## create linset that is draw from bottom left corner to top right corner - in remember is it in open3d coords
        points = [
            [-bb_width/2, -bb_height/2, -bb_depth/2],
            [ bb_width/2,  bb_height/2,  bb_depth/2],
        ]
        lines = [
            [0, 1]
        ]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.rotate(object_rot, center=(0,0,0))
        line_set.rotate(r_o2l, center = (0,0,0))
        line_set.translate(bb_xyz)
        vis.add_geometry(line_set)
        geometries.append(line_set)

        Zx, Zy, Zz = object_rot[2]

        object_rot_z = object_rot[2, :].squeeze()
        z_axis = L_Rot_Mat[2, :].squeeze()
        object_yaw = np.arccos(np.clip(np.dot(object_rot_z,z_axis), -1.0, 1.0))

        # print(idx, Zx, Zy, np.rad2deg(object_yaw))
        if Zx < 0 and Zy < 0:
            object_yaw = object_yaw * -1
        elif Zx > 0 and Zy < 0:
            object_yaw = object_yaw
        elif Zx < 0 and Zy > 0:
            object_yaw = object_yaw *-1
        else:
            object_yaw = object_yaw

        ## calculate rotation without yaw - remember this shit is in the optitrack frame so y is up 
        # object_rot_noyaw = np.matmul(np.copy(object_rot),R.from_euler("xyz", [0,object_yaw,0]).as_matrix())

        ## create a new box that is orientated with z forward, y up (rhr)
        object_bb_z_align = o3d.geometry.TriangleMesh.create_box(bb_width, bb_height, bb_depth).get_minimal_oriented_bounding_box()
        object_bb_z_align.color = [1, 0.647, 0]
        object_bb_z_align.translate([-bb_width/2, -bb_height/2, -bb_depth/2]) ## centre on (0,0,0)
        object_bb_z_align.rotate(object_rot, center = (0,0,0))
        object_bb_z_align.rotate(r_o2l, center = (0,0,0))

        ## transformations in open3d
        object_bb_z_align.translate(bb_xyz)
        object_bb_z_align.rotate(R.from_euler("xyz", [0,0, object_yaw]).as_matrix(), center = bb_xyz)

        # # object_bb_z_align.translate(bb_xyz) ## move to the loaction 
        # # visualise object_bb_z_align
        vis.add_geometry(object_bb_z_align)
        geometries.append(object_bb_z_align)

        # ## this is to check if the bounding box used to calculate the final dims is aligned with z
        align_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.3 if cls_no==1 else 0.4, origin=(0,0,0))
        align_origin.rotate(object_rot, center = (0,0,0))
        # align_origin.translate(object_loc)
        align_origin.rotate(r_o2l, center = (0,0,0))
        align_origin.rotate(R.from_euler("xyz", [0,0, object_yaw]).as_matrix(), center = (0,0,0))
        # vis.add_geometry(align_origin)
        # geometries.append(align_origin)

        # ## calculate a box that will enclose object_bb_z_align but is horizontally aligned
        object_pc_bounds = object_bb_z_align.get_axis_aligned_bounding_box().get_box_points()
        if cls_no == 1:
            dims_dict = box_from_points(np.asarray(object_pc_bounds), scale=1)
        else:
            dims_dict = box_from_points(np.asarray(object_pc_bounds), scale=1, ordering="DWH")
        bb_width, bb_height, bb_depth = dims_dict["W"], dims_dict["H"], dims_dict["D"]
        
        # ## create repesentation of the final box for visualisation - NOTE THAT AT THIS POINT WE ARE USING OPEN3D CONVENTION 
        object_bb_out = o3d.geometry.TriangleMesh.create_box(bb_width, bb_height, bb_depth).get_minimal_oriented_bounding_box()
        object_bb_out.color = [0, 1, 0]
        object_bb_out.translate([-bb_width/2, -bb_height/2, -bb_depth/2]) ## centre on (0,0,0)
        object_bb_out.rotate(R.from_euler("xyz", [0,0,-object_yaw]).as_matrix(), center = (0,0,0))
        object_bb_out.translate(bb_xyz)
        vis.add_geometry(object_bb_out)
        geometries.append(object_bb_out)

        ## needs to be in camera coordenate system:
        # x-axis -> right (width), y-axis -> bottom (height), z-axis -> forward (depth)
        # x = x , y = -z, z = y, bb_depth = bb_height, bb_height = bb_depth
        # ## type, truncated, occluded, alpha, 2d_xyxy_bbox, 3d_dims, 3d_loc, 3d_yaw, score
        open3d_bb_x, open3d_bb_y, open3d_bb_z = bb_xyz  
        camera_bb_x, camera_bb_y, camera_bb_z = open3d_bb_x, -open3d_bb_z, open3d_bb_y
        bb_description = [0,0,0,0,0,0,0] + [bb_depth, bb_width, bb_height] + [camera_bb_x, camera_bb_y, camera_bb_z, object_yaw]

        if WRITE_FILE:
            f.write(cls_name + str(bb_description).replace("[", ",").replace("]", "\n"))
      
    
    if WRITE_FILE:
        f.close()
        ## Daniel Task 2: code to convert pcd to binary then save in correct directory 
        bin_data = convert_pcd_to_bin(pcd)
        bin_file_path = velodyne_path / pc_filename.replace('.pcd', "{}.bin".format(set_name))
        save_bin_file(bin_data, bin_file_path)

    vis.poll_events()
    vis.update_renderer()

    time.sleep(.05)
    break
    # if idx >= 200:
    #     break 

vis.run()
vis.destroy_window()
