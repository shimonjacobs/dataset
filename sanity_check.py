import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from scipy.spatial.transform import Rotation as R


# Function to read binary file and convert to point cloud
def read_bin_file(filepath):
    with open(filepath, 'rb') as f:
        bin_data = f.read()
    points = np.frombuffer(bin_data, dtype=np.float32).reshape(-1, 3)
    return points

## function taken straight from api for 3D object detection 
def get_label(label_file):
    assert label_file.exists()
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # [N, 8]: (x y z dx dy dz heading_angle category_id)
    gt_boxes = []
    gt_names = []
    for line in lines:
        line_list = [l for l in line.strip().split(' ') if len(l) !=0]
        # print(line_list)
        gt_boxes.append(line_list[:-1])
        gt_names.append(line_list[-1])
    
    return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

# Path to the binary file
npy_file_path = Path("/home/maxwell/projects/optitrack_obj_det_gen/detector_training/OpenPCDet/data/custom/points/16_44_19_535323_take_3.npy")
label_file = Path("/home/maxwell/projects/optitrack_obj_det_gen/detector_training/OpenPCDet/data/custom/labels/16_44_19_535323_take_3.txt")

# Read npy file
points = np.load(npy_file_path)
points = points[points[:, 0] >= -5]
points = points[points[:, 0] <= 5]
points = points[points[:, 1] >= -5]
points = points[points[:, 1] <= 5]
points = points[points[:, 2] >= -5]
points = points[points[:, 2] <= 5]
# Read Label file 
bbox, names = get_label(label_file)

xs, ys, zs = points.T

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(projection='3d')

ax.scatter(xs, ys, zs, s = 5)

bbox_i = bbox[0]
print(bbox_i)
cx, cy, cz = bbox_i[:3]

yaw = bbox_i[6]
r_mat = R.from_euler("xyz", [0,0,-yaw]).as_matrix()

dims = np.matmul(bbox_i[3:6], r_mat)


dx, dy, dz = dims
print(dx,dy,dz)

corners = np.array([
    [cx - dx/2, cy - dy/2, cz - dz/2],
    [cx - dx/2, cy - dy/2, cz + dz/2],
    [cx - dx/2, cy + dy/2, cz - dz/2],
    [cx - dx/2, cy + dy/2, cz + dz/2],
    [cx + dx/2, cy - dy/2, cz - dz/2],
    [cx + dx/2, cy - dy/2, cz + dz/2],
    [cx + dx/2, cy + dy/2, cz - dz/2],
    [cx + dx/2, cy + dy/2, cz + dz/2]
])



# for i_c in range(corners.shape[0]):
#     corners[i_c] = np.matmul(corners[i_c], r_mat)

cor_x, cor_y, cor_z = corners.T

ax.scatter(cor_x, cor_y, cor_z, s = 20, marker = '^')

print(corners)

plt.savefig("temp_3d.png")

## 2d visual

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot()

ax.scatter(xs, ys)
ax.scatter(cor_x, cor_y, marker='^')

# print(cx, cy, dx, dy, yaw)
# rect = patches.Rectangle(cx, cy, dx, dy, yaw)

plt.show()

plt.savefig("temp.png")

# # Create Open3D point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(np.tile([135/255,31/255,120/255], (points.shape[0], 1)))

# # Visualize point cloud
# o3d.visualization.draw_geometries([pcd])