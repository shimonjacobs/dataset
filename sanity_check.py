import numpy as np
import open3d as o3d

# Function to read binary file and convert to point cloud
def read_bin_file(filepath):
    with open(filepath, 'rb') as f:
        bin_data = f.read()
    points = np.frombuffer(bin_data, dtype=np.float32).reshape(-1, 3)
    return points

# Path to the binary file
bin_file_path = "training/velodyne/16_44_37_820960take_3.bin"

# Read binary file
points = read_bin_file(bin_file_path)

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])