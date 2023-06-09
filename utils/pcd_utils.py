import open3d as o3d
from typing import List
from tqdm import tqdm
import os
def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def savepcd(filename, pcd):
    o3d.io.write_point_cloud(filename, pcd)

from typing import List
import open3d as o3d
import os

from typing import List
import open3d as o3d
import numpy as np
import os

from typing import List
import open3d as o3d
import numpy as np
import os

def visualize_pcds(pcd_list: List[o3d.geometry.PointCloud], target_pcd: o3d.geometry.PointCloud) -> None:
    print("""
    ---------------------------------------------------
    Visualizing the input point clouds for debugging...
    ---------------------------------------------------\n""")
    
    output_folder = 'output_images/'
    os.makedirs(output_folder, exist_ok=True)
    
    # Set a color for the target point cloud
    target_pcd.paint_uniform_color([0, 1, 0])  # green for the target pcd

    for i, pcd in enumerate(pcd_list):
        if i%10 == 1:
            filename = f"{output_folder}pointcloud_{i}.png"
            
            # Set a color for the current point cloud
            pcd.paint_uniform_color([1, 0, 0])  # red for the first pcd

            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(pcd)
            vis.add_geometry(target_pcd)  # add the target point cloud to every visualization
            vis.update_renderer()
            vis.capture_screen_image(filename, do_render=True)
            vis.destroy_window()
            
            print(f"Saved image: {filename}")



