import open3d as o3d
from typing import List
from tqdm import tqdm
def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def savepcd(filename, pcd):
    o3d.io.write_point_cloud(filename, pcd)

def visualize_pcds(pcd_list: List) -> None:
    print("""
    ---------------------------------------------------
    Visualizing the input point clouds for debugging...
    press SPACE would change to the next point cloud.
    press B would change to the previouw point cloud. 
    ---------------------------------------------------\n""")
    g_idx = 0
    vis = o3d.visualization.VisualizerWithKeyCallback()
    pbar = tqdm(total = len(pcd_list))

    def show_pointcloud(vis):
        nonlocal g_idx
        pcd = pcd_list[g_idx]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        vis.clear_geometries()
        vis.add_geometry(pcd) # fuck bug, vis.update_geometry(pcd)没有用！
        vis.update_renderer()
        vis.poll_events()

    def key_forward_callback(vis):
        nonlocal g_idx
        g_idx += 1
        if g_idx >= len(pcd_list):
            g_idx = len(pcd_list) - 1
        show_pointcloud(vis)
        # print(f"{g_idx / len(pcd_list):.3f}%", end='\r')
        pbar.update(1)
        return True

    def key_back_callback(vis):
        nonlocal g_idx
        g_idx -= 1
        if g_idx < 0:
            g_idx = 0
        show_pointcloud(vis)
        # print(f"{g_idx / len(pcd_list):.3f}%", end='\r')
        pbar.update(-2)
        pbar.update(1)
        return True

    vis.create_window()
    vis.get_render_option().point_size = 0.2  # set points size

    vis.register_key_callback(ord(' '), key_forward_callback)  # space
    vis.register_key_callback(ord('B'), key_back_callback)  # fuck bug, 字母必须是大写!
    vis.run()

    
