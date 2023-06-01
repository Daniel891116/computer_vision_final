import open3d as o3d

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def savepcd(filename, pcd):
    o3d.io.write_point_cloud(filename, pcd)