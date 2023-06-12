import os
from typing import List, Dict
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def savepcd(filename, pcd):
    o3d.io.write_point_cloud(filename, pcd)


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
            
def csv_reader(filename):
    # read csv file into numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data

def ICP(source, target, threshold, init_pose, iteration=30):
    # implement iterative closet point and return transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_pose,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration)
    )
    print(reg_p2p)
    assert len(reg_p2p.correspondence_set) != 0, 'The size of correspondence_set between your point cloud and sub_map should not be zero.'
    print(reg_p2p.transformation)
    return reg_p2p.transformation

def compare_contour(cnt1, cnt2):
    """
    compare two contours, if the center of first contour is closer to the origin return True
    """
    A1 = cv2.contourArea(cnt1)
    A2 = cv2.contourArea(cnt2)
    if A1/max(A2, 1e-4) > 2:
        return True
    elif A2/max(A1, 1e-4) > 2:
        return False
    else:
        M1 = cv2.moments(cnt1)
        cX1 = int(M1["m10"] / max(M1["m00"], 1e-4))
        cY1 = int(M1["m01"] / max(M1["m00"], 1e-4))
        M2 = cv2.moments(cnt2)
        cX2 = int(M2["m10"] / max(M2["m00"], 1e-4))
        cY2 = int(M2["m01"] / max(M2["m00"], 1e04))
        return ((cX1-250)**2+(cY1-250)**2) < ((cX2-250)**2+(cY2-250)**2)

def merge_pcd(pcd_dicts: List, iou_thres: float) -> np.array:
    """
    merge the point clouds

    param: 
        pcd_dicts: the list of dictionary that contains
        iou_thres: the threshold of intersection detection
    """
    all_pcd_dicts = []
    for pcd_dict in pcd_dicts:
        pcds = pcd_dict["point_clouds"]
        for pcd in pcds:
            pcd["type"] = pcd_dict["type"]
        all_pcd_dicts += pcds
    
    non_intersection = np.ones((len(all_pcd_dicts)), dtype = bool)
    black = np.zeros((500, 500, 3), dtype = np.uint8)
    for i in range(len(all_pcd_dicts)):
        if not non_intersection[i]:
            continue
        query_contour = cv2.drawContours(black.copy(), [np.array(all_pcd_dicts[i]["contour"])], 0, (255, 255, 255), thickness=cv2.FILLED)
        for j in range(i+1, len(all_pcd_dicts)):
            if not non_intersection[j]:
                continue
            if all_pcd_dicts[i]["type"] == all_pcd_dicts[j]["type"]:
                continue
            contour = cv2.drawContours(black.copy(), [np.array(all_pcd_dicts[j]["contour"])], 0, (255, 255, 255), thickness=cv2.FILLED)
            iou = np.sum(np.logical_and(query_contour, contour)) / (np.sum(np.logical_or(query_contour, contour)))
            if iou > iou_thres:
                # plt.imshow(
                #     cv2.drawContours(black.copy(), [np.array(all_pcd_dicts[i]["contour"])], 0, (255, 0, 0), thickness=cv2.FILLED)+\
                #     cv2.drawContours(black.copy(), [np.array(all_pcd_dicts[j]["contour"])], 0, (0, 255, 0), thickness=cv2.FILLED)
                # )
                # plt.show()
                if compare_contour(np.array(all_pcd_dicts[i]["contour"]), np.array(all_pcd_dicts[j]["contour"])):
                    # plt.imshow(
                    #     cv2.drawContours(black.copy(), [np.array(all_pcd_dicts[i]["contour"])], 0, (255, 0, 0), thickness=cv2.FILLED)
                    # )
                    # plt.show()
                    non_intersection[j] = False
                else:
                    # plt.imshow(
                    #     cv2.drawContours(black.copy(), [np.array(all_pcd_dicts[j]["contour"])], 0, (0, 255, 0), thickness=cv2.FILLED)
                    # )
                    # plt.show()
                    non_intersection[i] = False
                    break
    non_intersect_pcd = []
    (valid, ) = np.where(non_intersection)
    print(f"remove {non_intersection.shape[0] - valid.shape[0]} contours")
    for index in valid:
        non_intersect_pcd.append(all_pcd_dicts[index]["points"])
    
    return np.concatenate(non_intersect_pcd, axis = 0)
        