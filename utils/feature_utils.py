from typing import List, Tuple

import cv2
import numpy as np


def get_SIFT_features(image: np.array) -> Tuple[List, np.ndarray]:
    """
    get all feature points using SIFT algorithm

    param:
        image: input image
    return:
        (keypoints, descriptor)
    """
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    keypoints, descriptor = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    return (keypoints, descriptor)

def contours_to_keypoints(contours: List) -> List[cv2.KeyPoint]:
    key_points = []
    for set in contours:
        for contour in set:
            kp = cv2.KeyPoint(int(contour[0][0]), int(contour[0][1]), 1)
            key_points.append(kp)
    return key_points

def get_SIFT_descriptor(image: np.array, contours: List) -> Tuple[List, np.ndarray]:
    """
    compute SIFT descriptors of given contours points
    """
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)
    kps = contours_to_keypoints(contours)
    des = sift.compute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), kps)
    return kps, des

def filter_keypoints(keypoints: List, descriptors: np.ndarray, contours: List) -> Tuple[List, np.ndarray]:
    """
    discard the keypoints that are outside of coutour

    param:
        keypoints: list of keypoints that get from SIFT algorithm
        descriptors: SIFT descriptors of input keypoints
        contours: List of filtering contour
    """
    filtered_keypoints = []
    filtered_descriptors = []
    assert len(keypoints) == descriptors.shape[0], "the number of keypoints not match descriptors size"
    for keypoint, descriptor in zip(keypoints, descriptors):
        for contour in contours:
            distance = cv2.pointPolygonTest(contour, (int(keypoint.pt[0]), int(keypoint.pt[1])), True)
            if distance >= 0:
                filtered_keypoints.append(keypoint)
                filtered_descriptors.append(descriptor)
    return (filtered_keypoints, filtered_descriptors)
