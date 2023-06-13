# 3D Reconstruction from Road Marker Feature Points

This repository contains the code for performing 3D reconstruction from road marker feature points. The pipeline consists of several steps, including segmentation, 3D reconstruction, merging views, and refinement. This readme provides instructions on how to use the code and prepare the environment.

## Pipeline

The pipeline consists of the following steps:

1. Segmentation: This step involves segmenting the road marker region from an image. It includes cropping the bounding box region, instance segmentation, finding contours, and approximating polygonal curves.

2. 3D Reconstruction: The segmented feature points are used for 3D reconstruction. The pipeline currently supports two methods: structure from motion and pinhole modeling.

3. Merge Views: The single-view 3D point clouds are merged into one to create a more complete representation.

4. Refinement: The estimated 3D point cloud is refined using time series data, acceleration, and velocity information.

## Preparation

### Environment

To set up the environment, follow these steps:

```bash
conda create -n cv python=3.9
conda activate cv
```

### Clone Repository

Clone the repository and navigate to the `computer_vision_final` directory:

```bash
git clone https://github.com/Daniel891116/computer_vision_final.git
cd computer_vision_final
```

### Download Data

Download the dataset from the following link: [Dataset Link](https://drive.google.com/file/d/19jDCQhw3pUMERftAxQjz7p9JOzpuNw4A/view?usp=drive_link)

After downloading the zip file, move it to the repository folder and unzip it. Rename the output folder to `ITRI_dataset`.

### Package Installation

Install the required packages by running the following command:

```bash
pip3 install -r requirements.txt
```

Additionally, install the following packages:

- [Segment Anything](https://github.com/facebookresearch/segment-anything): Install the package and download the model checkpoint (ViT-H SAM model). Rename the checkpoint file to `sam_vit_h.pth` and place it in the `checkpoint` folder.
    ```bash
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

- [Mask2Former](https://github.com/facebookresearch/Mask2Former): Install the detectron2 package using the provided command.
    ```bash
    # detectron2
    python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

## Usage

To reconstruct the point cloud file for each sequence of the dataset, use the following command:

```bash
python3 reconstruct.py --seq_dir ITRI_dataset/[seq_name] --camera_dir_path ITRI_dataset/camera_info/lucid_cameras_x00 --segment_method [segment_method] --range 25
```

Replace `[seq_name]` with the name of the sequence and `[segment_method]` with the desired segmentation method. This command will generate a directory containing the point clouds for each camera and sequence.

    [seq_name]
    ├── [segment_method]_pointclouds_[range]
    │   ├── b
    │   │   ├──[timestamps].csv
    │   │   └──...
    │   ├── f
    │   ├── fl
    │   └── fr

After generating the point clouds, you can calculate the dx and dy of each frame using the [ICP]((https://zhuanlan.zhihu.com/p/107218828)) method (threshold: 0.7, iteration: 30) based on the sub_map.csv file in the dataset. Use the following command:

```bash
python3 gendxdy.py --target_pcd_dir ./[seq_name]/[segment_method]_pointclouds_[range] --output_file solution/[seq_name]/pred_pose.txt --seq_dir_path [seq_name] --visualize
```

Replace `[seq_name]` with the name of the sequence. This command will generate a prediction file containing the pose information.

To visualize the predicted trajectory, use the following command:

```bash
python3 utils/plot_utils.py --pred_file solution/[seq_name]/pred_pose.txt --smooth
```

Replace `[seq_name]` with the name of the sequence.

## Related Information

### Dataset Format

The dataset follows a specific
 format. Here is an overview of the directory structure:

- seq/ (e.g., seq1, seq2, seq3): Contains the dataset for each sequence.
  - dataset/{time_stamp}/ (e.g., 1681710717_532211005): Contains the data for each timestamp.
    - camera.csv: Camera name.
    - detect_road_marker.csv: Detected bounding boxes and their properties.
    - initial_pose.csv: Initial pose for ICP in "base_link" frame.
    - raw_image.png: Captured RGB image.
    - sub_map.csv: Map points for ICP (x, y, z).
    - gound_turth_pose.csv (not available in all directories): Ground truth localization in "base_link" frame.
- other_data/: Contains additional data files.

- camera_info/{camera}/ (e.g., lucid_cameras_x00): Contains camera-specific information, such as intrinsic parameters, mask images, and transformation parameters.

### Useful Links

Here are some useful links related to the project:

1. [Introduction to camera intrinsics and extrinsics](https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec)
2. [Base link implementation](http://wiki.ros.org/tf2_ros)
3. [OpenCV find contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)