# 3D Reconstruction from Road Marker Feature Points

## Pipeline

`Image` + `object bounding box` (predicted by pretrained Yolo network)

1. Segmentation: For one bounding boux (zebracross/stopline)
    - Crop bounding box region from image
    - Instance segmentation
        1. threshold + inRange (`南`)
        2. Segment anything  (`傅`)
    - Find contour
    - Approx Poly DP

`Four corner points` (zerbracross) or `points on a line` (stopline)

2. 3D reconstruction:
    - `Structure from motion` (Currently used) (`黃`)
    - Pinhole modeling

`Frame-wise` 3D point cloud per camera

3. Merge views: Merge single-view 3D point clouds to one (`葉`)
    - base_link methods

`Merged frame-wise` 3D point cloud

4. Refinement: Using the following datas and information to refine the estimated 3D point cloud (`蔡`)
    - Time series data
    - Acceleration and velocity

`Refined and merged frame-wise` 3D point cloud

## Preperation

### Environment
```bash
conda create -n cv python=3.9
# You should press 'y' and 'enter'
conda activate cv
```

### Download data

1. Public dataset
    - Go to https://140.112.48.121:25251/sharing/Lw8QTICUf
    - Press download
    - Move the zip file into the repository folder and unzip it
    - Rename the output folder to  `ITRI_dataset`
2. Private dataset:
    - Go to https://140.112.48.121:25251/sharing/PyViYwNsv
    - Press download
    - Move the zip file into the repository folder and unzip it
    - Rename the output folder to  `ITRI_DLC`
3. Updated init_pose
    - Go to https://140.112.48.121:25251/sharing/Gb5NhrV3v
    - Press download
    - Move the zip file into the repository folder and unzip it
    - Rename the output folder to  `ITRI_DLC2`

### Package Installation

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
    1. Install package
        ```bash
        pip install git+https://github.com/facebookresearch/segment-anything.git
        ```
    2. Download model checkpoint (currently use ViT-H SAM model) and put them in the `checkpoint` folder
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
    1. Install package
        ```bash
        # detectron2
        python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        ```
- Other packages
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

cloning our repository and go into the computer_vision_final directory.
```bash
git clone https://github.com/Daniel891116/computer_vision_final.git
cd computer_vision_final
```

reconstruct the point cloud file of each sequence of dataset. (Get more information about dataset in the next section.)
```bash
python3 reconstruct --seq_dir ITRI_dataset/seq1 --camera_dir_path ITRI_dataset/camera_info/lucid_cameras_x00 --segment_method SAM
```

This would generate a directory that contains the point clouds of each camera of each sequence. Below is the structure of generated directory

    [sequence name]
    ├── [segment_method]_pointclouds_[range]
    │   ├── b
    │   │   ├──[timestamps].csv
    │   │   └──...
    │   ├── f
    │   ├── fl
    │   └── fr

After generating corresponding point clouds of given timestamps, this step will calculate the dx and dy of each frame using [ICP](https://zhuanlan.zhihu.com/p/107218828) method base on the sub_map.csv file in the dataset. **Notice** if visualize is set to true, it would create a window that shows the point cloud.
```bash
python3 gendxdy.py --target_pcd_dir ./seq1/CV_pointclouds --output_file solution/seq1/pred_pose.txt --seq_dir_path seq1 --visualize
```
After generating prediction position, we provide a function to visualize your predicted trajectory. This function will 
```bash
python3 viz_trajectory.py solution/seq1/pred_pose_25.txt
```

## Related Informations

### Dataset format

    seq/ (e.g. seq1, seq2, seq3)
        dataset/
            {time_stamp}/ (e.g. 1681710717_532211005)
                1. camera.csv: 
                    camera name
                2. detect_road_marker.csv:
                    a. detected bounding boxes, the bounding box are not always correct.
                    b. format: (x1, y1, x2, y2, class_id, probability)
                    c. class_id: (0:zebracross, 1:stopline, 2:arrow, 3:junctionbox, 4:other)
                3. initial_pose.csv:
                    initial pose for ICP in "base_link" frame.
                4. raw_image.png:
                    captured RGB image
                5. sub_map.csv:
                    map points for ICP, (x, y, z).
                6. gound_turth_pose.csv: """not exist in all dirs"""
                    x, y localization ground turth in "base_link" frame.

        other_data/
            {timestamp}_raw_speed.csv: (e.g. 1681710717_572170877_raw_speed.csv)
                car speed(km/hr)
            {timestamp}_raw_imu.csv:
                1st line: orientation: x, y, z, w
                2nd line: angular_velocity: x, y, z
                3rd line: linear_acceleration: x, y, z

        all_timestamp.txt:
            list all directories in time order
        localization_timestamp.txt:
            list all directories with "gound_turth_pose.csv" in time order


    camera_info/
        {camera}/ (e.g. lucid_cameras_x00)
            {camera_name}_camera_info.yaml: (e.g. gige_100_b_hdr_camera_info.yaml)
                intrinsic parameters
            {camera_name}_mask.png:
                The mask for the ego car show in the image, it could help for decreasing some false alarms in detection.
            camera_extrinsic_static_tf.launch:
                transformation parameters between cameras
                key_word: tf2_ros, Robot Operating System (ROS)


### Useful links

1. [Introduction to camera intrinsics and extrinsics](https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec)
2. [Base link implementation](http://wiki.ros.org/tf2_ros)
3. [OpenCV find contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
