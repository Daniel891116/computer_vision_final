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
    - Move the downloaded zip file into `data` and unzip it
    - Rename the output folder to  `public`
2. Private dataset:
    - Go to https://140.112.48.121:25251/sharing/PyViYwNsv
    - Press download
    - Move the downloaded zip file into `data` and unzip it
    - Rename the output folder to  `private`

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


## Related Informations

### Useful links

1. [Introduction to camera intrinsics and extrinsics](https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec)
2. [Base link implementation](http://wiki.ros.org/tf2_ros)
3. [OpenCV find contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
