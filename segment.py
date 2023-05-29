import torch
from torchvision import transforms as T
import numpy as np
from PIL import Image
import cv2
from typing import Any, List, Union, Dict


class BaseSegmentWorke:
    """
    Base class for image segmentation and corner detection. All children that inherit this class should implement the "segment" function.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, img: Union[Image.Image, torch.Tensor], detection: Dict[str, torch.Tensor]) -> Any:
        cropped_imgs = self.__crop_bbox_region(img, detection)

        masks = []
        for crop_i in cropped_imgs:
            masks.append({"type": crop_i["type"], "mask": self.__segment(crop_i["img"])})

    def __crop_bbox_region(self, img: Union[Image.Image, torch.Tensor], detection: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Crop the bounding box region from the given image.

        params:
            img: A frame of a video sequence. PIL image or torch.Tensor (3, h, w).
            detection: The objects detected by Yolo, should at least contain two keys: ["labels", "boxes"]
                labels: The type of the object that the bbox have surrounded, 0 stands for zebracross and 1 stands for stopline. torch.Tensor.
                boxes: The bounding box that surround one object. The format should be [xmin, ymin, xmax, ymax]. Either the boundary format of the boxes is float or int is accepted.
                %% Notice: the number of labels should match of that of the boxes. i.e. the "labels" should have a shape of (N, ) and the "boxes" should have a shape of (N, 4)
        return:
            The cropped image with object type specified.
            [
                {
                    "type": 0 or 1,
                    "img": torch.Tensor
                }, ...
            ]
        """
        img = img if isinstance(img, torch.Tensor) else T.PILToTensor()(img)
        img = img / 255.0  # normalize to 0 ~ 1

        N_OBJECTS = detection["labels"].shape[0]
        assert (
            N_OBJECTS == detection["boxes"].shape[0]
        ), """The number of labels should match of that of the boxes. i.e. the "labels" should have a shape of (N, ) and the "boxes" should have a shape of (N, 4)"""

        boxes = torch.round(detection["boxes"])
        cropped_imgs = []
        for i in range(N_OBJECTS):
            xmin, ymin, xmax, ymax = boxes[i]
            cropped_imgs.append({"type": detection["labels"][i].item(), "img": img[:, ymin:ymax, xmin:xmax]})

        return cropped_imgs

    def __segment(self, img: torch.Tensor) -> torch.Tensor:
        """
        Segment the instances in the image. Child classes should implement this function

        params:
            img: The cropped image. torch.Tensor (3, h, w).
        return:
            Masks of instances. The dimension is (N_INS, h, w), where N_INS is the number of instances.
        """
        raise NotImplementedError()

    def __find_contour(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Find the contour of one instance mask.

        params:
            mask: mask of one instance. torch.Tensor (h, w), the tensor values should be either 0/1 or boolean.
        return:
            The contour of the mask. torch.Tensor (h, w)
        """
        mask_bin: np.ndarray = mask.type(torch.bool).numpy()
        # TODO: check whether cv2.CHAIN_APPROX_NONE or cv2.CHAIN_APPROX_SIMPLE is suitable
        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def __get_corner(self, contours: np.ndarray) -> np.ndarray:
        ...
