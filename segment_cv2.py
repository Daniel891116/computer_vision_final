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

    def __init__(self, eps_coef, thres, show_contour = False) -> None:
        self.eps_coef = eps_coef
        self.contour_thres = thres
        self.show_contour = show_contour
        
    def __call__(self, img: Union[Image.Image, torch.Tensor], detection: Dict[str, torch.Tensor]) -> Any:
        cropped_imgs = self.__crop_bbox_region(img, detection)

        contours = []
        for crop_i in cropped_imgs:
            approx = self.__segment(crop_i["img"])
            contours.append({"type": crop_i["type"], "contour": approx})
            temp_img = crop_i['img'].permute(1,2,0).numpy()
            terminated = False
            if self.show_contour:
                cv2.drawContours(temp_img, approx, -1, (0, 255, 0), 1)
                cv2.imshow('Rectangles', temp_img)
                a = cv2.waitKey(0)
                if a == 27:
                    terminated = True
                cv2.destroyAllWindows()
            
        return contours, terminated

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
                    "type": labels,
                    "img": torch.Tensor
                }, ...
            ]
        """
        img = img if isinstance(img, torch.Tensor) else T.PILToTensor()(img)

        N_OBJECTS = detection["labels"].shape[0]
        assert (
            N_OBJECTS == detection["boxes"].shape[0]
        ), """The number of labels should match of that of the boxes. i.e. the "labels" should have a shape of (N, ) and the "boxes" should have a shape of (N, 4)"""

        boxes = torch.round(detection["boxes"]).to(torch.int32)
        cropped_imgs = []
        for i in range(N_OBJECTS):
            xmin, ymin, xmax, ymax = boxes[i]
            cropped_imgs.append({"type": detection["labels"][i].item(), "img": img[:, ymin:ymax, xmin:xmax]})

        return cropped_imgs

    def __segment(self, img: torch.Tensor) -> torch.Tensor:
        img_array = np.array(img.permute(1, 2, 0))
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Rectangles', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        mean = gray.mean()
        std = gray.std()
        low = self.contour_thres*std+mean
        ret, thresh = cv2.threshold(gray, low, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Rectangles', thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        contours = self.__find_contour(torch.from_numpy(thresh))
        approx_list = self.__get_approx(contours)
        return approx_list

    def __find_contour(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Find the contour of one instance mask.

        params:
            mask: mask of one instance. torch.Tensor (h, w), the tensor values should be either 0/1 or boolean.
        return:
            The contour of the mask. torch.Tensor (h, w)
        """
        # print(mask)
        mask_bin: np.ndarray = mask.numpy()
        # TODO: check whether cv2.CHAIN_APPROX_NONE or cv2.CHAIN_APPROX_SIMPLE is suitable
        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def __get_approx(self, contours: np.ndarray) -> np.ndarray:
        approx_list = []
        for contour in contours:
            epsilon = self.eps_coef * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_list.append(approx)
        return approx_list
