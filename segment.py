import torch
from torchvision import transforms as T
import numpy as np
from PIL import Image
import cv2
from typing import Any, List, Union, Dict
from matplotlib import pyplot as plt
from matplotlib import patches
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from tools import read_road_marker


class BaseSegmentWorker:
    """
    Base class for image segmentation and corner detection. All children that inherit this class should implement the "segment" function.
    """

    TYPE = ["zebracross", "stopline", "arrow", "junctionbox", "other"]

    def __init__(self) -> None:
        pass

    def __call__(self, img: Union[Image.Image, torch.Tensor], detection: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        masks = self.segment(T.PILToTensor()(img), detection)
        # return masks

        # h, w = T.PILToTensor()(img).shape[1:]
        # view = np.zeros((h, w))
        # color = {"zebracross": "r", "stopline": "g", "arrow": "b", "junctionbox": ""}
        points = {}
        for type_name, segments in masks.items():
            contours = []
            for seg in segments:
                # view += seg
                contours.append(self.__find_contour(seg))

            point = []
            # if type_name == self.TYPE[0]:  # zebracross
            for contour in contours:
                contour = contour.reshape((-1, 2))
                point.append(contour)
                # plt.scatter(
                #     contour[::5, 0],
                #     contour[::5, 1],
                #     s=2,
                #     c=[self.TYPE.index(type_name)],
                # )
            points[type_name] = point
        return points
        # view = cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # plt.imshow(view)
        # plt.title(type_name)
        # plt.axis("off")
        # plt.show()

    def segment(self, img: torch.Tensor, detection: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Segment the instances in the image. Child classes should implement this function

        params:
            img: The cropped image. torch.Tensor (3, h, w).
            detection: Predicted bounding boxes.
        return:
            Dictionary with type as key and segment masks as
        """
        raise NotImplementedError()

    def __find_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        Find the contour of one instance mask.

        params:
            mask: mask of one instance. torch.Tensor (h, w), the tensor values should be either 0/1 or boolean.
        return:
            The contour of the mask. torch.Tensor (h, w)
        """
        contours, hierarchy = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return np.concatenate(contours, axis=0)

    def __get_corner(self, contours: np.ndarray) -> np.ndarray:
        ...


class SegmentAnythingWorker(BaseSegmentWorker):
    """
    Uses Facebook/segment-anything to perform instance segmentation
    """

    BOUND_TOLERATE = 5
    COLOR_THRESHOLD = 160
    VALID_COLOR_RATIO = 0.3

    def __init__(self, model_checkpoint_dir: str) -> None:
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # segment anything
        model_type = "default"
        if "vit_l" in model_checkpoint_dir:
            model_type = "vit_l"
        elif "vit_b" in model_checkpoint_dir:
            model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=model_checkpoint_dir)
        self.mask_generator = SamAutomaticMaskGenerator(sam.to(self.device))

        # Mask2Former
        self.img_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic").to(self.device)

    def segment(self, img: torch.Tensor, detection: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        c, h, w = img.shape
        img = img.permute(1, 2, 0).numpy()

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        masks = self.mask_generator.generate(img)
        print("Number of mask:", len(masks))

        segments = [mask["segmentation"] for mask in masks]

        # for i, seg in enumerate(segments):
        #     cv2.imwrite("mask_%d.png" % i, seg.astype(np.uint8) * 255)

        # ------------------------------Mask2Former----------------------------------
        input_mf = {k: v.to(self.device) for k, v in self.img_processor(images=img, return_tensors="pt").items()}
        with torch.no_grad():
            output_mf = self.mask2former(**input_mf)
        result_mf = self.img_processor.post_process_panoptic_segmentation(output_mf, target_sizes=[[h, w]])[0]
        road_mask = np.zeros((h, w))
        for segment in result_mf["segments_info"]:
            obj_id = segment["id"]
            label_id = segment["label_id"]
            label_name = self.mask2former.config.id2label[label_id]
            if "road" in label_name:
                road_mask += result_mf["segmentation"].cpu().numpy() == obj_id
        # ------------------------------Segment Anything----------------------------------

        # Out of bounding box regions but on road
        # road_mask = cv2.imread("mask/road.png", cv2.IMREAD_GRAYSCALE) == 255

        # segments = [cv2.cvtColor(cv2.imread("mask/mask_%d.png" % img_id), cv2.COLOR_BGR2GRAY) for img_id in range(73)]

        classified_segment = {k: [] for k in self.TYPE}
        boxes = detection["boxes"]
        labels = detection["labels"]
        for i, seg in tqdm(enumerate(segments)):
            y, x = np.where(seg)
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)

            in_box = (
                (xmin >= (boxes[:, 0] - self.BOUND_TOLERATE))
                * (xmax <= (boxes[:, 2] + self.BOUND_TOLERATE))
                * (ymin >= (boxes[:, 1] - self.BOUND_TOLERATE))
                * (ymax <= (boxes[:, 3] + self.BOUND_TOLERATE))
            )

            idx = np.where(in_box)[0]

            if len(idx) == 0:  # not a valid segment
                seg_n = np.sum(seg)
                seg_n_road = np.sum(seg * road_mask)
                seg_n_color = np.sum(seg * (gray_img > self.COLOR_THRESHOLD))

                if seg_n_road / seg_n >= 0.99 and seg_n_color / seg_n >= self.VALID_COLOR_RATIO:  # most in road region and color should be correct
                    classified_segment["other"].append(seg)
            elif len(idx) == 1:  # possible valid segment
                type_id = labels[idx[0]]
                type_name = self.TYPE[type_id]

                classified_segment[type_name].append(seg)
            elif len(idx) > 1:  # TODO
                continue

        return classified_segment

        # final_seg = np.zeros_like(segments[0])
        # for type_name, segments in classified_segment.items():
        #     for seg in segments:
        #         final_seg += seg

        # fig, ax = plt.subplots()
        # for t, box in enumerate(boxes):
        #     bxmin, bymin, bxmax, bymax = box
        #     rect = patches.Rectangle((bxmin, bymin), bxmax - bxmin, bymax - bymin, linewidth=1, edgecolor="r", facecolor="none")
        #     ax.add_patch(rect)
        #     ax.annotate(self.TYPE[labels[t]], (bxmin, bymin - 10), color="r", fontsize=5)

        # plt.imshow(final_seg)
        # plt.show()


if __name__ == "__main__":
    segment_worker = SegmentAnythingWorker("checkpoint/sam_vit_b.pth")

    frame_dir = "data/public/seq1/dataset/1681710717_571311700"
    img = Image.open(os.path.join(frame_dir, "raw_image.jpg"))
    detection = read_road_marker(os.path.join(frame_dir, "detect_road_marker.csv"))

    contours = segment_worker(img, detection)

    for k, v in contours.items():
        print("type name:", k)
        for seg in v:
            print(seg.max(), seg.dtype, seg.shape)
