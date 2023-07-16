import os
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from utils.marker_utils import read_road_marker
from shapely import geometry
from scipy import optimize
from rasterio.features import rasterize


class BaseSegmentWorker:
    """
    Base class for image segmentation and corner detection. All children that inherit this class should implement the "segment" function.
    """

    TYPE = ["zebracross", "stopline", "arrow", "junctionbox", "other"]

    def __init__(self) -> None:
        pass

    def __call__(self, img: Union[Image.Image, torch.Tensor], detection: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        masks = self.segment(T.PILToTensor()(img), detection)

        points = {}
        for type_name, segments in masks.items():
            contours = []
            for seg in segments:
                contours.append(self.__find_contour(seg))

            point = []

            for contour in contours:
                contour = contour.reshape((-1, 2))
                plt.scatter(contour[:, 0], contour[:, 1], s=3)
                point.append(contour)

            points[type_name] = point
        return points

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

    def __get_mask_corner(self, mask: np.ndarray) -> np.ndarray:
        mask = mask if mask.dtype == np.uint8 else mask.astype(np.uint8) * 255
        corners = cv2.goodFeaturesToTrack(mask, 6, 0.1, 2, useHarrisDetector=True).reshape((-1, 2))
        print(corners.shape)
        plt.scatter(corners[:, 0], corners[:, 1], s=10)
        return corners

    def __get_mask_corner_optimize(self, mask: np.ndarray) -> np.ndarray:
        def objective(vertices: np.ndarray) -> float:
            x = vertices[:4]
            y = vertices[4:]
            return geometry.Polygon(zip(x, y)).area

        def constraint(vertices: np.ndarray, mask: np.ndarray) -> int:
            x = vertices[:4]
            y = vertices[4:]
            # print(x)
            # print(y)
            poly = geometry.Polygon(zip(x, y))

            poly_mask = rasterize([poly], out_shape=mask.shape)
            m = mask > 0
            cover = poly_mask * m

            # plt.subplot(121)
            # plt.imshow(poly_mask * 255)
            # plt.subplot(122)
            # plt.imshow(m)
            # plt.show()

            return cover.sum() - m.sum()

        y, x = np.where(mask)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        initial = np.array([xmin, xmax, xmax, xmin, ymin, ymin, ymax, ymax])
        bounds = [(xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (ymin, ymax), (ymin, ymax), (ymin, ymax), (ymin, ymax)]
        constraints = {
            "type": "ineq",
            "fun": constraint,
            "args": (mask,),
        }

        result = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimized_vertex = result.x
        x, y = optimized_vertex[:4], optimized_vertex[4:]
        corners = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
        print(corners.shape)

        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(rasterize([geometry.Polygon(corners)], out_shape=mask.shape))
        plt.show()
        return corners

    def __get_corner(self, contour: np.ndarray) -> np.ndarray:
        """
        Find the corner of given contour.

        params:
            contour: contour of a segment, the size should be (n, 2)
        return:
            4 corners
        """
        print(f"Original contour: {contour.shape[0]}")
        # EPS = [2**i if i < 0 else i for i in range(-50, 10)]
        # for eps in EPS:
        #     approx = cv2.approxPolyDP(contour, eps, True).reshape((-1, 2))
        #     if approx.shape[0] == 4:
        #         break
        # print("eps:", eps, ", approx:", approx.shape)

        MAX_TOLERANCE = 10
        poly = geometry.Polygon(contour.reshape((-1, 2)))
        # print(poly.boundary)
        # poly_s = poly.simplify(TOLERANCE)
        # buffer = poly.buffer(
        #     distance=10,
        #     quad_segs=4,
        #     cap_style="square",
        #     join_style=2,
        # )
        # print(buffer)

        for tolerance in range(MAX_TOLERANCE + 1):
            poly_s = poly.simplify(tolerance)
            approx = np.array(poly_s.boundary.coords)

            print("Tolerance:", tolerance, ", approx:", approx.shape)
            if approx.shape[0] == 4:
                break

        # print(approx.shape)
        print("---------------end----------------")

        plt.scatter(approx[:, 0], approx[:, 1], s=2)
        return approx


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

        # ------------------------------Segment Anything----------------------------------
        masks = self.mask_generator.generate(img)
        print("Number of mask:", len(masks))

        segments = [mask["segmentation"] for mask in masks]

        # os.makedirs("mask", exist_ok=True)
        # for i, seg in enumerate(segments):
        #     cv2.imwrite("mask/mask_%d.png" % i, seg.astype(np.uint8) * 255)

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

        # cv2.imwrite("mask/road_mask.png", road_mask.astype(np.uint8) * 255)

        # Out of bounding box regions but on road
        # road_mask = cv2.imread("mask/road_mask.png", cv2.IMREAD_GRAYSCALE) == 255

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

        final_seg = np.zeros_like(segments[0])
        for type_name, segments in classified_segment.items():
            for seg in segments:
                final_seg += seg

        # fig, ax = plt.subplots()
        # for t, box in enumerate(boxes):
        #     bxmin, bymin, bxmax, bymax = box
        #     rect = patches.Rectangle((bxmin, bymin), bxmax - bxmin, bymax - bymin, linewidth=1, edgecolor="r", facecolor="none")
        #     ax.add_patch(rect)
        #     ax.annotate(self.TYPE[labels[t]], (bxmin, bymin - 10), color="r", fontsize=5)

        plt.imshow(final_seg)
        return classified_segment


if __name__ == "__main__":
    segment_worker = SegmentAnythingWorker("checkpoint/sam_vit_h.pth")

    frame_dir = "data/public/seq1/dataset/1681710717_571311700"
    # frame_dir = "ITRI_dataset/seq1/dataset/1681710730_129726917"
    img = Image.open(os.path.join(frame_dir, "raw_image.jpg"))
    detection = read_road_marker(os.path.join(frame_dir, "detect_road_marker.csv"))

    contours = segment_worker(img, detection)
    plt.axis("off")
    plt.show()
    # print(f"contours: {contours}")
    # image = np.asarray(img)
    # for k, v in contours.items():
    #     img = cv2.drawContours(image.copy(), v, -1, (0, 255, 0), 1)
    # print("type name:", k)
    # for seg in v:
    # print(seg.max(), seg.dtype, seg.shape)
    # plt.imsave(f"segment_anything{k}.png", img)
