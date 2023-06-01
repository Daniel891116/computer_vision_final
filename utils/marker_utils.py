from typing import Dict

import numpy as np
import torch


def read_road_marker(marker_file: str) -> Dict[str, np.ndarray]:
    with open(marker_file, "r") as file:
        boxes, labels, scores = [], [], []
        for line in file.readlines():
            num_str = line.replace("\n", "").split(",")
            if len(num_str) != 6:
                break

            boxes.append([float(bound) for bound in num_str[:4]])
            labels.append(int(num_str[-2]))
            scores.append(float(num_str[-1]))

    return {
        "labels": np.array(labels),
        "boxes": np.array(boxes),
        "scores": np.array(scores),
    }
