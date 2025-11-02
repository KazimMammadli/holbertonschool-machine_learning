#!/usr/bin/env python3
"""YOLO v3 with filter_boxes method."""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """YOLO v3 object detection class."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process YOLO outputs."""
        image_h, image_w = image_size
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            tx, ty, tw, th = (output[..., 0], output[..., 1],
                              output[..., 2], output[..., 3])
            object_conf = output[..., 4:5]
            class_probs = output[..., 5:]

            cx = np.tile(np.arange(grid_w).reshape(1, grid_w, 1),
                         (grid_h, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_h).reshape(grid_h, 1, 1),
                         (1, grid_w, anchor_boxes))

            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_w
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_h
            bw = (anchors[:, 0] * np.exp(tw)) / self.model.input.shape[1]
            bh = (anchors[:, 1] * np.exp(th)) / self.model.input.shape[2]

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-object_conf)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes by score threshold."""
        f_boxes, f_classes, f_scores = [], [], []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            mask = class_scores >= self.class_t

            f_boxes.append(boxes[i][mask])
            f_classes.append(classes[mask])
            f_scores.append(class_scores[mask])

        f_boxes = np.concatenate(f_boxes, axis=0)
        f_classes = np.concatenate(f_classes, axis=0)
        f_scores = np.concatenate(f_scores, axis=0)

        return f_boxes, f_classes, f_scores
