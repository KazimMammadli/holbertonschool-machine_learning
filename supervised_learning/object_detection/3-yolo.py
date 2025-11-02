#!/usr/bin/env python3
"""YOLO v3 non-max suppression implementation."""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """YOLO v3 object detection class."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO model."""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process model outputs."""
        image_h, image_w = image_size
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]
            tx, ty, tw, th = (output[..., j] for j in range(4))
            obj_conf = output[..., 4:5]
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
            box_confidences.append(1 / (1 + np.exp(-obj_conf)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes using object and class confidence thresholds."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_scores_i = (box_confidences[i] * box_class_probs[i])
            box_classes_i = np.argmax(box_scores_i, axis=-1)
            box_class_scores_i = np.max(box_scores_i, axis=-1)

            mask = box_class_scores_i >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_classes_i[mask])
            box_scores.append(box_class_scores_i[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression."""
        box_predictions = []
        predicted_classes = []
        predicted_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]
            order = np.argsort(cls_scores)[::-1]

            while len(order) > 0:
                i = order[0]
                box_predictions.append(cls_boxes[i])
                predicted_classes.append(cls)
                predicted_scores.append(cls_scores[i])

                if len(order) == 1:
                    break

                x1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                y1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                x2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                y2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                box_area = ((cls_boxes[i, 2] - cls_boxes[i, 0]) *
                            (cls_boxes[i, 3] - cls_boxes[i, 1]))
                other_areas = ((cls_boxes[order[1:], 2] -
                                cls_boxes[order[1:], 0]) *
                               (cls_boxes[order[1:], 3] -
                                cls_boxes[order[1:], 1]))

                union = box_area + other_areas - inter_area
                iou = inter_area / union
                keep = np.where(iou <= self.nms_t)[0]
                order = order[keep + 1]

        box_predictions = np.array(box_predictions)
        predicted_classes = np.array(predicted_classes)
        predicted_scores = np.array(predicted_scores)

        return box_predictions, predicted_classes, predicted_scores
