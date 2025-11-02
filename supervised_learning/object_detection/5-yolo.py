#!/usr/bin/env python3
"""
5-yolo.py
Contains the Yolo class with image preprocessing.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    """Yolo v3 object detector."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo."""
        self.model = load_model(model_path)
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [c.strip() for c in f if c.strip()]
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Dummy placeholder (from 2-yolo.py)."""
        pass

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Dummy placeholder (from 2-yolo.py)."""
        pass

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Suppress overlapping boxes using NMS."""
        box_predictions = []
        predicted_classes = []
        predicted_scores = []

        for c in np.unique(box_classes):
            idxs = np.where(box_classes == c)
            boxes = filtered_boxes[idxs]
            scores = box_scores[idxs]
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            while order.size > 0:
                i = order[0]
                box_predictions.append(boxes[i])
                predicted_classes.append(c)
                predicted_scores.append(scores[i])

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

        box_predictions = np.array(box_predictions)
        predicted_classes = np.array(predicted_classes)
        predicted_scores = np.array(predicted_scores)
        return box_predictions, predicted_classes, predicted_scores

    @staticmethod
    def load_images(folder_path):
        """Load all images in a folder."""
        images = []
        image_paths = []
        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    image_paths.append(path)
        return images, image_paths

    def preprocess_images(self, images):
        """Resize, normalize, and convert images for YOLO."""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            h, w = img.shape[:2]
            image_shapes.append([h, w])

            # Resize using INTER_AREA
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_AREA)

            # Normalize pixel values
            normalized = resized / 255.0
            pimages.append(normalized)

        return np.array(pimages), np.array(image_shapes)
