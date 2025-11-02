#!/usr/bin/env python3
"""
1-yolo.py

Extends the Yolo class to include the process_outputs method
for decoding the raw predictions of a YOLO v3 Darknet model.
"""

import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    """YOLO v3 object detector."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the YOLO model and parameters."""
        self.model = load_model(model_path)

        with open(classes_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs.

        Args:
            outputs: list of numpy.ndarrays containing model predictions
                     each of shape (grid_h, grid_w, anchor_boxes, 4 + 1 + classes)
            image_size: numpy.ndarray of shape (2,) -> [image_height, image_width]

        Returns:
            (boxes, box_confidences, box_class_probs)
                boxes: list of ndarrays with shape (grid_h, grid_w, anchor_boxes, 4)
                box_confidences: list of ndarrays (grid_h, grid_w, anchor_boxes, 1)
                box_class_probs: list of ndarrays (grid_h, grid_w, anchor_boxes, classes)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_h, img_w = image_size
        input_h, input_w = self.model.input.shape[1].value, self.model.input.shape[2].value
        # For TensorFlow 2.x eager mode:
        if input_h is None or input_w is None:
            input_h, input_w = self.model.input.shape[1:3]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract components
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Box confidence and class probabilities
            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))  # sigmoid
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))  # sigmoid

            # Grid offsets
            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_x = np.tile(c_x, (grid_h, 1, anchor_boxes))
            c_y = np.tile(c_y, (1, grid_w, anchor_boxes))

            # Apply sigmoid to center coordinates
            bx = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            # Scale anchors
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = (np.exp(t_w) * pw) / input_w
            bh = (np.exp(t_h) * ph) / input_h

            # Convert to corner coordinates
            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            box = np.stack((x1, y1, x2, y2), axis=-1)

            boxes.append(box)
            box_confidences.append(box_conf)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
