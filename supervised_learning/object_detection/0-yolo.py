#!/usr/bin/env python3
"""
0-yolo.py

Contains the Yolo class which loads a Darknet Keras model and its class names,
and stores thresholds and anchors for detection post-processing.
"""

from typing import List
import numpy as np
from tensorflow.keras.models import load_model


class Yolo:
    """
    Yolo v3 object detector helper.

    Public attributes:
        model: the Darknet Keras model
        class_names: list of class names (strings)
        class_t: box score threshold for the initial filtering step (float)
        nms_t: IoU threshold for non-max suppression (float)
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
    """

    def __init__(self, model_path: str, classes_path: str,
                 class_t: float, nms_t: float, anchors: np.ndarray):
        """
        Initialize Yolo instance.

        Args:
            model_path: path to Darknet Keras model (.h5)
            classes_path: path to file containing class names (one per line)
            class_t: box score threshold for initial filtering
            nms_t: IoU threshold for non-max suppression
            anchors: numpy.ndarray (outputs, anchor_boxes, 2)
        """
        # Load the Darknet Keras model
        self.model = load_model(model_path)

        # Load class names from file
        with open(classes_path, 'r', encoding='utf-8') as f:
            # strip whitespace/newlines and ignore empty lines
            self.class_names: List[str] = [
                line.strip() for line in f if line.strip()
            ]

        # Store thresholds and anchors
        self.class_t: float = float(class_t)
        self.nms_t: float = float(nms_t)
        self.anchors: np.ndarray = anchors
