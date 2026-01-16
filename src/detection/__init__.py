"""
Detection module for USB Camera Viewer
Contains object detection using YOLO with 3D position estimation

Supports two backends:
- ultralytics (YOLOv8) - requires torch
- OpenCV DNN (YOLOv4-tiny) - fallback, no torch required

Features:
- Person detection with bounding boxes
- 2D center point (x, y) in pixels
- 3D position estimation (X, Y, Z) in meters using pinhole camera model
- Depth (Z-axis) estimation based on bbox height and assumed human height
"""

from .person_detector import (
    PersonDetector,
    PersonDetectorOpenCV,
    PersonDetectorUltralytics,
    PersonDetectorAsync,
    Detection,
    DepthEstimator
)

__all__ = [
    'PersonDetector',
    'PersonDetectorOpenCV',
    'PersonDetectorUltralytics',
    'PersonDetectorAsync',
    'Detection',
    'DepthEstimator'
]
