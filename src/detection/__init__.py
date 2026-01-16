"""
Detection module for USB Camera Viewer
Contains object detection using YOLO

Supports two backends:
- ultralytics (YOLOv8) - requires torch
- OpenCV DNN (YOLOv4-tiny) - fallback, no torch required
"""

from .person_detector import (
    PersonDetector,
    PersonDetectorOpenCV,
    PersonDetectorUltralytics,
    PersonDetectorAsync,
    Detection
)

__all__ = [
    'PersonDetector',
    'PersonDetectorOpenCV',
    'PersonDetectorUltralytics',
    'PersonDetectorAsync',
    'Detection'
]
