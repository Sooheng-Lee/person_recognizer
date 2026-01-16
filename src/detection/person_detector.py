"""
Person detection module using YOLO
Detects people in video frames and extracts their coordinates

Supports:
- YOLOv8 via ultralytics (primary, requires torch)
- YOLOv4-tiny via OpenCV DNN (fallback, no torch required)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import time
import os
import urllib.request

from ..utils.logger import get_logger


@dataclass
class Detection:
    """
    Represents a single person detection with 3D coordinates.
    
    Attributes:
        id: Tracking ID (if tracking enabled)
        bbox: Bounding box [x1, y1, x2, y2]
        center: Center point (x, y)
        center_3d: 3D center point (x, y, z) where z is estimated depth
        depth_z: Estimated depth/distance in relative units or meters
        confidence: Detection confidence score
        class_id: Class ID (0 for person in COCO)
        class_name: Class name
    """
    id: int = -1
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    center: Tuple[int, int] = (0, 0)
    center_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z
    depth_z: float = 0.0  # Estimated depth in meters (approximate)
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = "person"
    
    @property
    def x1(self) -> int:
        return self.bbox[0]
    
    @property
    def y1(self) -> int:
        return self.bbox[1]
    
    @property
    def x2(self) -> int:
        return self.bbox[2]
    
    @property
    def y2(self) -> int:
        return self.bbox[3]
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "center_3d": list(self.center_3d),
            "depth_z": round(self.depth_z, 2),
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "width": self.width,
            "height": self.height,
            "area": self.area
        }
    
    def __str__(self) -> str:
        return f"Person(id={self.id}, center={self.center}, depth={self.depth_z:.2f}m, conf={self.confidence:.2f})"


class DepthEstimator:
    """
    Estimates depth (z-axis distance) from bounding box size.
    
    Uses the pinhole camera model with assumed average human height.
    This is an approximation and works best when:
    - Camera is roughly at human chest/face height
    - People are standing upright
    - Focal length is calibrated
    
    Formula: Z = (f * H_real) / h_pixel
    where:
        Z = depth distance
        f = focal length (in pixels)
        H_real = real height of person (average ~1.7m)
        h_pixel = height of person in pixels (bounding box height)
    """
    
    # Average human height in meters
    AVERAGE_HUMAN_HEIGHT = 1.7  # meters
    
    # Default focal length (can be calibrated)
    # Approximate for typical webcam at 1080p
    DEFAULT_FOCAL_LENGTH = 800  # pixels
    
    def __init__(
        self,
        focal_length: float = None,
        reference_height: float = None,
        frame_height: int = 1080
    ):
        """
        Initialize depth estimator.
        
        Args:
            focal_length: Camera focal length in pixels (None for auto-estimate)
            reference_height: Reference human height in meters
            frame_height: Frame height for focal length estimation
        """
        self.reference_height = reference_height or self.AVERAGE_HUMAN_HEIGHT
        self.frame_height = frame_height
        
        if focal_length is not None:
            self.focal_length = focal_length
        else:
            # Estimate focal length based on frame height
            # Typical webcam FOV is ~60-70 degrees
            # f = frame_height / (2 * tan(FOV/2))
            import math
            fov_degrees = 65  # Typical webcam FOV
            fov_radians = math.radians(fov_degrees)
            self.focal_length = frame_height / (2 * math.tan(fov_radians / 2))
    
    def estimate_depth(
        self,
        bbox_height: int,
        frame_height: int = None
    ) -> float:
        """
        Estimate depth from bounding box height.
        
        Args:
            bbox_height: Height of person bounding box in pixels
            frame_height: Current frame height (for dynamic adjustment)
            
        Returns:
            Estimated depth in meters
        """
        if bbox_height <= 0:
            return 0.0
        
        # Adjust focal length if frame height changed
        if frame_height and frame_height != self.frame_height:
            scale = frame_height / self.frame_height
            adjusted_focal = self.focal_length * scale
        else:
            adjusted_focal = self.focal_length
        
        # Calculate depth: Z = (f * H_real) / h_pixel
        depth = (adjusted_focal * self.reference_height) / bbox_height
        
        # Clamp to reasonable range (0.5m to 20m)
        depth = max(0.5, min(20.0, depth))
        
        return depth
    
    def estimate_3d_position(
        self,
        center_x: int,
        center_y: int,
        bbox_height: int,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float, float]:
        """
        Estimate 3D position (X, Y, Z) in camera coordinate system.
        
        Args:
            center_x: Center X in pixels
            center_y: Center Y in pixels
            bbox_height: Bounding box height in pixels
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            (X, Y, Z) in meters where:
            - X: horizontal distance (positive = right)
            - Y: vertical distance (positive = down)
            - Z: depth distance (positive = forward)
        """
        # Estimate depth
        z = self.estimate_depth(bbox_height, frame_height)
        
        # Calculate X and Y using similar triangles
        # X = (center_x - frame_width/2) * Z / focal_length
        # Y = (center_y - frame_height/2) * Z / focal_length
        
        # Adjust focal length for current frame
        scale = frame_height / self.frame_height if self.frame_height else 1.0
        adjusted_focal = self.focal_length * scale
        
        # Center of frame
        cx = frame_width / 2
        cy = frame_height / 2
        
        # Calculate X and Y in meters
        x = (center_x - cx) * z / adjusted_focal
        y = (center_y - cy) * z / adjusted_focal
        
        return (round(x, 2), round(y, 2), round(z, 2))
    
    def calibrate(self, known_distance: float, bbox_height: int, frame_height: int = None):
        """
        Calibrate focal length using a known distance.
        
        Args:
            known_distance: Known distance to person in meters
            bbox_height: Bounding box height at that distance
            frame_height: Frame height during calibration
        """
        if frame_height and frame_height != self.frame_height:
            self.frame_height = frame_height
        
        # Solve for focal length: f = (Z * h_pixel) / H_real
        self.focal_length = (known_distance * bbox_height) / self.reference_height


class PersonDetectorOpenCV:
    """
    Person detection using OpenCV DNN with YOLOv4-tiny.
    This is a fallback when PyTorch/ultralytics is not available.
    """
    
    # COCO class ID for person
    PERSON_CLASS_ID = 0
    
    # Model URLs
    YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
    YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        enable_depth: bool = True
    ):
        """
        Initialize OpenCV-based person detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            enable_depth: Enable depth estimation
        """
        self.logger = get_logger("PersonDetectorOpenCV")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.enable_depth = enable_depth
        
        self._net = None
        self._classes = []
        self._output_layers = []
        self._is_loaded = False
        self._last_detections: List[Detection] = []
        self._inference_time: float = 0.0
        
        # Depth estimator
        self._depth_estimator = DepthEstimator() if enable_depth else None
        
        # Model directory
        self._model_dir = Path(__file__).parent / "models"
        self._model_dir.mkdir(exist_ok=True)
        
        # Colors for visualization
        self._box_color = (0, 255, 0)  # Green
        self._text_color = (255, 255, 255)  # White
        self._center_color = (0, 0, 255)  # Red
        self._depth_color = (255, 165, 0)  # Orange for depth info
    
    def _download_file(self, url: str, filepath: Path) -> bool:
        """Download a file from URL."""
        try:
            self.logger.info(f"Downloading {filepath.name}...")
            urllib.request.urlretrieve(url, str(filepath))
            self.logger.info(f"Downloaded {filepath.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load the YOLO model using OpenCV DNN.
        
        Returns:
            True if model loaded successfully
        """
        try:
            weights_path = self._model_dir / "yolov4-tiny.weights"
            cfg_path = self._model_dir / "yolov4-tiny.cfg"
            names_path = self._model_dir / "coco.names"
            
            # Download model files if not exist
            if not weights_path.exists():
                if not self._download_file(self.YOLO_WEIGHTS_URL, weights_path):
                    return False
            
            if not cfg_path.exists():
                if not self._download_file(self.YOLO_CFG_URL, cfg_path):
                    return False
            
            if not names_path.exists():
                if not self._download_file(self.COCO_NAMES_URL, names_path):
                    return False
            
            # Load class names
            with open(names_path, 'r') as f:
                self._classes = [line.strip() for line in f.readlines()]
            
            # Load network
            self.logger.info("Loading YOLO model with OpenCV DNN...")
            self._net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
            
            # Try to use CUDA if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.logger.info("Using CUDA backend")
            else:
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.logger.info("Using CPU backend")
            
            # Get output layer names
            layer_names = self._net.getLayerNames()
            self._output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
            
            self._is_loaded = True
            self.logger.info("YOLO model loaded successfully (OpenCV DNN)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect people in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects for detected people
        """
        if not self._is_loaded:
            if not self.load_model():
                return []
        
        if frame is None:
            return []
        
        detections = []
        start_time = time.time()
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416), 
                swapRB=True, crop=False
            )
            
            # Run forward pass
            self._net.setInput(blob)
            outputs = self._net.forward(self._output_layers)
            
            # Process outputs
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Only detect persons (class_id = 0)
                    if class_id == self.PERSON_CLASS_ID and confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, 
                self.confidence_threshold, 
                self.nms_threshold
            )
            
            # Create Detection objects
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(width, x + w), min(height, y + h)
                    bbox_height = y2 - y1
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Estimate depth if enabled
                    depth_z = 0.0
                    center_3d = (0.0, 0.0, 0.0)
                    if self.enable_depth and self._depth_estimator:
                        center_3d = self._depth_estimator.estimate_3d_position(
                            center_x, center_y, bbox_height, width, height
                        )
                        depth_z = center_3d[2]
                    
                    detection = Detection(
                        id=-1,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        center_3d=center_3d,
                        depth_z=depth_z,
                        confidence=confidences[i],
                        class_id=class_ids[i],
                        class_name="person"
                    )
                    
                    detections.append(detection)
            
            self._inference_time = time.time() - start_time
            self._last_detections = detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        draw_bbox: bool = True,
        draw_center: bool = True,
        draw_label: bool = True,
        draw_coordinates: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            detections: List of detections (uses last detections if None)
            draw_bbox: Draw bounding boxes
            draw_center: Draw center points
            draw_label: Draw labels with confidence
            draw_coordinates: Draw coordinate text
            
        Returns:
            Frame with drawn detections
        """
        if frame is None:
            return frame
        
        if detections is None:
            detections = self._last_detections
        
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center
            
            # Draw bounding box
            if draw_bbox:
                cv2.rectangle(output, (x1, y1), (x2, y2), self._box_color, 2)
            
            # Draw center point
            if draw_center:
                cv2.circle(output, (cx, cy), 5, self._center_color, -1)
                cv2.circle(output, (cx, cy), 8, self._center_color, 2)
            
            # Draw label with depth info
            if draw_label:
                if det.id >= 0:
                    label = f"Person #{det.id} ({det.confidence:.2f})"
                else:
                    label = f"Person ({det.confidence:.2f})"
                
                # Add depth to label if available
                if det.depth_z > 0:
                    label += f" Z:{det.depth_z:.1f}m"
                
                # Label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    output,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    self._box_color,
                    -1
                )
                cv2.putText(
                    output, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self._text_color, 2
                )
            
            # Draw 3D coordinates
            if draw_coordinates:
                # 2D pixel coordinates
                coord_text = f"({cx}, {cy})"
                cv2.putText(
                    output, coord_text,
                    (cx - 30, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self._text_color, 2
                )
                
                # 3D coordinates (X, Y, Z in meters)
                if det.depth_z > 0:
                    x3d, y3d, z3d = det.center_3d
                    coord_3d_text = f"3D:({x3d:.1f}, {y3d:.1f}, {z3d:.1f})m"
                    cv2.putText(
                        output, coord_3d_text,
                        (cx - 60, cy + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, self._depth_color, 2
                    )
        
        return output
    
    def get_coordinates(self, detections: Optional[List[Detection]] = None) -> List[Dict]:
        """
        Get coordinates of all detected people including 3D position.
        
        Args:
            detections: List of detections (uses last detections if None)
            
        Returns:
            List of coordinate dictionaries with 2D and 3D positions
        """
        if detections is None:
            detections = self._last_detections
        
        coordinates = []
        for det in detections:
            coordinates.append({
                "id": det.id,
                "center_x": det.center[0],
                "center_y": det.center[1],
                "center_3d": {
                    "x": det.center_3d[0],
                    "y": det.center_3d[1],
                    "z": det.center_3d[2]
                },
                "depth_z": round(det.depth_z, 2),
                "bbox": {
                    "x1": det.x1,
                    "y1": det.y1,
                    "x2": det.x2,
                    "y2": det.y2,
                    "width": det.width,
                    "height": det.height
                },
                "confidence": round(det.confidence, 3)
            })
        
        return coordinates
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set detection confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_colors(
        self,
        box_color: Tuple[int, int, int] = None,
        text_color: Tuple[int, int, int] = None,
        center_color: Tuple[int, int, int] = None
    ) -> None:
        """Set visualization colors (BGR format)."""
        if box_color:
            self._box_color = box_color
        if text_color:
            self._text_color = text_color
        if center_color:
            self._center_color = center_color
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def last_detections(self) -> List[Detection]:
        """Get last detection results."""
        return self._last_detections
    
    @property
    def inference_time(self) -> float:
        """Get last inference time in seconds."""
        return self._inference_time
    
    @property
    def inference_fps(self) -> float:
        """Get inference FPS."""
        if self._inference_time > 0:
            return 1.0 / self._inference_time
        return 0.0
    
    @property
    def person_count(self) -> int:
        """Get number of people detected in last frame."""
        return len(self._last_detections)


class PersonDetectorUltralytics:
    """
    Person detection using YOLOv8 via ultralytics.
    Requires torch and ultralytics packages.
    """
    
    # COCO class ID for person
    PERSON_CLASS_ID = 0
    
    # Available YOLO models (from smallest to largest)
    AVAILABLE_MODELS = {
        'yolov8n': 'yolov8n.pt',      # Nano - fastest
        'yolov8s': 'yolov8s.pt',      # Small
        'yolov8m': 'yolov8m.pt',      # Medium
        'yolov8l': 'yolov8l.pt',      # Large
        'yolov8x': 'yolov8x.pt',      # Extra large - most accurate
    }
    
    def __init__(
        self,
        model_name: str = 'yolov8n',
        confidence_threshold: float = 0.5,
        device: str = 'auto',
        enable_tracking: bool = False,
        enable_depth: bool = True
    ):
        """
        Initialize person detector with ultralytics.
        
        Args:
            model_name: YOLO model to use ('yolov8n', 'yolov8s', etc.)
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cpu', 'cuda', 'auto')
            enable_tracking: Enable object tracking
            enable_depth: Enable depth estimation
        """
        self.logger = get_logger("PersonDetectorUltralytics")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.enable_tracking = enable_tracking
        self.enable_depth = enable_depth
        
        self._model = None
        self._is_loaded = False
        self._last_detections: List[Detection] = []
        self._inference_time: float = 0.0
        
        # Depth estimator
        self._depth_estimator = DepthEstimator() if enable_depth else None
        
        # Colors for visualization
        self._box_color = (0, 255, 0)  # Green
        self._text_color = (255, 255, 255)  # White
        self._center_color = (0, 0, 255)  # Red
        self._depth_color = (255, 165, 0)  # Orange for depth info
    
    def load_model(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            True if model loaded successfully
        """
        try:
            from ultralytics import YOLO
            
            model_file = self.AVAILABLE_MODELS.get(self.model_name, 'yolov8n.pt')
            
            self.logger.info(f"Loading YOLO model: {model_file}")
            
            self._model = YOLO(model_file)
            
            # Set device
            if self.device == 'auto':
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.logger.info(f"Model loaded on device: {self.device}")
            self._is_loaded = True
            
            return True
            
        except ImportError:
            self.logger.error(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect people in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects for detected people
        """
        if not self._is_loaded:
            if not self.load_model():
                return []
        
        if frame is None:
            return []
        
        detections = []
        start_time = time.time()
        
        try:
            # Run inference
            if self.enable_tracking:
                results = self._model.track(
                    frame,
                    classes=[self.PERSON_CLASS_ID],  # Only detect people
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False,
                    persist=True  # Keep track IDs between frames
                )
            else:
                results = self._model(
                    frame,
                    classes=[self.PERSON_CLASS_ID],  # Only detect people
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False
                )
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for i, box in enumerate(boxes):
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Get confidence
                    conf = float(box.conf[0])
                    
                    # Get class ID
                    class_id = int(box.cls[0])
                    
                    # Get tracking ID if available
                    track_id = -1
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    
                    # Calculate center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Estimate depth if enabled
                    depth_z = 0.0
                    center_3d = (0.0, 0.0, 0.0)
                    bbox_height = y2 - y1
                    if self.enable_depth and self._depth_estimator:
                        height, width = frame.shape[:2]
                        center_3d = self._depth_estimator.estimate_3d_position(
                            center_x, center_y, bbox_height, width, height
                        )
                        depth_z = center_3d[2]
                    
                    detection = Detection(
                        id=track_id,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        center_3d=center_3d,
                        depth_z=depth_z,
                        confidence=conf,
                        class_id=class_id,
                        class_name="person"
                    )
                    
                    detections.append(detection)
            
            self._inference_time = time.time() - start_time
            self._last_detections = detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        draw_bbox: bool = True,
        draw_center: bool = True,
        draw_label: bool = True,
        draw_coordinates: bool = True
    ) -> np.ndarray:
        """Draw detection results on frame."""
        if frame is None:
            return frame
        
        if detections is None:
            detections = self._last_detections
        
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center
            
            if draw_bbox:
                cv2.rectangle(output, (x1, y1), (x2, y2), self._box_color, 2)
            
            if draw_center:
                cv2.circle(output, (cx, cy), 5, self._center_color, -1)
                cv2.circle(output, (cx, cy), 8, self._center_color, 2)
            
            if draw_label:
                if det.id >= 0:
                    label = f"Person #{det.id} ({det.confidence:.2f})"
                else:
                    label = f"Person ({det.confidence:.2f})"
                
                # Add depth to label if available
                if det.depth_z > 0:
                    label += f" Z:{det.depth_z:.1f}m"
                
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    output,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    self._box_color,
                    -1
                )
                cv2.putText(
                    output, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self._text_color, 2
                )
            
            if draw_coordinates:
                # 2D pixel coordinates
                coord_text = f"({cx}, {cy})"
                cv2.putText(
                    output, coord_text,
                    (cx - 30, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self._text_color, 2
                )
                
                # 3D coordinates (X, Y, Z in meters)
                if det.depth_z > 0:
                    x3d, y3d, z3d = det.center_3d
                    coord_3d_text = f"3D:({x3d:.1f}, {y3d:.1f}, {z3d:.1f})m"
                    cv2.putText(
                        output, coord_3d_text,
                        (cx - 60, cy + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, self._depth_color, 2
                    )
        
        return output
    
    def get_coordinates(self, detections: Optional[List[Detection]] = None) -> List[Dict]:
        """Get coordinates of all detected people including 3D position."""
        if detections is None:
            detections = self._last_detections
        
        coordinates = []
        for det in detections:
            coordinates.append({
                "id": det.id,
                "center_x": det.center[0],
                "center_y": det.center[1],
                "center_3d": {
                    "x": det.center_3d[0],
                    "y": det.center_3d[1],
                    "z": det.center_3d[2]
                },
                "depth_z": round(det.depth_z, 2),
                "bbox": {
                    "x1": det.x1,
                    "y1": det.y1,
                    "x2": det.x2,
                    "y2": det.y2,
                    "width": det.width,
                    "height": det.height
                },
                "confidence": round(det.confidence, 3)
            })
        
        return coordinates
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set detection confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_colors(
        self,
        box_color: Tuple[int, int, int] = None,
        text_color: Tuple[int, int, int] = None,
        center_color: Tuple[int, int, int] = None
    ) -> None:
        """Set visualization colors (BGR format)."""
        if box_color:
            self._box_color = box_color
        if text_color:
            self._text_color = text_color
        if center_color:
            self._center_color = center_color
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def last_detections(self) -> List[Detection]:
        return self._last_detections
    
    @property
    def inference_time(self) -> float:
        return self._inference_time
    
    @property
    def inference_fps(self) -> float:
        if self._inference_time > 0:
            return 1.0 / self._inference_time
        return 0.0
    
    @property
    def person_count(self) -> int:
        return len(self._last_detections)


def PersonDetector(
    model_name: str = 'yolov8n',
    confidence_threshold: float = 0.5,
    use_opencv_fallback: bool = True,
    **kwargs
):
    """
    Factory function to create the best available person detector.
    
    Tries ultralytics (YOLOv8) first, falls back to OpenCV DNN (YOLOv4-tiny).
    
    Args:
        model_name: YOLO model name (for ultralytics)
        confidence_threshold: Detection confidence threshold
        use_opencv_fallback: Whether to use OpenCV DNN as fallback
        **kwargs: Additional arguments for the detector
        
    Returns:
        PersonDetector instance
    """
    logger = get_logger("PersonDetector")
    
    # Try ultralytics first
    try:
        import torch
        from ultralytics import YOLO
        
        logger.info("Using ultralytics (YOLOv8) backend")
        return PersonDetectorUltralytics(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
    except ImportError:
        logger.warning("ultralytics or torch not available")
    except Exception as e:
        logger.warning(f"Failed to initialize ultralytics: {e}")
    
    # Fallback to OpenCV DNN
    if use_opencv_fallback:
        logger.info("Falling back to OpenCV DNN (YOLOv4-tiny) backend")
        return PersonDetectorOpenCV(
            confidence_threshold=confidence_threshold
        )
    
    raise RuntimeError("No person detection backend available")


class PersonDetectorAsync:
    """
    Asynchronous person detector that runs in a separate thread.
    Useful for maintaining high FPS while doing detection.
    """
    
    def __init__(self, detector):
        """
        Initialize async detector.
        
        Args:
            detector: PersonDetector instance to use
        """
        from threading import Thread, Lock, Event
        from queue import Queue
        
        self.detector = detector
        self._frame_queue: Queue = Queue(maxsize=2)
        self._result_lock = Lock()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._last_detections: List[Detection] = []
        self._is_running = False
    
    def start(self) -> None:
        """Start the detection thread."""
        from threading import Thread
        
        if self._is_running:
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        self._is_running = True
    
    def stop(self) -> None:
        """Stop the detection thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._is_running = False
    
    def submit_frame(self, frame: np.ndarray) -> None:
        """Submit a frame for detection (non-blocking)."""
        if not self._is_running:
            return
        
        try:
            # Clear old frames if queue is full
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except:
                    break
            
            self._frame_queue.put_nowait(frame.copy())
        except:
            pass
    
    def get_detections(self) -> List[Detection]:
        """Get the latest detection results."""
        with self._result_lock:
            return self._last_detections.copy()
    
    def _detection_loop(self) -> None:
        """Internal detection loop."""
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
                detections = self.detector.detect(frame)
                
                with self._result_lock:
                    self._last_detections = detections
                    
            except:
                continue
    
    @property
    def is_running(self) -> bool:
        return self._is_running
