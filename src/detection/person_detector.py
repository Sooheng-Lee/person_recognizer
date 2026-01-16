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
    Represents a single person detection.
    
    Attributes:
        id: Tracking ID (if tracking enabled)
        bbox: Bounding box [x1, y1, x2, y2]
        center: Center point (x, y)
        confidence: Detection confidence score
        class_id: Class ID (0 for person in COCO)
        class_name: Class name
    """
    id: int = -1
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    center: Tuple[int, int] = (0, 0)
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
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "width": self.width,
            "height": self.height,
            "area": self.area
        }
    
    def __str__(self) -> str:
        return f"Person(id={self.id}, center={self.center}, conf={self.confidence:.2f})"


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
        nms_threshold: float = 0.4
    ):
        """
        Initialize OpenCV-based person detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.logger = get_logger("PersonDetectorOpenCV")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self._net = None
        self._classes = []
        self._output_layers = []
        self._is_loaded = False
        self._last_detections: List[Detection] = []
        self._inference_time: float = 0.0
        
        # Model directory
        self._model_dir = Path(__file__).parent / "models"
        self._model_dir.mkdir(exist_ok=True)
        
        # Colors for visualization
        self._box_color = (0, 255, 0)  # Green
        self._text_color = (255, 255, 255)  # White
        self._center_color = (0, 0, 255)  # Red
    
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
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detection = Detection(
                        id=-1,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
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
            
            # Draw label
            if draw_label:
                if det.id >= 0:
                    label = f"Person #{det.id} ({det.confidence:.2f})"
                else:
                    label = f"Person ({det.confidence:.2f})"
                
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
            
            # Draw coordinates
            if draw_coordinates:
                coord_text = f"({cx}, {cy})"
                cv2.putText(
                    output, coord_text,
                    (cx - 30, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self._text_color, 2
                )
        
        return output
    
    def get_coordinates(self, detections: Optional[List[Detection]] = None) -> List[Dict]:
        """
        Get coordinates of all detected people.
        
        Args:
            detections: List of detections (uses last detections if None)
            
        Returns:
            List of coordinate dictionaries
        """
        if detections is None:
            detections = self._last_detections
        
        coordinates = []
        for det in detections:
            coordinates.append({
                "id": det.id,
                "center_x": det.center[0],
                "center_y": det.center[1],
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
        enable_tracking: bool = False
    ):
        """
        Initialize person detector with ultralytics.
        
        Args:
            model_name: YOLO model to use ('yolov8n', 'yolov8s', etc.)
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cpu', 'cuda', 'auto')
            enable_tracking: Enable object tracking
        """
        self.logger = get_logger("PersonDetectorUltralytics")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.enable_tracking = enable_tracking
        
        self._model = None
        self._is_loaded = False
        self._last_detections: List[Detection] = []
        self._inference_time: float = 0.0
        
        # Colors for visualization
        self._box_color = (0, 255, 0)  # Green
        self._text_color = (255, 255, 255)  # White
        self._center_color = (0, 0, 255)  # Red
    
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
                    
                    detection = Detection(
                        id=track_id,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
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
                coord_text = f"({cx}, {cy})"
                cv2.putText(
                    output, coord_text,
                    (cx - 30, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self._text_color, 2
                )
        
        return output
    
    def get_coordinates(self, detections: Optional[List[Detection]] = None) -> List[Dict]:
        """Get coordinates of all detected people."""
        if detections is None:
            detections = self._last_detections
        
        coordinates = []
        for det in detections:
            coordinates.append({
                "id": det.id,
                "center_x": det.center[0],
                "center_y": det.center[1],
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
