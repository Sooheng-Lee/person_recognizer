"""
Camera device detection module for USB Camera Viewer
Detects and enumerates connected camera devices via USB
"""

import cv2
from dataclasses import dataclass
from typing import List, Optional, Callable
from threading import Thread, Event
import time

from ..utils.logger import get_logger


@dataclass
class CameraDevice:
    """
    Represents a detected camera device.
    
    Attributes:
        index: OpenCV camera index
        name: Device name
        width: Maximum supported width
        height: Maximum supported height
        fps: Maximum supported FPS
        backend: OpenCV backend used
    """
    index: int
    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = "DirectShow"
    
    def __str__(self) -> str:
        if self.width and self.height:
            return f"{self.name} ({self.width}x{self.height})"
        return self.name
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "index": self.index,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "backend": self.backend
        }


class CameraDetector:
    """
    Detects and monitors connected camera devices.
    
    Features:
    - Enumerate all connected cameras
    - Get camera capabilities (resolution, FPS)
    - Monitor for device connect/disconnect events
    """
    
    # OpenCV backend for Windows (DirectShow provides better device names)
    BACKEND = cv2.CAP_DSHOW
    
    def __init__(self, max_devices: int = 10):
        """
        Initialize camera detector.
        
        Args:
            max_devices: Maximum number of device indices to scan
        """
        self.logger = get_logger("CameraDetector")
        self.max_devices = max_devices
        self._devices: List[CameraDevice] = []
        self._monitor_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._on_device_change: Optional[Callable[[List[CameraDevice]], None]] = None
        
    def detect_devices(self) -> List[CameraDevice]:
        """
        Scan and detect all connected camera devices.
        
        Returns:
            List of detected CameraDevice objects
        """
        self.logger.info("Scanning for camera devices...")
        devices = []
        
        for index in range(self.max_devices):
            device = self._probe_device(index)
            if device:
                devices.append(device)
                self.logger.info(f"Found device: {device}")
        
        self._devices = devices
        self.logger.info(f"Total devices found: {len(devices)}")
        return devices
    
    def _probe_device(self, index: int) -> Optional[CameraDevice]:
        """
        Probe a specific device index to check if a camera exists.
        
        Args:
            index: Device index to probe
            
        Returns:
            CameraDevice if found, None otherwise
        """
        try:
            cap = cv2.VideoCapture(index, self.BACKEND)
            
            if not cap.isOpened():
                return None
            
            # Try to read a frame to confirm device works
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return None
            
            # Get device properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to get device name (may not always work)
            backend_name = cap.getBackendName()
            
            cap.release()
            
            # Create device name
            device_name = f"Camera {index}"
            
            return CameraDevice(
                index=index,
                name=device_name,
                width=width,
                height=height,
                fps=fps if fps > 0 else 30.0,
                backend=backend_name
            )
            
        except Exception as e:
            self.logger.debug(f"Error probing device {index}: {e}")
            return None
    
    def get_device(self, index: int) -> Optional[CameraDevice]:
        """
        Get a specific device by index.
        
        Args:
            index: Device index
            
        Returns:
            CameraDevice if found in cached devices, None otherwise
        """
        for device in self._devices:
            if device.index == index:
                return device
        return None
    
    def get_device_capabilities(self, index: int) -> dict:
        """
        Get detailed capabilities of a camera device.
        
        Args:
            index: Device index
            
        Returns:
            Dictionary with device capabilities
        """
        capabilities = {
            "resolutions": [],
            "fps_options": [],
            "supports_autofocus": False,
            "supports_exposure": False
        }
        
        try:
            cap = cv2.VideoCapture(index, self.BACKEND)
            if not cap.isOpened():
                return capabilities
            
            # Common resolutions to test
            test_resolutions = [
                (640, 480),
                (800, 600),
                (1280, 720),
                (1920, 1080),
                (2560, 1440),
                (3840, 2160)
            ]
            
            for width, height in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if (actual_width, actual_height) == (width, height):
                    if (width, height) not in capabilities["resolutions"]:
                        capabilities["resolutions"].append((width, height))
            
            # Test FPS options
            for fps in [15, 24, 30, 60]:
                cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                if abs(actual_fps - fps) < 1:
                    capabilities["fps_options"].append(fps)
            
            # Check autofocus support
            autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            capabilities["supports_autofocus"] = autofocus >= 0
            
            # Check exposure support
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            capabilities["supports_exposure"] = exposure != 0
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"Error getting capabilities for device {index}: {e}")
        
        return capabilities
    
    def start_monitoring(
        self, 
        callback: Callable[[List[CameraDevice]], None],
        interval: float = 2.0
    ) -> None:
        """
        Start monitoring for device changes.
        
        Args:
            callback: Function to call when devices change
            interval: Check interval in seconds
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.warning("Monitor already running")
            return
        
        self._on_device_change = callback
        self._stop_event.clear()
        
        self._monitor_thread = Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Device monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring for device changes."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None
        self.logger.info("Device monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """
        Internal monitoring loop that checks for device changes.
        
        Args:
            interval: Check interval in seconds
        """
        previous_indices = {d.index for d in self._devices}
        
        while not self._stop_event.is_set():
            time.sleep(interval)
            
            if self._stop_event.is_set():
                break
            
            # Detect current devices
            current_devices = self.detect_devices()
            current_indices = {d.index for d in current_devices}
            
            # Check for changes
            if current_indices != previous_indices:
                added = current_indices - previous_indices
                removed = previous_indices - current_indices
                
                if added:
                    self.logger.info(f"Devices connected: {added}")
                if removed:
                    self.logger.info(f"Devices disconnected: {removed}")
                
                if self._on_device_change:
                    self._on_device_change(current_devices)
                
                previous_indices = current_indices
    
    @property
    def devices(self) -> List[CameraDevice]:
        """Get list of currently detected devices."""
        return self._devices.copy()
    
    @property
    def device_count(self) -> int:
        """Get number of detected devices."""
        return len(self._devices)
    
    def refresh(self) -> List[CameraDevice]:
        """
        Refresh the device list.
        
        Returns:
            Updated list of devices
        """
        return self.detect_devices()


# Singleton instance for convenience
_detector_instance: Optional[CameraDetector] = None


def get_detector() -> CameraDetector:
    """Get the global CameraDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CameraDetector()
    return _detector_instance
