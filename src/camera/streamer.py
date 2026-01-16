"""
Camera streaming module for USB Camera Viewer
Handles real-time video capture and frame delivery
"""

import cv2
import numpy as np
from threading import Thread, Event, Lock
from typing import Optional, Callable, Tuple
from queue import Queue, Empty
import time

from ..utils.logger import get_logger
from .detector import CameraDevice


class CameraStreamer:
    """
    Handles video streaming from a camera device.
    
    Features:
    - Threaded video capture for smooth performance
    - Frame queue for asynchronous processing
    - FPS calculation and monitoring
    - Resolution and FPS configuration
    """
    
    # OpenCV backend for Windows
    BACKEND = cv2.CAP_DSHOW
    
    def __init__(
        self,
        device: Optional[CameraDevice] = None,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        buffer_size: int = 2
    ):
        """
        Initialize camera streamer.
        
        Args:
            device: CameraDevice to stream from
            resolution: Desired resolution (width, height)
            fps: Desired frames per second
            buffer_size: Frame buffer size for queue
        """
        self.logger = get_logger("CameraStreamer")
        
        self._device = device
        self._resolution = resolution
        self._target_fps = fps
        self._buffer_size = buffer_size
        
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._frame_lock = Lock()
        
        self._current_frame: Optional[np.ndarray] = None
        self._frame_queue: Queue = Queue(maxsize=buffer_size)
        
        # Statistics
        self._actual_fps: float = 0.0
        self._frame_count: int = 0
        self._last_fps_time: float = 0.0
        self._fps_frame_count: int = 0
        
        # Callbacks
        self._on_frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._on_error_callback: Optional[Callable[[str], None]] = None
        
        # State
        self._is_streaming = False
        self._is_connected = False
    
    def connect(self, device: Optional[CameraDevice] = None) -> bool:
        """
        Connect to a camera device.
        
        Args:
            device: CameraDevice to connect to (optional, uses stored device)
            
        Returns:
            True if connection successful, False otherwise
        """
        if device:
            self._device = device
        
        if not self._device:
            self.logger.error("No device specified")
            return False
        
        try:
            self.logger.info(f"Connecting to device: {self._device.name}")
            
            # Release existing capture if any
            if self._capture:
                self._capture.release()
            
            # Open video capture
            self._capture = cv2.VideoCapture(self._device.index, self.BACKEND)
            
            if not self._capture.isOpened():
                self.logger.error(f"Failed to open device {self._device.index}")
                return False
            
            # Set resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
            
            # Set FPS
            self._capture.set(cv2.CAP_PROP_FPS, self._target_fps)
            
            # Set buffer size (reduces latency)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(
                f"Connected: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
            )
            
            self._is_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            if self._on_error_callback:
                self._on_error_callback(str(e))
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the camera device."""
        self.stop()
        
        if self._capture:
            self._capture.release()
            self._capture = None
        
        self._is_connected = False
        self.logger.info("Disconnected from device")
    
    def start(self) -> bool:
        """
        Start video streaming.
        
        Returns:
            True if streaming started successfully
        """
        if self._is_streaming:
            self.logger.warning("Already streaming")
            return True
        
        if not self._is_connected:
            if not self.connect():
                return False
        
        self._stop_event.clear()
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
        # Start capture thread
        self._capture_thread = Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture"
        )
        self._capture_thread.start()
        
        self._is_streaming = True
        self.logger.info("Streaming started")
        return True
    
    def stop(self) -> None:
        """Stop video streaming."""
        if not self._is_streaming:
            return
        
        self._stop_event.set()
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        
        # Clear frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
        
        self._is_streaming = False
        self.logger.info("Streaming stopped")
    
    def _capture_loop(self) -> None:
        """Internal capture loop running in separate thread."""
        frame_interval = 1.0 / self._target_fps
        last_frame_time = time.time()
        
        while not self._stop_event.is_set():
            if not self._capture or not self._capture.isOpened():
                self.logger.error("Capture device lost")
                if self._on_error_callback:
                    self._on_error_callback("Camera disconnected")
                break
            
            # Read frame
            ret, frame = self._capture.read()
            
            if not ret:
                self.logger.warning("Failed to read frame")
                time.sleep(0.01)
                continue
            
            current_time = time.time()
            
            # Update current frame (thread-safe)
            with self._frame_lock:
                self._current_frame = frame.copy()
            
            # Add to queue (non-blocking)
            try:
                if self._frame_queue.full():
                    self._frame_queue.get_nowait()  # Remove old frame
                self._frame_queue.put_nowait(frame)
            except Exception:
                pass
            
            # Call frame callback
            if self._on_frame_callback:
                try:
                    self._on_frame_callback(frame)
                except Exception as e:
                    self.logger.error(f"Frame callback error: {e}")
            
            # Update statistics
            self._frame_count += 1
            self._fps_frame_count += 1
            
            # Calculate FPS every second
            elapsed = current_time - self._last_fps_time
            if elapsed >= 1.0:
                self._actual_fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_time = current_time
            
            # Frame rate limiting
            elapsed_since_last = current_time - last_frame_time
            if elapsed_since_last < frame_interval:
                sleep_time = frame_interval - elapsed_since_last
                time.sleep(sleep_time)
            
            last_frame_time = time.time()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame.
        
        Returns:
            Current frame as numpy array, or None if no frame available
        """
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None
    
    def get_frame_from_queue(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get a frame from the queue (blocking with timeout).
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Frame from queue or None if timeout
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Change the capture resolution.
        
        Args:
            width: New width
            height: New height
            
        Returns:
            True if resolution change successful
        """
        self._resolution = (width, height)
        
        if self._capture and self._capture.isOpened():
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if (actual_width, actual_height) == (width, height):
                self.logger.info(f"Resolution changed to {width}x{height}")
                return True
            else:
                self.logger.warning(
                    f"Resolution mismatch: requested {width}x{height}, "
                    f"got {actual_width}x{actual_height}"
                )
        
        return False
    
    def set_fps(self, fps: int) -> bool:
        """
        Change the target FPS.
        
        Args:
            fps: New target FPS
            
        Returns:
            True if FPS change successful
        """
        self._target_fps = fps
        
        if self._capture and self._capture.isOpened():
            self._capture.set(cv2.CAP_PROP_FPS, fps)
            self.logger.info(f"Target FPS set to {fps}")
            return True
        
        return False
    
    def set_frame_callback(
        self, 
        callback: Optional[Callable[[np.ndarray], None]]
    ) -> None:
        """
        Set callback function for new frames.
        
        Args:
            callback: Function to call with each new frame
        """
        self._on_frame_callback = callback
    
    def set_error_callback(
        self, 
        callback: Optional[Callable[[str], None]]
    ) -> None:
        """
        Set callback function for errors.
        
        Args:
            callback: Function to call on errors
        """
        self._on_error_callback = callback
    
    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to a device."""
        return self._is_connected
    
    @property
    def fps(self) -> float:
        """Get actual FPS."""
        return self._actual_fps
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution."""
        if self._capture and self._capture.isOpened():
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return self._resolution
    
    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
    
    @property
    def device(self) -> Optional[CameraDevice]:
        """Get current device."""
        return self._device
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
