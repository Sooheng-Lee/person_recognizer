"""
Camera controller module for USB Camera Viewer
Handles image processing and camera settings adjustments
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import get_config


class CameraController:
    """
    Controls camera settings and image processing.
    
    Features:
    - Brightness, contrast, saturation adjustment
    - Image rotation and flipping
    - Screenshot capture
    - Video recording
    """
    
    def __init__(self):
        """Initialize camera controller."""
        self.logger = get_logger("CameraController")
        self.config = get_config()
        
        # Image adjustment settings (0-100 range, 50 is neutral)
        self._brightness: int = 50
        self._contrast: int = 50
        self._saturation: int = 50
        
        # Transform settings
        self._rotation: int = 0  # 0, 90, 180, 270
        self._flip_horizontal: bool = False
        self._flip_vertical: bool = False
        self._zoom: float = 1.0
        
        # Recording state
        self._is_recording: bool = False
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._recording_start_time: Optional[datetime] = None
        self._recording_frame_count: int = 0
        
        # Load settings from config
        self._load_settings()
    
    def _load_settings(self) -> None:
        """Load settings from configuration."""
        self._brightness = self.config.get("brightness", 50)
        self._contrast = self.config.get("contrast", 50)
        self._saturation = self.config.get("saturation", 50)
        self._rotation = self.config.get("rotation", 0)
        self._flip_horizontal = self.config.get("flip_horizontal", False)
        self._flip_vertical = self.config.get("flip_vertical", False)
    
    def _save_settings(self) -> None:
        """Save current settings to configuration."""
        self.config.update({
            "brightness": self._brightness,
            "contrast": self._contrast,
            "saturation": self._saturation,
            "rotation": self._rotation,
            "flip_horizontal": self._flip_horizontal,
            "flip_vertical": self._flip_vertical
        })
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all image processing to a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Processed frame
        """
        if frame is None:
            return frame
        
        processed = frame.copy()
        
        # Apply brightness and contrast
        if self._brightness != 50 or self._contrast != 50:
            processed = self._apply_brightness_contrast(processed)
        
        # Apply saturation
        if self._saturation != 50:
            processed = self._apply_saturation(processed)
        
        # Apply rotation
        if self._rotation != 0:
            processed = self._apply_rotation(processed)
        
        # Apply flipping
        if self._flip_horizontal or self._flip_vertical:
            processed = self._apply_flip(processed)
        
        # Apply zoom
        if self._zoom != 1.0:
            processed = self._apply_zoom(processed)
        
        return processed
    
    def _apply_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply brightness and contrast adjustments.
        
        Args:
            frame: Input frame
            
        Returns:
            Adjusted frame
        """
        # Convert 0-100 range to actual values
        # Brightness: -127 to 127 (50 = 0)
        # Contrast: 0.5 to 1.5 (50 = 1.0)
        brightness = (self._brightness - 50) * 2.54  # -127 to 127
        contrast = 0.5 + (self._contrast / 100.0)  # 0.5 to 1.5
        
        # Apply using linear transformation
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        return adjusted
    
    def _apply_saturation(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply saturation adjustment.
        
        Args:
            frame: Input frame
            
        Returns:
            Adjusted frame
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation (0-100 range, 50 is neutral)
        saturation_factor = self._saturation / 50.0  # 0 to 2
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply rotation.
        
        Args:
            frame: Input frame
            
        Returns:
            Rotated frame
        """
        if self._rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self._rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self._rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def _apply_flip(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply horizontal/vertical flipping.
        
        Args:
            frame: Input frame
            
        Returns:
            Flipped frame
        """
        if self._flip_horizontal and self._flip_vertical:
            return cv2.flip(frame, -1)  # Both
        elif self._flip_horizontal:
            return cv2.flip(frame, 1)  # Horizontal
        elif self._flip_vertical:
            return cv2.flip(frame, 0)  # Vertical
        return frame
    
    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply zoom effect (crop and resize).
        
        Args:
            frame: Input frame
            
        Returns:
            Zoomed frame
        """
        if self._zoom <= 1.0:
            return frame
        
        h, w = frame.shape[:2]
        
        # Calculate crop area
        crop_w = int(w / self._zoom)
        crop_h = int(h / self._zoom)
        
        # Center crop
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # Crop and resize back to original size
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed
    
    # ===== Property Setters =====
    
    def set_brightness(self, value: int) -> None:
        """Set brightness (0-100, 50 is neutral)."""
        self._brightness = max(0, min(100, value))
        self._save_settings()
    
    def set_contrast(self, value: int) -> None:
        """Set contrast (0-100, 50 is neutral)."""
        self._contrast = max(0, min(100, value))
        self._save_settings()
    
    def set_saturation(self, value: int) -> None:
        """Set saturation (0-100, 50 is neutral)."""
        self._saturation = max(0, min(100, value))
        self._save_settings()
    
    def set_rotation(self, degrees: int) -> None:
        """Set rotation (0, 90, 180, or 270 degrees)."""
        if degrees in [0, 90, 180, 270]:
            self._rotation = degrees
            self._save_settings()
    
    def rotate_clockwise(self) -> None:
        """Rotate 90 degrees clockwise."""
        self._rotation = (self._rotation + 90) % 360
        self._save_settings()
    
    def rotate_counterclockwise(self) -> None:
        """Rotate 90 degrees counterclockwise."""
        self._rotation = (self._rotation - 90) % 360
        self._save_settings()
    
    def set_flip_horizontal(self, flip: bool) -> None:
        """Set horizontal flip."""
        self._flip_horizontal = flip
        self._save_settings()
    
    def set_flip_vertical(self, flip: bool) -> None:
        """Set vertical flip."""
        self._flip_vertical = flip
        self._save_settings()
    
    def toggle_flip_horizontal(self) -> bool:
        """Toggle horizontal flip and return new state."""
        self._flip_horizontal = not self._flip_horizontal
        self._save_settings()
        return self._flip_horizontal
    
    def toggle_flip_vertical(self) -> bool:
        """Toggle vertical flip and return new state."""
        self._flip_vertical = not self._flip_vertical
        self._save_settings()
        return self._flip_vertical
    
    def set_zoom(self, zoom: float) -> None:
        """Set zoom level (1.0 to 4.0)."""
        self._zoom = max(1.0, min(4.0, zoom))
    
    def reset_adjustments(self) -> None:
        """Reset all adjustments to defaults."""
        self._brightness = 50
        self._contrast = 50
        self._saturation = 50
        self._rotation = 0
        self._flip_horizontal = False
        self._flip_vertical = False
        self._zoom = 1.0
        self._save_settings()
        self.logger.info("All adjustments reset to defaults")
    
    # ===== Screenshot =====
    
    def capture_screenshot(
        self, 
        frame: np.ndarray, 
        save_path: Optional[str] = None,
        format: str = "png"
    ) -> Optional[str]:
        """
        Save a screenshot of the current frame.
        
        Args:
            frame: Frame to save
            save_path: Custom save path (optional)
            format: Image format (png, jpg)
            
        Returns:
            Path to saved file or None if failed
        """
        if frame is None:
            self.logger.error("No frame to capture")
            return None
        
        try:
            # Determine save path
            if save_path is None:
                save_dir = self.config.save_path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.{format}"
                save_path = save_dir / filename
            else:
                save_path = Path(save_path)
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            success = cv2.imwrite(str(save_path), frame)
            
            if success:
                self.logger.info(f"Screenshot saved: {save_path}")
                return str(save_path)
            else:
                self.logger.error(f"Failed to save screenshot: {save_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Screenshot error: {e}")
            return None
    
    # ===== Video Recording =====
    
    def start_recording(
        self, 
        resolution: Tuple[int, int],
        fps: int = 30,
        save_path: Optional[str] = None,
        codec: str = "mp4v"
    ) -> bool:
        """
        Start video recording.
        
        Args:
            resolution: Video resolution (width, height)
            fps: Frames per second
            save_path: Custom save path (optional)
            codec: Video codec (mp4v, XVID, etc.)
            
        Returns:
            True if recording started successfully
        """
        if self._is_recording:
            self.logger.warning("Already recording")
            return False
        
        try:
            # Determine save path
            if save_path is None:
                save_dir = self.config.save_path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.mp4"
                save_path = save_dir / filename
            else:
                save_path = Path(save_path)
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self._video_writer = cv2.VideoWriter(
                str(save_path),
                fourcc,
                fps,
                resolution
            )
            
            if not self._video_writer.isOpened():
                self.logger.error("Failed to create video writer")
                return False
            
            self._is_recording = True
            self._recording_start_time = datetime.now()
            self._recording_frame_count = 0
            self._recording_path = save_path
            
            self.logger.info(f"Recording started: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Recording start error: {e}")
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the recording.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if frame written successfully
        """
        if not self._is_recording or self._video_writer is None:
            return False
        
        try:
            self._video_writer.write(frame)
            self._recording_frame_count += 1
            return True
        except Exception as e:
            self.logger.error(f"Frame write error: {e}")
            return False
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop video recording.
        
        Returns:
            Path to saved recording or None
        """
        if not self._is_recording:
            return None
        
        try:
            self._video_writer.release()
            self._video_writer = None
            self._is_recording = False
            
            duration = (datetime.now() - self._recording_start_time).total_seconds()
            
            self.logger.info(
                f"Recording stopped: {self._recording_frame_count} frames, "
                f"{duration:.1f} seconds"
            )
            
            return str(self._recording_path)
            
        except Exception as e:
            self.logger.error(f"Recording stop error: {e}")
            return None
    
    # ===== Properties =====
    
    @property
    def brightness(self) -> int:
        """Get current brightness."""
        return self._brightness
    
    @property
    def contrast(self) -> int:
        """Get current contrast."""
        return self._contrast
    
    @property
    def saturation(self) -> int:
        """Get current saturation."""
        return self._saturation
    
    @property
    def rotation(self) -> int:
        """Get current rotation."""
        return self._rotation
    
    @property
    def flip_horizontal(self) -> bool:
        """Get horizontal flip state."""
        return self._flip_horizontal
    
    @property
    def flip_vertical(self) -> bool:
        """Get vertical flip state."""
        return self._flip_vertical
    
    @property
    def zoom(self) -> float:
        """Get current zoom level."""
        return self._zoom
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    @property
    def recording_duration(self) -> float:
        """Get recording duration in seconds."""
        if self._is_recording and self._recording_start_time:
            return (datetime.now() - self._recording_start_time).total_seconds()
        return 0.0
    
    @property
    def recording_frames(self) -> int:
        """Get number of recorded frames."""
        return self._recording_frame_count
