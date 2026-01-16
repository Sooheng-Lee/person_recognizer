"""
Video display widget for USB Camera Viewer
Handles rendering of video frames in PyQt5
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont

from ..utils.logger import get_logger


class VideoWidget(QWidget):
    """
    Widget for displaying video frames.
    
    Features:
    - Smooth video rendering
    - Aspect ratio preservation
    - Double-click for fullscreen
    - Overlay text support (FPS, resolution)
    """
    
    # Signals
    double_clicked = pyqtSignal()
    frame_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize video widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.logger = get_logger("VideoWidget")
        
        # Setup UI
        self._setup_ui()
        
        # State
        self._current_frame: np.ndarray = None
        self._current_pixmap: QPixmap = None
        self._aspect_ratio: float = 16 / 9
        self._show_overlay: bool = True
        self._overlay_text: str = ""
        self._fps_text: str = ""
        self._resolution_text: str = ""
        
        # Placeholder for no video
        self._show_placeholder = True
        
    def _setup_ui(self) -> None:
        """Setup the widget UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Video display label
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        self._video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        
        layout.addWidget(self._video_label)
        
        # Set size policy for the widget
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        
        # Show placeholder initially
        self._update_placeholder()
    
    def _update_placeholder(self) -> None:
        """Show placeholder when no video is available."""
        # Create placeholder image
        width = max(self._video_label.width(), 640)
        height = max(self._video_label.height(), 480)
        
        placeholder = QPixmap(width, height)
        placeholder.fill(QColor(26, 26, 26))
        
        painter = QPainter(placeholder)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw text
        painter.setPen(QColor(128, 128, 128))
        font = QFont("Arial", 16)
        painter.setFont(font)
        painter.drawText(
            placeholder.rect(),
            Qt.AlignCenter,
            "No Camera Connected\n\nSelect a device and click Start"
        )
        
        painter.end()
        
        self._video_label.setPixmap(placeholder)
    
    def update_frame(self, frame: np.ndarray) -> None:
        """
        Update the displayed frame.
        
        Args:
            frame: OpenCV frame (BGR format)
        """
        if frame is None:
            return
        
        self._current_frame = frame
        self._show_placeholder = False
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        
        # Create QImage
        q_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # Convert to pixmap and scale to fit label while preserving aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        
        # Draw overlay if enabled
        if self._show_overlay and (self._fps_text or self._resolution_text):
            pixmap = self._draw_overlay(pixmap)
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(
            self._video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self._current_pixmap = scaled_pixmap
        self._video_label.setPixmap(scaled_pixmap)
        
        # Emit signal
        self.frame_updated.emit()
    
    def _draw_overlay(self, pixmap: QPixmap) -> QPixmap:
        """
        Draw overlay information on the pixmap.
        
        Args:
            pixmap: Source pixmap
            
        Returns:
            Pixmap with overlay drawn
        """
        # Create a copy to draw on
        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Semi-transparent background for text
        font = QFont("Consolas", 10)
        painter.setFont(font)
        
        # Draw FPS in top-left
        if self._fps_text:
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(10, 20, self._fps_text)
        
        # Draw resolution in top-right
        if self._resolution_text:
            painter.setPen(QColor(255, 255, 255))
            text_width = painter.fontMetrics().horizontalAdvance(self._resolution_text)
            painter.drawText(pixmap.width() - text_width - 10, 20, self._resolution_text)
        
        # Draw custom overlay text in bottom
        if self._overlay_text:
            painter.setPen(QColor(255, 255, 0))
            painter.drawText(10, pixmap.height() - 10, self._overlay_text)
        
        painter.end()
        return result
    
    def set_overlay_info(
        self, 
        fps: float = None, 
        resolution: tuple = None,
        custom_text: str = None
    ) -> None:
        """
        Set overlay information.
        
        Args:
            fps: Current FPS value
            resolution: Current resolution (width, height)
            custom_text: Custom text to display
        """
        if fps is not None:
            self._fps_text = f"FPS: {fps:.1f}"
        
        if resolution is not None:
            self._resolution_text = f"{resolution[0]}x{resolution[1]}"
        
        if custom_text is not None:
            self._overlay_text = custom_text
    
    def set_show_overlay(self, show: bool) -> None:
        """
        Enable or disable overlay display.
        
        Args:
            show: Whether to show overlay
        """
        self._show_overlay = show
    
    def clear(self) -> None:
        """Clear the video display and show placeholder."""
        self._current_frame = None
        self._current_pixmap = None
        self._show_placeholder = True
        self._fps_text = ""
        self._resolution_text = ""
        self._overlay_text = ""
        self._update_placeholder()
    
    def get_current_frame(self) -> np.ndarray:
        """
        Get the current frame.
        
        Returns:
            Current frame as numpy array or None
        """
        return self._current_frame.copy() if self._current_frame is not None else None
    
    def mouseDoubleClickEvent(self, event) -> None:
        """Handle double-click for fullscreen toggle."""
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)
    
    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        
        if self._show_placeholder:
            self._update_placeholder()
        elif self._current_pixmap:
            # Rescale current pixmap to new size
            pass
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(640, 480)
    
    def minimumSizeHint(self) -> QSize:
        """Return minimum size."""
        return QSize(320, 240)


class VideoOverlay(QWidget):
    """
    Transparent overlay widget for displaying information on top of video.
    """
    
    def __init__(self, parent=None):
        """Initialize overlay."""
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self._recording = False
        self._recording_time = ""
    
    def set_recording(self, recording: bool, time_str: str = "") -> None:
        """
        Set recording state.
        
        Args:
            recording: Whether recording is active
            time_str: Recording time string
        """
        self._recording = recording
        self._recording_time = time_str
        self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the overlay."""
        if not self._recording:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw recording indicator
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(15, 15, 12, 12)
        
        # Draw recording text
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(35, 24, f"REC {self._recording_time}")
        
        painter.end()
