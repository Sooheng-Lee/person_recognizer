"""
Main window for USB Camera Viewer application
"""

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QMessageBox, QApplication,
    QShortcut, QLabel
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence, QCloseEvent

from ..utils.logger import get_logger, setup_logger
from ..utils.config import get_config
from ..camera.detector import CameraDetector, CameraDevice
from ..camera.streamer import CameraStreamer
from ..camera.controller import CameraController
from ..camera.phone_camera import PhoneCameraManager, PhoneCameraStreamer, PhoneDevice
from ..detection.person_detector import PersonDetector
from .video_widget import VideoWidget
from .controls import ControlPanel, SettingsDialog, QuickAdjustPanel
from .phone_dialog import PhoneCameraDialog


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Coordinates all components: video display, controls, camera modules.
    """
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        
        # Setup logging
        setup_logger()
        self.logger = get_logger("MainWindow")
        self.config = get_config()
        
        # Initialize components
        self._detector = CameraDetector()
        self._streamer = CameraStreamer()
        self._controller = CameraController()
        
        # Phone camera components
        self._phone_manager = PhoneCameraManager()
        self._phone_streamer = PhoneCameraStreamer()
        self._phone_device: PhoneDevice = None
        self._is_phone_camera = False
        
        # Person detection
        self._person_detector = None
        self._detection_enabled = False
        self._last_detections = []
        
        # State
        self._is_fullscreen = False
        self._selected_device: CameraDevice = None
        
        # Setup UI
        self._setup_window()
        self._setup_ui()
        self._setup_shortcuts()
        self._setup_timers()
        self._connect_signals()
        
        # Initial device scan
        self._refresh_devices()
        
        self.logger.info("Application started")
    
    def _setup_window(self) -> None:
        """Configure main window properties."""
        self.setWindowTitle("USB Camera Viewer")
        
        # Set window size from config
        width, height = self.config.window_size
        self.resize(width, height)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.move(x, y)
        
        # Set minimum size
        self.setMinimumSize(800, 600)
        
        # Apply dark theme
        self._apply_dark_theme()
    
    def _apply_dark_theme(self) -> None:
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #aaa;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                selection-background-color: #007acc;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3c3c3c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #007acc;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #1e1e1e;
                color: #aaa;
            }
            QSplitter::handle {
                background-color: #3c3c3c;
            }
        """)
    
    def _setup_ui(self) -> None:
        """Setup the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Video widget (main area)
        self._video_widget = VideoWidget()
        splitter.addWidget(self._video_widget)
        
        # Right panel container
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Control panel
        self._control_panel = ControlPanel()
        right_layout.addWidget(self._control_panel)
        
        # Quick adjust panel (collapsible)
        self._quick_adjust = QuickAdjustPanel()
        right_layout.addWidget(self._quick_adjust)
        
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (70% video, 30% controls)
        splitter.setSizes([700, 300])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        
        # Status bar widgets
        self._status_device = QLabel("No device")
        self._status_resolution = QLabel("")
        self._status_fps = QLabel("")
        self._status_detection = QLabel("")
        self._status_recording = QLabel("")
        
        self._status_bar.addWidget(self._status_device)
        self._status_bar.addPermanentWidget(self._status_detection)
        self._status_bar.addPermanentWidget(self._status_resolution)
        self._status_bar.addPermanentWidget(self._status_fps)
        self._status_bar.addPermanentWidget(self._status_recording)
    
    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        # Space - Start/Stop
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_streaming)
        
        # F11 - Fullscreen
        QShortcut(QKeySequence(Qt.Key_F11), self, self._toggle_fullscreen)
        
        # Escape - Exit fullscreen
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._exit_fullscreen)
        
        # S - Screenshot
        QShortcut(QKeySequence("S"), self, self._take_screenshot)
        
        # R - Record
        QShortcut(QKeySequence("R"), self, self._toggle_recording)
        
        # P - Phone camera
        QShortcut(QKeySequence("P"), self, self._show_phone_dialog)
        
        # D - Toggle person detection
        QShortcut(QKeySequence("D"), self, self._toggle_detection)
        
        # Q - Quit
        QShortcut(QKeySequence("Q"), self, self.close)
    
    def _toggle_detection(self) -> None:
        """Toggle person detection on/off."""
        new_state = not self._detection_enabled
        self._control_panel.set_detection_enabled(new_state)
        self._on_detection_toggled(new_state)
    
    def _setup_timers(self) -> None:
        """Setup update timers."""
        # Frame update timer (for UI refresh)
        self._frame_timer = QTimer()
        self._frame_timer.timeout.connect(self._update_frame)
        
        # Status update timer
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(500)  # Update every 500ms
    
    def _connect_signals(self) -> None:
        """Connect component signals."""
        # Control panel signals
        self._control_panel.device_changed.connect(self._on_device_changed)
        self._control_panel.start_clicked.connect(self._start_streaming)
        self._control_panel.stop_clicked.connect(self._stop_streaming)
        self._control_panel.screenshot_clicked.connect(self._take_screenshot)
        self._control_panel.record_clicked.connect(self._toggle_recording)
        self._control_panel.settings_clicked.connect(self._show_settings)
        self._control_panel.fullscreen_clicked.connect(self._toggle_fullscreen)
        self._control_panel.refresh_clicked.connect(self._refresh_devices)
        
        # Phone camera button (add to control panel)
        self._control_panel.phone_clicked.connect(self._show_phone_dialog)
        
        # Person detection toggle
        self._control_panel.detection_toggled.connect(self._on_detection_toggled)
        
        # Video widget signals
        self._video_widget.double_clicked.connect(self._toggle_fullscreen)
        
        # Quick adjust signals
        self._quick_adjust.brightness_changed.connect(
            self._controller.set_brightness
        )
        self._quick_adjust.contrast_changed.connect(
            self._controller.set_contrast
        )
        self._quick_adjust.saturation_changed.connect(
            self._controller.set_saturation
        )
        self._quick_adjust.rotation_changed.connect(
            self._on_rotation_changed
        )
        self._quick_adjust.flip_h_changed.connect(
            self._controller.set_flip_horizontal
        )
        self._quick_adjust.flip_v_changed.connect(
            self._controller.set_flip_vertical
        )
        
        # Streamer callbacks
        self._streamer.set_error_callback(self._on_stream_error)
    
    def _refresh_devices(self) -> None:
        """Refresh the device list."""
        self.logger.info("Refreshing device list...")
        self._status_bar.showMessage("Scanning for cameras...", 2000)
        
        devices = self._detector.detect_devices()
        self._control_panel.update_devices(devices)
        
        if devices:
            self._status_bar.showMessage(f"Found {len(devices)} camera(s)", 3000)
        else:
            self._status_bar.showMessage("No cameras found. Try 'Phone Camera' button.", 3000)
    
    def _on_device_changed(self, index: int) -> None:
        """Handle device selection change."""
        self._is_phone_camera = False
        device = self._detector.get_device(index)
        if device:
            self._selected_device = device
            self._status_device.setText(f"Device: {device.name}")
            self.logger.info(f"Selected device: {device.name}")
    
    def _show_phone_dialog(self) -> None:
        """Show phone camera connection dialog."""
        dialog = PhoneCameraDialog(self)
        dialog.device_selected.connect(self._on_phone_device_selected)
        dialog.exec_()
    
    def _on_phone_device_selected(self, device: PhoneDevice) -> None:
        """Handle phone device selection."""
        self.logger.info(f"Phone device selected: {device.name}")
        self._phone_device = device
        self._is_phone_camera = True
        self._status_device.setText(f"📱 {device.name}")
        
        # Automatically start streaming from phone
        self._start_phone_streaming()
    
    def _start_phone_streaming(self) -> None:
        """Start streaming from phone camera."""
        if not self._phone_device:
            QMessageBox.warning(
                self,
                "No Phone",
                "Please connect to a phone camera first."
            )
            return
        
        self.logger.info(f"Starting phone stream: {self._phone_device.stream_url}")
        
        # Stop any existing stream
        if self._streamer.is_streaming:
            self._stop_streaming()
        
        # Connect to phone camera
        if self._phone_streamer.connect(self._phone_device):
            if self._phone_streamer.start():
                self._is_phone_camera = True
                self._control_panel.set_streaming_state(True)
                self._frame_timer.start(16)  # ~60fps update rate
                self._status_bar.showMessage("Phone camera streaming started", 2000)
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to start phone camera stream."
                )
        else:
            QMessageBox.critical(
                self,
                "Error",
                "Failed to connect to phone camera.\n"
                "Make sure the camera app is running on your phone."
            )
    
    def _start_streaming(self) -> None:
        """Start video streaming."""
        # If phone camera is selected, use phone streaming
        if self._is_phone_camera and self._phone_device:
            self._start_phone_streaming()
            return
        
        if not self._selected_device:
            device_index = self._control_panel.get_selected_device_index()
            if device_index < 0:
                QMessageBox.warning(
                    self, 
                    "No Device", 
                    "Please select a camera device first.\n\n"
                    "Or click 'Phone Camera' to connect to your phone."
                )
                return
            self._selected_device = self._detector.get_device(device_index)
        
        self.logger.info(f"Starting stream from {self._selected_device.name}")
        
        # Configure streamer
        resolution = self.config.default_resolution
        fps = self.config.default_fps
        
        self._streamer = CameraStreamer(
            device=self._selected_device,
            resolution=resolution,
            fps=fps
        )
        
        if self._streamer.start():
            self._is_phone_camera = False
            self._control_panel.set_streaming_state(True)
            self._frame_timer.start(16)  # ~60fps update rate
            self._status_bar.showMessage("Streaming started", 2000)
        else:
            QMessageBox.critical(
                self,
                "Error",
                "Failed to start camera stream.\nPlease check the device connection."
            )
    
    def _stop_streaming(self) -> None:
        """Stop video streaming."""
        self._frame_timer.stop()
        
        # Stop recording if active
        if self._controller.is_recording:
            self._controller.stop_recording()
            self._control_panel.set_recording_state(False)
        
        # Stop appropriate streamer
        if self._is_phone_camera:
            self._phone_streamer.stop()
        else:
            self._streamer.stop()
        
        self._control_panel.set_streaming_state(False)
        self._video_widget.clear()
        
        self._status_bar.showMessage("Streaming stopped", 2000)
        self.logger.info("Streaming stopped")
    
    def _toggle_streaming(self) -> None:
        """Toggle streaming state."""
        is_streaming = (
            self._streamer.is_streaming if not self._is_phone_camera 
            else self._phone_streamer.is_streaming
        )
        
        if is_streaming:
            self._stop_streaming()
        else:
            self._start_streaming()
    
    def _on_detection_toggled(self, enabled: bool) -> None:
        """Handle detection toggle change."""
        self._detection_enabled = enabled
        
        if enabled:
            # Initialize detector if not already done
            if self._person_detector is None:
                self._status_bar.showMessage("Loading YOLO model...", 0)
                QApplication.processEvents()
                try:
                    self._person_detector = PersonDetector()
                    self._status_bar.showMessage("YOLO model loaded", 2000)
                    self.logger.info("Person detector initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize person detector: {e}")
                    self._status_bar.showMessage(f"Failed to load YOLO: {e}", 5000)
                    self._detection_enabled = False
                    self._control_panel.set_detection_enabled(False)
                    return
        else:
            self._last_detections = []
        
        self.logger.info(f"Person detection {'enabled' if enabled else 'disabled'}")
    
    def _update_frame(self) -> None:
        """Update the video frame display."""
        # Get frame from appropriate source
        if self._is_phone_camera:
            if not self._phone_streamer.is_streaming:
                return
            frame = self._phone_streamer.get_frame()
            fps = self._phone_streamer.fps
            resolution = self._phone_streamer.resolution
        else:
            if not self._streamer.is_streaming:
                return
            frame = self._streamer.get_frame()
            fps = self._streamer.fps
            resolution = self._streamer.resolution
        
        if frame is not None:
            # Apply image processing
            processed_frame = self._controller.process_frame(frame)
            
            # Person detection
            if self._detection_enabled and self._person_detector is not None:
                try:
                    # Run detection
                    self._last_detections = self._person_detector.detect(processed_frame)
                    
                    # Draw detections on frame
                    processed_frame = self._person_detector.draw_detections(
                        processed_frame,
                        self._last_detections,
                        draw_bbox=True,
                        draw_center=True,
                        draw_label=True
                    )
                    
                    # Log coordinates if any detections
                    if self._last_detections:
                        coords = self._person_detector.get_coordinates(self._last_detections)
                        self.logger.debug(f"Detected {len(coords)} person(s): {coords}")
                except Exception as e:
                    self.logger.error(f"Detection error: {e}")
            
            # Write to recording if active
            if self._controller.is_recording:
                self._controller.write_frame(processed_frame)
            
            # Update video widget
            self._video_widget.update_frame(processed_frame)
            
            # Update overlay info
            if self.config.get("show_fps", True) or self.config.get("show_resolution", True):
                show_fps = fps if self.config.get("show_fps", True) else None
                show_res = resolution if self.config.get("show_resolution", True) else None
                self._video_widget.set_overlay_info(fps=show_fps, resolution=show_res)
    
    def _update_status(self) -> None:
        """Update status bar information."""
        is_streaming = (
            self._streamer.is_streaming if not self._is_phone_camera 
            else self._phone_streamer.is_streaming
        )
        
        if is_streaming:
            # Get info from appropriate source
            if self._is_phone_camera:
                resolution = self._phone_streamer.resolution
                fps = self._phone_streamer.fps
            else:
                resolution = self._streamer.resolution
                fps = self._streamer.fps
            
            # Resolution
            self._status_resolution.setText(f"{resolution[0]}x{resolution[1]}")
            
            # FPS
            self._status_fps.setText(f"{fps:.1f} FPS")
            
            # Detection status
            if self._detection_enabled and self._last_detections:
                count = len(self._last_detections)
                self._status_detection.setText(f"👤 {count} person(s)")
            elif self._detection_enabled:
                self._status_detection.setText("👤 Detecting...")
            else:
                self._status_detection.setText("")
            
            # Recording
            if self._controller.is_recording:
                duration = self._controller.recording_duration
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                self._status_recording.setText(f"🔴 REC {minutes:02d}:{seconds:02d}")
            else:
                self._status_recording.setText("")
        else:
            self._status_resolution.setText("")
            self._status_fps.setText("")
            self._status_detection.setText("")
            self._status_recording.setText("")
    
    def _take_screenshot(self) -> None:
        """Take a screenshot of the current frame."""
        is_streaming = (
            self._streamer.is_streaming if not self._is_phone_camera 
            else self._phone_streamer.is_streaming
        )
        
        if not is_streaming:
            return
        
        frame = self._video_widget.get_current_frame()
        if frame is not None:
            # Apply processing to screenshot
            processed = self._controller.process_frame(frame)
            path = self._controller.capture_screenshot(
                processed,
                format=self.config.get("image_format", "png")
            )
            if path:
                self._status_bar.showMessage(f"Screenshot saved: {path}", 3000)
            else:
                self._status_bar.showMessage("Failed to save screenshot", 3000)
    
    def _toggle_recording(self) -> None:
        """Toggle video recording."""
        is_streaming = (
            self._streamer.is_streaming if not self._is_phone_camera 
            else self._phone_streamer.is_streaming
        )
        
        if not is_streaming:
            return
        
        if self._controller.is_recording:
            # Stop recording
            path = self._controller.stop_recording()
            self._control_panel.set_recording_state(False)
            if path:
                self._status_bar.showMessage(f"Recording saved: {path}", 3000)
        else:
            # Start recording
            if self._is_phone_camera:
                resolution = self._phone_streamer.resolution
            else:
                resolution = self._streamer.resolution
            fps = self.config.default_fps
            
            if self._controller.start_recording(resolution, fps):
                self._control_panel.set_recording_state(True)
                self._status_bar.showMessage("Recording started", 2000)
            else:
                self._status_bar.showMessage("Failed to start recording", 3000)
    
    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        if self._is_fullscreen:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()
    
    def _enter_fullscreen(self) -> None:
        """Enter fullscreen mode."""
        self._is_fullscreen = True
        self.showFullScreen()
    
    def _exit_fullscreen(self) -> None:
        """Exit fullscreen mode."""
        if self._is_fullscreen:
            self._is_fullscreen = False
            self.showNormal()
    
    def _on_rotation_changed(self, degrees: int) -> None:
        """Handle rotation change."""
        if degrees > 0:
            self._controller.rotate_clockwise()
        else:
            self._controller.rotate_counterclockwise()
    
    def _on_stream_error(self, error: str) -> None:
        """Handle streaming error."""
        self.logger.error(f"Stream error: {error}")
        self._stop_streaming()
        QMessageBox.warning(
            self,
            "Stream Error",
            f"Camera stream error:\n{error}"
        )
    
    def _show_settings(self) -> None:
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec_()
    
    def _on_settings_changed(self, settings: dict) -> None:
        """Handle settings change."""
        # Update controller with new image settings
        if "brightness" in settings:
            self._controller.set_brightness(settings["brightness"])
            self._quick_adjust.set_values(
                settings.get("brightness", 50),
                settings.get("contrast", 50),
                settings.get("saturation", 50)
            )
        
        self.logger.info("Settings updated")
    
    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        # Stop streaming
        if self._streamer.is_streaming:
            self._streamer.stop()
        if self._phone_streamer.is_streaming:
            self._phone_streamer.stop()
        
        # Stop device monitoring
        self._detector.stop_monitoring()
        
        # Save window size
        self.config.set("window_size", [self.width(), self.height()])
        
        self.logger.info("Application closed")
        event.accept()
