"""
Control panel and settings dialog for USB Camera Viewer
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QSlider, QGroupBox,
    QDialog, QDialogButtonBox, QCheckBox, QSpinBox,
    QFileDialog, QLineEdit, QTabWidget, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..camera.detector import CameraDevice
from typing import List, Optional


class ControlPanel(QWidget):
    """
    Main control panel with device selection and playback controls.
    """
    
    # Signals
    device_changed = pyqtSignal(int)  # Device index
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    screenshot_clicked = pyqtSignal()
    record_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    fullscreen_clicked = pyqtSignal()
    refresh_clicked = pyqtSignal()
    phone_clicked = pyqtSignal()  # Phone camera connection
    detection_toggled = pyqtSignal(bool)  # Person detection toggle
    
    def __init__(self, parent=None):
        """Initialize control panel."""
        super().__init__(parent)
        self.logger = get_logger("ControlPanel")
        
        self._setup_ui()
        self._connect_signals()
        
        # State
        self._is_streaming = False
        self._is_recording = False
    
    def _setup_ui(self) -> None:
        """Setup the control panel UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ===== Device Selection Section =====
        device_group = QGroupBox("Camera Device")
        device_layout = QHBoxLayout(device_group)
        
        # Device combo box
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(200)
        self._device_combo.setPlaceholderText("Select a camera...")
        device_layout.addWidget(self._device_combo, 1)
        
        # Refresh button
        self._refresh_btn = QPushButton("🔄 Refresh")
        self._refresh_btn.setFixedWidth(90)
        self._refresh_btn.setToolTip("Scan for connected cameras")
        device_layout.addWidget(self._refresh_btn)
        
        main_layout.addWidget(device_group)
        
        # ===== Playback Controls =====
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Start button
        self._start_btn = QPushButton("▶ Start")
        self._start_btn.setMinimumHeight(40)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        controls_layout.addWidget(self._start_btn)
        
        # Stop button
        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        controls_layout.addWidget(self._stop_btn)
        
        # Fullscreen button
        self._fullscreen_btn = QPushButton("⛶")
        self._fullscreen_btn.setMinimumHeight(40)
        self._fullscreen_btn.setFixedWidth(50)
        self._fullscreen_btn.setToolTip("Toggle fullscreen (F11)")
        self._fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        controls_layout.addWidget(self._fullscreen_btn)
        
        main_layout.addWidget(controls_group)
        
        # ===== Capture Controls =====
        capture_group = QGroupBox("Capture")
        capture_layout = QHBoxLayout(capture_group)
        
        # Screenshot button
        self._screenshot_btn = QPushButton("📷 Screenshot")
        self._screenshot_btn.setMinimumHeight(35)
        self._screenshot_btn.setToolTip("Take screenshot (S)")
        self._screenshot_btn.setEnabled(False)
        capture_layout.addWidget(self._screenshot_btn)
        
        # Record button
        self._record_btn = QPushButton("🎥 Record")
        self._record_btn.setMinimumHeight(35)
        self._record_btn.setToolTip("Start/Stop recording (R)")
        self._record_btn.setEnabled(False)
        self._record_btn.setCheckable(True)
        capture_layout.addWidget(self._record_btn)
        
        main_layout.addWidget(capture_group)
        
        # ===== Detection Controls =====
        detection_group = QGroupBox("Person Detection")
        detection_layout = QHBoxLayout(detection_group)
        
        # Detection toggle checkbox
        self._detection_check = QCheckBox("Enable YOLO Detection")
        self._detection_check.setToolTip("Detect and track people in video (D)")
        detection_layout.addWidget(self._detection_check)
        
        main_layout.addWidget(detection_group)
        
        # ===== Phone Camera Button =====
        self._phone_btn = QPushButton("📱 Phone Camera")
        self._phone_btn.setMinimumHeight(40)
        self._phone_btn.setToolTip("Connect to phone camera via IP (P)")
        self._phone_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        main_layout.addWidget(self._phone_btn)
        
        # ===== Settings Button =====
        self._settings_btn = QPushButton("⚙ Settings")
        self._settings_btn.setMinimumHeight(35)
        main_layout.addWidget(self._settings_btn)
        
        # Add stretch at bottom
        main_layout.addStretch()
    
    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._start_btn.clicked.connect(self._on_start_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._screenshot_btn.clicked.connect(self.screenshot_clicked.emit)
        self._record_btn.clicked.connect(self._on_record_clicked)
        self._settings_btn.clicked.connect(self.settings_clicked.emit)
        self._fullscreen_btn.clicked.connect(self.fullscreen_clicked.emit)
        self._refresh_btn.clicked.connect(self.refresh_clicked.emit)
        self._phone_btn.clicked.connect(self.phone_clicked.emit)
        self._detection_check.stateChanged.connect(self._on_detection_toggled)
    
    def _on_device_changed(self, index: int) -> None:
        """Handle device selection change."""
        if index >= 0:
            self.device_changed.emit(index)
    
    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        self.start_clicked.emit()
    
    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self.stop_clicked.emit()
    
    def _on_record_clicked(self) -> None:
        """Handle record button click."""
        self.record_clicked.emit()
    
    def _on_detection_toggled(self, state: int) -> None:
        """Handle detection toggle change."""
        enabled = state == Qt.Checked
        self.detection_toggled.emit(enabled)
        self.logger.info(f"Person detection {'enabled' if enabled else 'disabled'}")
    
    def set_detection_enabled(self, enabled: bool) -> None:
        """Set detection checkbox state programmatically."""
        self._detection_check.blockSignals(True)
        self._detection_check.setChecked(enabled)
        self._detection_check.blockSignals(False)
    
    def is_detection_enabled(self) -> bool:
        """Check if detection is currently enabled."""
        return self._detection_check.isChecked()
    
    def update_devices(self, devices: List[CameraDevice]) -> None:
        """
        Update the device list.
        
        Args:
            devices: List of detected camera devices
        """
        current_index = self._device_combo.currentIndex()
        
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        
        for device in devices:
            self._device_combo.addItem(str(device), device.index)
        
        # Restore selection if possible
        if current_index >= 0 and current_index < len(devices):
            self._device_combo.setCurrentIndex(current_index)
        
        self._device_combo.blockSignals(False)
        
        self.logger.info(f"Device list updated: {len(devices)} devices")
    
    def get_selected_device_index(self) -> int:
        """Get the currently selected device index."""
        index = self._device_combo.currentIndex()
        if index >= 0:
            return self._device_combo.itemData(index)
        return -1
    
    def set_streaming_state(self, streaming: bool) -> None:
        """
        Update UI for streaming state.
        
        Args:
            streaming: Whether streaming is active
        """
        self._is_streaming = streaming
        
        self._start_btn.setEnabled(not streaming)
        self._stop_btn.setEnabled(streaming)
        self._device_combo.setEnabled(not streaming)
        self._refresh_btn.setEnabled(not streaming)
        self._screenshot_btn.setEnabled(streaming)
        self._record_btn.setEnabled(streaming)
    
    def set_recording_state(self, recording: bool) -> None:
        """
        Update UI for recording state.
        
        Args:
            recording: Whether recording is active
        """
        self._is_recording = recording
        
        if recording:
            self._record_btn.setText("⏹ Stop Rec")
            self._record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)
        else:
            self._record_btn.setText("🎥 Record")
            self._record_btn.setStyleSheet("")
            self._record_btn.setChecked(False)


class SettingsDialog(QDialog):
    """
    Settings dialog for camera and application configuration.
    """
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize settings dialog."""
        super().__init__(parent)
        self.logger = get_logger("SettingsDialog")
        self.config = get_config()
        
        self.setWindowTitle("Settings")
        self.setMinimumSize(400, 450)
        self.setModal(True)
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self) -> None:
        """Setup the settings dialog UI."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        tabs = QTabWidget()
        
        # ===== Video Tab =====
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        
        # Resolution
        res_group = QGroupBox("Resolution")
        res_layout = QHBoxLayout(res_group)
        
        res_layout.addWidget(QLabel("Preset:"))
        self._resolution_combo = QComboBox()
        self._resolution_combo.addItem("480p (640x480)", (640, 480))
        self._resolution_combo.addItem("720p (1280x720)", (1280, 720))
        self._resolution_combo.addItem("1080p (1920x1080)", (1920, 1080))
        self._resolution_combo.addItem("1440p (2560x1440)", (2560, 1440))
        self._resolution_combo.addItem("4K (3840x2160)", (3840, 2160))
        res_layout.addWidget(self._resolution_combo)
        
        video_layout.addWidget(res_group)
        
        # FPS
        fps_group = QGroupBox("Frame Rate")
        fps_layout = QHBoxLayout(fps_group)
        
        fps_layout.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(15, 120)
        self._fps_spin.setValue(30)
        self._fps_spin.setSuffix(" fps")
        fps_layout.addWidget(self._fps_spin)
        fps_layout.addStretch()
        
        video_layout.addWidget(fps_group)
        
        video_layout.addStretch()
        tabs.addTab(video_tab, "Video")
        
        # ===== Image Tab =====
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        
        # Brightness
        bright_group = QGroupBox("Brightness")
        bright_layout = QHBoxLayout(bright_group)
        self._brightness_slider = QSlider(Qt.Horizontal)
        self._brightness_slider.setRange(0, 100)
        self._brightness_slider.setValue(50)
        self._brightness_label = QLabel("50")
        self._brightness_slider.valueChanged.connect(
            lambda v: self._brightness_label.setText(str(v))
        )
        bright_layout.addWidget(self._brightness_slider)
        bright_layout.addWidget(self._brightness_label)
        image_layout.addWidget(bright_group)
        
        # Contrast
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QHBoxLayout(contrast_group)
        self._contrast_slider = QSlider(Qt.Horizontal)
        self._contrast_slider.setRange(0, 100)
        self._contrast_slider.setValue(50)
        self._contrast_label = QLabel("50")
        self._contrast_slider.valueChanged.connect(
            lambda v: self._contrast_label.setText(str(v))
        )
        contrast_layout.addWidget(self._contrast_slider)
        contrast_layout.addWidget(self._contrast_label)
        image_layout.addWidget(contrast_group)
        
        # Saturation
        sat_group = QGroupBox("Saturation")
        sat_layout = QHBoxLayout(sat_group)
        self._saturation_slider = QSlider(Qt.Horizontal)
        self._saturation_slider.setRange(0, 100)
        self._saturation_slider.setValue(50)
        self._saturation_label = QLabel("50")
        self._saturation_slider.valueChanged.connect(
            lambda v: self._saturation_label.setText(str(v))
        )
        sat_layout.addWidget(self._saturation_slider)
        sat_layout.addWidget(self._saturation_label)
        image_layout.addWidget(sat_group)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_image_settings)
        image_layout.addWidget(reset_btn)
        
        image_layout.addStretch()
        tabs.addTab(image_tab, "Image")
        
        # ===== Save Tab =====
        save_tab = QWidget()
        save_layout = QVBoxLayout(save_tab)
        
        # Save path
        path_group = QGroupBox("Save Location")
        path_layout = QHBoxLayout(path_group)
        
        self._save_path_edit = QLineEdit()
        self._save_path_edit.setReadOnly(True)
        path_layout.addWidget(self._save_path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_save_path)
        path_layout.addWidget(browse_btn)
        
        save_layout.addWidget(path_group)
        
        # Image format
        img_format_group = QGroupBox("Screenshot Format")
        img_format_layout = QHBoxLayout(img_format_group)
        
        self._image_format_combo = QComboBox()
        self._image_format_combo.addItem("PNG", "png")
        self._image_format_combo.addItem("JPEG", "jpg")
        img_format_layout.addWidget(self._image_format_combo)
        img_format_layout.addStretch()
        
        save_layout.addWidget(img_format_group)
        
        # Video format
        vid_format_group = QGroupBox("Video Format")
        vid_format_layout = QHBoxLayout(vid_format_group)
        
        self._video_format_combo = QComboBox()
        self._video_format_combo.addItem("MP4", "mp4")
        self._video_format_combo.addItem("AVI", "avi")
        vid_format_layout.addWidget(self._video_format_combo)
        vid_format_layout.addStretch()
        
        save_layout.addWidget(vid_format_group)
        
        save_layout.addStretch()
        tabs.addTab(save_tab, "Save")
        
        # ===== General Tab =====
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self._auto_connect_check = QCheckBox("Auto-connect to last device")
        options_layout.addWidget(self._auto_connect_check)
        
        self._show_fps_check = QCheckBox("Show FPS overlay")
        options_layout.addWidget(self._show_fps_check)
        
        self._show_resolution_check = QCheckBox("Show resolution overlay")
        options_layout.addWidget(self._show_resolution_check)
        
        general_layout.addWidget(options_group)
        general_layout.addStretch()
        
        tabs.addTab(general_tab, "General")
        
        layout.addWidget(tabs)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self._save_and_close)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)
        
        layout.addWidget(button_box)
    
    def _load_settings(self) -> None:
        """Load current settings into UI."""
        # Resolution
        resolution = self.config.default_resolution
        for i in range(self._resolution_combo.count()):
            if self._resolution_combo.itemData(i) == resolution:
                self._resolution_combo.setCurrentIndex(i)
                break
        
        # FPS
        self._fps_spin.setValue(self.config.default_fps)
        
        # Image adjustments
        self._brightness_slider.setValue(self.config.get("brightness", 50))
        self._contrast_slider.setValue(self.config.get("contrast", 50))
        self._saturation_slider.setValue(self.config.get("saturation", 50))
        
        # Save path
        self._save_path_edit.setText(str(self.config.save_path))
        
        # Formats
        img_format = self.config.get("image_format", "png")
        for i in range(self._image_format_combo.count()):
            if self._image_format_combo.itemData(i) == img_format:
                self._image_format_combo.setCurrentIndex(i)
                break
        
        vid_format = self.config.get("video_format", "mp4")
        for i in range(self._video_format_combo.count()):
            if self._video_format_combo.itemData(i) == vid_format:
                self._video_format_combo.setCurrentIndex(i)
                break
        
        # General options
        self._auto_connect_check.setChecked(self.config.auto_connect)
        self._show_fps_check.setChecked(self.config.get("show_fps", True))
        self._show_resolution_check.setChecked(self.config.get("show_resolution", True))
    
    def _get_settings(self) -> dict:
        """Get current settings from UI."""
        return {
            "default_resolution": list(self._resolution_combo.currentData()),
            "default_fps": self._fps_spin.value(),
            "brightness": self._brightness_slider.value(),
            "contrast": self._contrast_slider.value(),
            "saturation": self._saturation_slider.value(),
            "save_path": self._save_path_edit.text(),
            "image_format": self._image_format_combo.currentData(),
            "video_format": self._video_format_combo.currentData(),
            "auto_connect": self._auto_connect_check.isChecked(),
            "show_fps": self._show_fps_check.isChecked(),
            "show_resolution": self._show_resolution_check.isChecked()
        }
    
    def _apply_settings(self) -> None:
        """Apply settings without closing."""
        settings = self._get_settings()
        self.config.update(settings)
        self.settings_changed.emit(settings)
        self.logger.info("Settings applied")
    
    def _save_and_close(self) -> None:
        """Save settings and close dialog."""
        self._apply_settings()
        self.accept()
    
    def _reset_image_settings(self) -> None:
        """Reset image settings to defaults."""
        self._brightness_slider.setValue(50)
        self._contrast_slider.setValue(50)
        self._saturation_slider.setValue(50)
    
    def _browse_save_path(self) -> None:
        """Open directory browser for save path."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            self._save_path_edit.text()
        )
        if path:
            self._save_path_edit.setText(path)


class QuickAdjustPanel(QWidget):
    """
    Quick adjustment panel for real-time image adjustments.
    Displayed as a collapsible sidebar.
    """
    
    brightness_changed = pyqtSignal(int)
    contrast_changed = pyqtSignal(int)
    saturation_changed = pyqtSignal(int)
    rotation_changed = pyqtSignal(int)
    flip_h_changed = pyqtSignal(bool)
    flip_v_changed = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        """Initialize quick adjust panel."""
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Quick Adjust")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        # Brightness
        layout.addWidget(QLabel("Brightness"))
        self._brightness = QSlider(Qt.Horizontal)
        self._brightness.setRange(0, 100)
        self._brightness.setValue(50)
        self._brightness.valueChanged.connect(self.brightness_changed.emit)
        layout.addWidget(self._brightness)
        
        # Contrast
        layout.addWidget(QLabel("Contrast"))
        self._contrast = QSlider(Qt.Horizontal)
        self._contrast.setRange(0, 100)
        self._contrast.setValue(50)
        self._contrast.valueChanged.connect(self.contrast_changed.emit)
        layout.addWidget(self._contrast)
        
        # Saturation
        layout.addWidget(QLabel("Saturation"))
        self._saturation = QSlider(Qt.Horizontal)
        self._saturation.setRange(0, 100)
        self._saturation.setValue(50)
        self._saturation.valueChanged.connect(self.saturation_changed.emit)
        layout.addWidget(self._saturation)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)
        
        # Transform buttons
        transform_layout = QGridLayout()
        
        # Rotate buttons
        rotate_left = QPushButton("↺")
        rotate_left.setToolTip("Rotate left")
        rotate_left.clicked.connect(lambda: self.rotation_changed.emit(-90))
        transform_layout.addWidget(rotate_left, 0, 0)
        
        rotate_right = QPushButton("↻")
        rotate_right.setToolTip("Rotate right")
        rotate_right.clicked.connect(lambda: self.rotation_changed.emit(90))
        transform_layout.addWidget(rotate_right, 0, 1)
        
        # Flip buttons
        flip_h = QPushButton("↔")
        flip_h.setToolTip("Flip horizontal")
        flip_h.setCheckable(True)
        flip_h.toggled.connect(self.flip_h_changed.emit)
        transform_layout.addWidget(flip_h, 1, 0)
        
        flip_v = QPushButton("↕")
        flip_v.setToolTip("Flip vertical")
        flip_v.setCheckable(True)
        flip_v.toggled.connect(self.flip_v_changed.emit)
        transform_layout.addWidget(flip_v, 1, 1)
        
        layout.addLayout(transform_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self._reset_all)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
    
    def _reset_all(self) -> None:
        """Reset all adjustments."""
        self._brightness.setValue(50)
        self._contrast.setValue(50)
        self._saturation.setValue(50)
    
    def set_values(self, brightness: int, contrast: int, saturation: int) -> None:
        """Set slider values."""
        self._brightness.blockSignals(True)
        self._contrast.blockSignals(True)
        self._saturation.blockSignals(True)
        
        self._brightness.setValue(brightness)
        self._contrast.setValue(contrast)
        self._saturation.setValue(saturation)
        
        self._brightness.blockSignals(False)
        self._contrast.blockSignals(False)
        self._saturation.blockSignals(False)
