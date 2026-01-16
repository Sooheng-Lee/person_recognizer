"""
Camera module for USB Camera Viewer
Handles device detection, video streaming, and camera control
"""

from .detector import CameraDetector, CameraDevice
from .streamer import CameraStreamer
from .controller import CameraController
from .phone_camera import (
    PhoneCameraManager,
    PhoneCameraStreamer,
    PhoneDevice,
    ADBHelper,
    IPCameraScanner,
    get_droidcam_url,
    get_ip_webcam_url,
    get_iriun_url
)

__all__ = [
    'CameraDetector',
    'CameraDevice',
    'CameraStreamer',
    'CameraController',
    'PhoneCameraManager',
    'PhoneCameraStreamer',
    'PhoneDevice',
    'ADBHelper',
    'IPCameraScanner',
    'get_droidcam_url',
    'get_ip_webcam_url',
    'get_iriun_url'
]
