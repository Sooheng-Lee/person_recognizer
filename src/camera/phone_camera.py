"""
Phone camera module for USB Camera Viewer
Supports Android devices via ADB and IP camera streams
Including USB direct connection via ADB port forwarding
"""

import cv2
import subprocess
import socket
import re
import numpy as np
from typing import Optional, List, Tuple
from threading import Thread, Event
from dataclasses import dataclass
import time
import os

from ..utils.logger import get_logger


@dataclass
class PhoneDevice:
    """
    Represents a connected phone device.
    """
    device_id: str
    name: str
    connection_type: str  # 'adb', 'adb_usb', 'ip', 'usb_webcam'
    ip_address: Optional[str] = None
    port: int = 8080
    stream_url: Optional[str] = None
    local_port: Optional[int] = None  # For ADB port forwarding
    
    def __str__(self) -> str:
        if self.connection_type == 'ip':
            return f"📱 {self.name} (IP: {self.ip_address}:{self.port})"
        elif self.connection_type == 'adb':
            return f"📱 {self.name} (ADB: {self.device_id})"
        elif self.connection_type == 'adb_usb':
            return f"📱 {self.name} (USB via ADB)"
        return f"📱 {self.name}"


class ADBHelper:
    """
    Helper class for Android Debug Bridge operations.
    Supports USB port forwarding for direct camera access.
    """
    
    # Common camera app ports
    CAMERA_APP_PORTS = {
        'droidcam': 4747,
        'ip_webcam': 8080,
        'iriun': 4747,
    }
    
    def __init__(self):
        self.logger = get_logger("ADBHelper")
        self._adb_path = self._find_adb()
        self._forwarded_ports: dict = {}  # device_id -> local_port
    
    def _find_adb(self) -> Optional[str]:
        """Find ADB executable in system PATH or common locations."""
        # Check if adb is in PATH
        try:
            result = subprocess.run(
                ['adb', 'version'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            if result.returncode == 0:
                self.logger.info("ADB found in system PATH")
                return 'adb'
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Common ADB locations on Windows
        common_paths = [
            os.path.expandvars(r'%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe'),
            os.path.expandvars(r'%PROGRAMFILES%\Android\android-sdk\platform-tools\adb.exe'),
            os.path.expandvars(r'%USERPROFILE%\AppData\Local\Android\Sdk\platform-tools\adb.exe'),
            r'C:\Android\platform-tools\adb.exe',
            r'C:\adb\adb.exe',
            r'C:\platform-tools\adb.exe'
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.logger.info(f"ADB found at: {path}")
                return path
        
        self.logger.warning("ADB not found")
        return None
    
    @property
    def is_available(self) -> bool:
        """Check if ADB is available."""
        return self._adb_path is not None
    
    def _run_adb(self, args: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
        """Run an ADB command."""
        cmd = [self._adb_path] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
    
    def get_devices(self) -> List[Tuple[str, str]]:
        """
        Get list of connected Android devices.
        
        Returns:
            List of tuples (device_id, device_name)
        """
        if not self._adb_path:
            return []
        
        try:
            result = self._run_adb(['devices', '-l'])
            
            devices = []
            for line in result.stdout.strip().split('\n')[1:]:
                if '\tdevice' in line or ' device ' in line:
                    parts = line.split()
                    device_id = parts[0]
                    
                    # Try to get device model name
                    model = self._get_device_model(device_id)
                    devices.append((device_id, model or device_id))
            
            return devices
            
        except subprocess.SubprocessError as e:
            self.logger.error(f"ADB command failed: {e}")
            return []
    
    def _get_device_model(self, device_id: str) -> Optional[str]:
        """Get device model name."""
        try:
            result = self._run_adb(['-s', device_id, 'shell', 
                 'getprop', 'ro.product.model'], timeout=5)
            return result.stdout.strip() if result.returncode == 0 else None
        except subprocess.SubprocessError:
            return None
    
    def forward_port(self, device_id: str, local_port: int, remote_port: int) -> bool:
        """
        Forward a port from device to local machine via USB.
        This enables accessing phone apps via USB instead of Wi-Fi.
        
        Args:
            device_id: ADB device ID
            local_port: Local port on PC
            remote_port: Port on the phone (app's server port)
            
        Returns:
            True if forwarding successful
        """
        if not self._adb_path:
            return False
        
        try:
            result = self._run_adb(['-s', device_id, 'forward',
                 f'tcp:{local_port}', f'tcp:{remote_port}'])
            
            if result.returncode == 0:
                self._forwarded_ports[device_id] = local_port
                self.logger.info(f"Port forwarding: localhost:{local_port} -> {device_id}:{remote_port}")
                return True
            else:
                self.logger.error(f"Port forwarding failed: {result.stderr}")
                return False
                
        except subprocess.SubprocessError as e:
            self.logger.error(f"Port forwarding failed: {e}")
            return False
    
    def remove_forward(self, device_id: str, local_port: int) -> bool:
        """Remove a port forwarding."""
        if not self._adb_path:
            return False
        
        try:
            result = self._run_adb(['-s', device_id, 'forward', '--remove',
                 f'tcp:{local_port}'])
            
            if device_id in self._forwarded_ports:
                del self._forwarded_ports[device_id]
            
            return result.returncode == 0
        except subprocess.SubprocessError:
            return False
    
    def remove_all_forwards(self) -> None:
        """Remove all port forwardings."""
        if not self._adb_path:
            return
        
        try:
            self._run_adb(['forward', '--remove-all'])
            self._forwarded_ports.clear()
        except subprocess.SubprocessError:
            pass
    
    def check_app_running(self, device_id: str, package_name: str) -> bool:
        """
        Check if a camera app is running on the device.
        
        Args:
            device_id: ADB device ID
            package_name: Android package name
            
        Returns:
            True if app is running
        """
        if not self._adb_path:
            return False
        
        try:
            result = self._run_adb(['-s', device_id, 'shell',
                 'pidof', package_name], timeout=5)
            return bool(result.stdout.strip())
        except subprocess.SubprocessError:
            return False
    
    def start_app(self, device_id: str, package_name: str, activity: str = "") -> bool:
        """
        Start an app on the device.
        
        Args:
            device_id: ADB device ID
            package_name: Android package name
            activity: Activity to start (optional)
            
        Returns:
            True if app started
        """
        if not self._adb_path:
            return False
        
        try:
            if activity:
                cmd = ['-s', device_id, 'shell', 'am', 'start', '-n', 
                       f'{package_name}/{activity}']
            else:
                cmd = ['-s', device_id, 'shell', 'monkey', '-p', package_name,
                       '-c', 'android.intent.category.LAUNCHER', '1']
            
            result = self._run_adb(cmd)
            return result.returncode == 0
        except subprocess.SubprocessError:
            return False
    
    def setup_usb_camera(self, device_id: str, app_type: str = 'droidcam', 
                         local_port: int = None) -> Optional[str]:
        """
        Setup USB camera access via ADB port forwarding.
        
        Args:
            device_id: ADB device ID
            app_type: Type of camera app ('droidcam', 'ip_webcam', 'iriun')
            local_port: Local port to use (auto-assigned if None)
            
        Returns:
            Stream URL (localhost) if successful, None otherwise
        """
        if not self._adb_path:
            self.logger.error("ADB not available")
            return None
        
        # Get remote port for the app
        remote_port = self.CAMERA_APP_PORTS.get(app_type, 8080)
        
        # Auto-assign local port if not specified
        if local_port is None:
            local_port = self._find_free_port()
        
        # Setup port forwarding
        if self.forward_port(device_id, local_port, remote_port):
            # Build stream URL
            if app_type == 'droidcam':
                stream_url = f"http://127.0.0.1:{local_port}/video"
            elif app_type == 'ip_webcam':
                stream_url = f"http://127.0.0.1:{local_port}/video"
            elif app_type == 'iriun':
                stream_url = f"http://127.0.0.1:{local_port}/video"
            else:
                stream_url = f"http://127.0.0.1:{local_port}/video"
            
            self.logger.info(f"USB camera setup complete: {stream_url}")
            return stream_url
        
        return None
    
    def _find_free_port(self, start: int = 10000, end: int = 60000) -> int:
        """Find a free local port."""
        import random
        for _ in range(100):
            port = random.randint(start, end)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', port))
                sock.close()
                return port
            except OSError:
                continue
        return start  # Fallback


class IPCameraScanner:
    """
    Scans for IP cameras on the local network.
    """
    
    COMMON_PORTS = [8080, 8081, 4747, 8000, 554, 80]
    COMMON_PATHS = [
        '/video',
        '/videofeed',
        '/mjpegfeed',
        '/shot.jpg',
        '/',
        '/cam',
        '/stream'
    ]
    
    def __init__(self):
        self.logger = get_logger("IPCameraScanner")
    
    def get_local_network(self) -> Optional[str]:
        """Get the local network address prefix."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Get network prefix (e.g., "192.168.1")
            prefix = '.'.join(local_ip.split('.')[:-1])
            return prefix
        except Exception as e:
            self.logger.error(f"Failed to get local network: {e}")
            return None
    
    def scan_port(self, ip: str, port: int, timeout: float = 0.5) -> bool:
        """Check if a port is open on an IP address."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def test_stream_url(self, url: str, timeout: float = 3.0) -> bool:
        """Test if a URL is a valid video stream."""
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
            
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except Exception:
            return False
    
    def find_ip_cameras(
        self, 
        subnet: Optional[str] = None,
        progress_callback=None
    ) -> List[PhoneDevice]:
        """
        Scan network for IP cameras.
        
        Args:
            subnet: Network subnet to scan (e.g., "192.168.1")
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of discovered IP camera devices
        """
        if subnet is None:
            subnet = self.get_local_network()
        
        if not subnet:
            self.logger.error("Could not determine local network")
            return []
        
        self.logger.info(f"Scanning network {subnet}.0/24 for IP cameras...")
        devices = []
        
        # Scan common IP range (limited for speed)
        for i in range(1, 255):
            ip = f"{subnet}.{i}"
            
            if progress_callback:
                progress_callback(i / 255 * 100)
            
            for port in self.COMMON_PORTS:
                if self.scan_port(ip, port, timeout=0.3):
                    # Found open port, try to connect as camera
                    for path in self.COMMON_PATHS:
                        url = f"http://{ip}:{port}{path}"
                        if self.test_stream_url(url, timeout=2.0):
                            device = PhoneDevice(
                                device_id=f"ip_{ip}_{port}",
                                name=f"IP Camera ({ip})",
                                connection_type='ip',
                                ip_address=ip,
                                port=port,
                                stream_url=url
                            )
                            devices.append(device)
                            self.logger.info(f"Found IP camera: {url}")
                            break
        
        return devices


class PhoneCameraStreamer:
    """
    Streams video from phone cameras via various methods.
    """
    
    def __init__(self):
        self.logger = get_logger("PhoneCameraStreamer")
        self._capture: Optional[cv2.VideoCapture] = None
        self._device: Optional[PhoneDevice] = None
        self._is_streaming = False
        self._current_frame: Optional[np.ndarray] = None
        self._stop_event = Event()
        self._capture_thread: Optional[Thread] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
    
    def connect(self, device: PhoneDevice) -> bool:
        """
        Connect to a phone camera.
        
        Args:
            device: PhoneDevice to connect to
            
        Returns:
            True if connection successful
        """
        self._device = device
        
        if device.connection_type == 'ip':
            return self._connect_ip_camera(device)
        elif device.connection_type == 'adb_usb':
            return self._connect_usb_camera(device)
        elif device.connection_type == 'adb':
            return self._connect_adb_camera(device)
        
        return False
    
    def _connect_ip_camera(self, device: PhoneDevice) -> bool:
        """Connect to IP camera stream."""
        if not device.stream_url:
            self.logger.error("No stream URL provided")
            return False
        
        try:
            self.logger.info(f"Connecting to IP camera: {device.stream_url}")
            
            self._capture = cv2.VideoCapture(device.stream_url)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self._capture.isOpened():
                # Try to read a test frame
                ret, _ = self._capture.read()
                if ret:
                    self.logger.info("IP camera connected successfully")
                    return True
            
            self.logger.error("Failed to open IP camera stream")
            return False
            
        except Exception as e:
            self.logger.error(f"IP camera connection error: {e}")
            return False
    
    def _connect_usb_camera(self, device: PhoneDevice) -> bool:
        """
        Connect to phone camera via USB (ADB port forwarding).
        """
        if not device.stream_url:
            self.logger.error("No stream URL provided for USB camera")
            return False
        
        try:
            self.logger.info(f"Connecting to USB camera: {device.stream_url}")
            
            # Give some time for port forwarding to stabilize
            time.sleep(0.5)
            
            self._capture = cv2.VideoCapture(device.stream_url)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self._capture.isOpened():
                # Try to read a test frame
                ret, _ = self._capture.read()
                if ret:
                    self.logger.info("USB camera connected successfully")
                    return True
            
            self.logger.error("Failed to open USB camera stream. Is the camera app running on your phone?")
            return False
            
        except Exception as e:
            self.logger.error(f"USB camera connection error: {e}")
            return False
    
    def _connect_adb_camera(self, device: PhoneDevice) -> bool:
        """
        Connect to Android camera via ADB.
        Note: This typically requires an app on the phone that exposes
        the camera as a stream (like DroidCam or IP Webcam).
        """
        self.logger.warning(
            "Direct ADB camera access requires a streaming app on the phone.\n"
            "Please install 'DroidCam' or 'IP Webcam' on your phone."
        )
        return False
    
    def start(self) -> bool:
        """Start streaming."""
        if not self._capture or not self._capture.isOpened():
            self.logger.error("No camera connected")
            return False
        
        self._stop_event.clear()
        self._is_streaming = True
        self._frame_count = 0
        
        self._capture_thread = Thread(
            target=self._capture_loop,
            daemon=True
        )
        self._capture_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop streaming."""
        self._stop_event.set()
        self._is_streaming = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        
        if self._capture:
            self._capture.release()
            self._capture = None
    
    def _capture_loop(self) -> None:
        """Internal capture loop."""
        last_fps_time = time.time()
        fps_frame_count = 0
        
        while not self._stop_event.is_set():
            if not self._capture or not self._capture.isOpened():
                break
            
            ret, frame = self._capture.read()
            
            if ret:
                self._current_frame = frame
                self._frame_count += 1
                fps_frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                elapsed = current_time - last_fps_time
                if elapsed >= 1.0:
                    self._fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    last_fps_time = current_time
            else:
                time.sleep(0.01)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        return self._current_frame.copy() if self._current_frame is not None else None
    
    @property
    def is_streaming(self) -> bool:
        return self._is_streaming
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def resolution(self) -> Tuple[int, int]:
        if self._capture and self._capture.isOpened():
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)


class PhoneCameraManager:
    """
    Manages phone camera detection and connection.
    Supports both Wi-Fi IP cameras and USB connections via ADB.
    """
    
    def __init__(self):
        self.logger = get_logger("PhoneCameraManager")
        self.adb = ADBHelper()
        self.scanner = IPCameraScanner()
        self._devices: List[PhoneDevice] = []
    
    def detect_phones(self) -> List[PhoneDevice]:
        """
        Detect connected phones via ADB and IP cameras on network.
        
        Returns:
            List of detected phone devices
        """
        devices = []
        
        # Check for ADB devices
        if self.adb.is_available:
            self.logger.info("Scanning for ADB devices...")
            adb_devices = self.adb.get_devices()
            for device_id, name in adb_devices:
                device = PhoneDevice(
                    device_id=device_id,
                    name=name,
                    connection_type='adb'
                )
                devices.append(device)
                self.logger.info(f"Found ADB device: {name}")
        
        self._devices = devices
        return devices
    
    def setup_usb_camera(self, device_id: str, app_type: str = 'droidcam') -> Optional[PhoneDevice]:
        """
        Setup USB camera access for an ADB-connected phone.
        Uses port forwarding to access camera app over USB instead of Wi-Fi.
        
        Args:
            device_id: ADB device ID
            app_type: Camera app type ('droidcam', 'ip_webcam', 'iriun')
            
        Returns:
            PhoneDevice configured for USB access, or None if failed
        """
        if not self.adb.is_available:
            self.logger.error("ADB not available")
            return None
        
        # Find the device name
        adb_devices = self.adb.get_devices()
        device_name = device_id
        for did, name in adb_devices:
            if did == device_id:
                device_name = name
                break
        
        # Setup port forwarding
        stream_url = self.adb.setup_usb_camera(device_id, app_type)
        
        if stream_url:
            # Extract local port from URL
            import re
            port_match = re.search(r':(\d+)/', stream_url)
            local_port = int(port_match.group(1)) if port_match else 0
            
            device = PhoneDevice(
                device_id=device_id,
                name=f"{device_name} (USB)",
                connection_type='adb_usb',
                ip_address='127.0.0.1',
                port=local_port,
                stream_url=stream_url,
                local_port=local_port
            )
            
            # Add to device list
            self._devices.append(device)
            
            return device
        
        return None
    
    def scan_ip_cameras(self, subnet: Optional[str] = None) -> List[PhoneDevice]:
        """
        Scan network for IP cameras.
        
        Args:
            subnet: Network subnet to scan
            
        Returns:
            List of IP camera devices
        """
        self.logger.info("Scanning for IP cameras...")
        ip_devices = self.scanner.find_ip_cameras(subnet)
        self._devices.extend(ip_devices)
        return ip_devices
    
    def add_manual_ip_camera(
        self, 
        ip: str, 
        port: int = 8080,
        path: str = '/video'
    ) -> Optional[PhoneDevice]:
        """
        Manually add an IP camera.
        
        Args:
            ip: IP address
            port: Port number
            path: Stream path
            
        Returns:
            PhoneDevice if connection successful, None otherwise
        """
        url = f"http://{ip}:{port}{path}"
        
        self.logger.info(f"Testing manual IP camera: {url}")
        
        if self.scanner.test_stream_url(url):
            device = PhoneDevice(
                device_id=f"ip_{ip}_{port}",
                name=f"IP Camera ({ip})",
                connection_type='ip',
                ip_address=ip,
                port=port,
                stream_url=url
            )
            self._devices.append(device)
            self.logger.info(f"Manual IP camera added: {url}")
            return device
        
        self.logger.warning(f"Could not connect to: {url}")
        return None
    
    def cleanup(self) -> None:
        """Cleanup resources (remove port forwardings, etc.)."""
        self.adb.remove_all_forwards()
    
    @property
    def devices(self) -> List[PhoneDevice]:
        return self._devices.copy()
    
    def get_device(self, device_id: str) -> Optional[PhoneDevice]:
        """Get device by ID."""
        for device in self._devices:
            if device.device_id == device_id:
                return device
        return None


def get_droidcam_url(ip: str, port: int = 4747) -> str:
    """
    Get DroidCam app stream URL.
    
    DroidCam is a popular app that turns Android/iOS phones into webcams.
    Install from Play Store/App Store and use this URL format.
    
    Args:
        ip: Phone's IP address (shown in DroidCam app)
        port: DroidCam port (default 4747)
        
    Returns:
        Stream URL for DroidCam
    """
    return f"http://{ip}:{port}/video"


def get_ip_webcam_url(ip: str, port: int = 8080) -> str:
    """
    Get IP Webcam app stream URL.
    
    IP Webcam is another popular Android app for streaming camera.
    
    Args:
        ip: Phone's IP address (shown in IP Webcam app)
        port: Port (default 8080)
        
    Returns:
        Stream URL for IP Webcam
    """
    return f"http://{ip}:{port}/video"


def get_iriun_url(ip: str, port: int = 4747) -> str:
    """
    Get Iriun Webcam app stream URL.
    
    Args:
        ip: Phone's IP address
        port: Port number
        
    Returns:
        Stream URL
    """
    return f"http://{ip}:{port}/video"
