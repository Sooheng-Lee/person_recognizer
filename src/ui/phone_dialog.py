"""
Phone camera connection dialog for USB Camera Viewer
Supports both Wi-Fi IP cameras and USB connection via ADB
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QGroupBox, QProgressBar, QListWidget, QListWidgetItem,
    QTabWidget, QWidget, QMessageBox, QSpinBox, QRadioButton,
    QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from ..utils.logger import get_logger
from ..camera.phone_camera import (
    PhoneCameraManager, PhoneDevice, IPCameraScanner,
    get_droidcam_url, get_ip_webcam_url
)
from typing import Optional, List


class ScanThread(QThread):
    """Background thread for scanning IP cameras."""
    
    progress = pyqtSignal(int)
    device_found = pyqtSignal(object)  # PhoneDevice
    finished = pyqtSignal(list)
    
    def __init__(self, scanner: IPCameraScanner, subnet: Optional[str] = None):
        super().__init__()
        self.scanner = scanner
        self.subnet = subnet
    
    def run(self):
        devices = self.scanner.find_ip_cameras(
            subnet=self.subnet,
            progress_callback=lambda p: self.progress.emit(int(p))
        )
        self.finished.emit(devices)


class PhoneCameraDialog(QDialog):
    """
    Dialog for connecting to phone cameras.
    Supports manual IP entry, network scanning, and USB via ADB.
    """
    
    device_selected = pyqtSignal(object)  # PhoneDevice
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger("PhoneCameraDialog")
        self.manager = PhoneCameraManager()
        self._scan_thread: Optional[ScanThread] = None
        
        self.setWindowTitle("Connect Phone Camera")
        self.setMinimumSize(550, 550)
        self.setModal(True)
        
        self._setup_ui()
        self._load_adb_devices()
    
    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(
            "📱 Connect your phone camera using USB or Wi-Fi.\n"
            "Install DroidCam or IP Webcam app on your phone first."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #aaa; padding: 10px;")
        layout.addWidget(info_label)
        
        # Tabs
        tabs = QTabWidget()
        
        # ===== USB Connection Tab (NEW!) =====
        usb_tab = QWidget()
        usb_layout = QVBoxLayout(usb_tab)
        
        usb_info = QLabel(
            "🔌 <b>USB 연결 (권장)</b><br><br>"
            "Wi-Fi 없이 USB 케이블만으로 휴대폰 카메라를 사용할 수 있습니다.<br><br>"
            "<b>사전 준비:</b><br>"
            "1. 휴대폰에서 <b>DroidCam</b> 또는 <b>IP Webcam</b> 앱 설치<br>"
            "2. 휴대폰 설정 → 개발자 옵션 → <b>USB 디버깅</b> 활성화<br>"
            "3. USB 케이블로 PC와 휴대폰 연결<br>"
            "4. 휴대폰에서 카메라 앱 실행<br>"
            "5. 아래에서 장치 선택 후 연결"
        )
        usb_info.setWordWrap(True)
        usb_info.setStyleSheet("padding: 10px; background-color: #1a3a1a; border-radius: 5px;")
        usb_layout.addWidget(usb_info)
        
        # USB device selection
        usb_device_group = QGroupBox("USB 연결된 Android 장치")
        usb_device_layout = QVBoxLayout(usb_device_group)
        
        self._usb_device_list = QListWidget()
        self._usb_device_list.setMinimumHeight(100)
        usb_device_layout.addWidget(self._usb_device_list)
        
        refresh_usb_btn = QPushButton("🔄 장치 새로고침")
        refresh_usb_btn.clicked.connect(self._load_adb_devices)
        usb_device_layout.addWidget(refresh_usb_btn)
        
        usb_layout.addWidget(usb_device_group)
        
        # Camera app selection for USB
        usb_app_group = QGroupBox("카메라 앱 선택")
        usb_app_layout = QHBoxLayout(usb_app_group)
        
        self._usb_app_combo = QComboBox()
        self._usb_app_combo.addItem("DroidCam (권장)", "droidcam")
        self._usb_app_combo.addItem("IP Webcam", "ip_webcam")
        self._usb_app_combo.addItem("Iriun Webcam", "iriun")
        usb_app_layout.addWidget(QLabel("앱:"))
        usb_app_layout.addWidget(self._usb_app_combo, 1)
        
        usb_layout.addWidget(usb_app_group)
        
        # USB connect button
        self._usb_connect_btn = QPushButton("🔌 USB로 연결")
        self._usb_connect_btn.setMinimumHeight(45)
        self._usb_connect_btn.setStyleSheet("""
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
        self._usb_connect_btn.clicked.connect(self._connect_usb)
        usb_layout.addWidget(self._usb_connect_btn)
        
        usb_layout.addStretch()
        tabs.addTab(usb_tab, "🔌 USB 연결")
        
        # ===== Wi-Fi IP Tab =====
        wifi_tab = QWidget()
        wifi_layout = QVBoxLayout(wifi_tab)
        
        # App preset selection
        preset_group = QGroupBox("앱 프리셋")
        preset_layout = QHBoxLayout(preset_group)
        
        preset_layout.addWidget(QLabel("앱:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("DroidCam (Port 4747)", ("4747", "/video"))
        self._preset_combo.addItem("IP Webcam (Port 8080)", ("8080", "/video"))
        self._preset_combo.addItem("Iriun Webcam (Port 4747)", ("4747", "/video"))
        self._preset_combo.addItem("Custom", ("8080", "/video"))
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._preset_combo, 1)
        
        wifi_layout.addWidget(preset_group)
        
        # IP Address input
        ip_group = QGroupBox("연결 설정")
        ip_layout = QGridLayout(ip_group)
        
        ip_layout.addWidget(QLabel("IP 주소:"), 0, 0)
        self._ip_edit = QLineEdit()
        self._ip_edit.setPlaceholderText("예: 192.168.1.100")
        ip_layout.addWidget(self._ip_edit, 0, 1, 1, 2)
        
        ip_layout.addWidget(QLabel("포트:"), 1, 0)
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(4747)
        ip_layout.addWidget(self._port_spin, 1, 1)
        
        ip_layout.addWidget(QLabel("경로:"), 2, 0)
        self._path_edit = QLineEdit("/video")
        ip_layout.addWidget(self._path_edit, 2, 1, 1, 2)
        
        # Preview URL
        ip_layout.addWidget(QLabel("URL:"), 3, 0)
        self._url_preview = QLineEdit()
        self._url_preview.setReadOnly(True)
        self._url_preview.setStyleSheet("background-color: #1a1a1a;")
        ip_layout.addWidget(self._url_preview, 3, 1, 1, 2)
        
        wifi_layout.addWidget(ip_group)
        
        # Connect to entered IP and port
        self._ip_edit.textChanged.connect(self._update_url_preview)
        self._port_spin.valueChanged.connect(self._update_url_preview)
        self._path_edit.textChanged.connect(self._update_url_preview)
        
        # Test & Connect buttons
        btn_layout = QHBoxLayout()
        
        self._test_btn = QPushButton("🔍 연결 테스트")
        self._test_btn.clicked.connect(self._test_connection)
        btn_layout.addWidget(self._test_btn)
        
        self._connect_wifi_btn = QPushButton("✓ Wi-Fi 연결")
        self._connect_wifi_btn.setStyleSheet(
            "background-color: #17a2b8; color: white;"
        )
        self._connect_wifi_btn.clicked.connect(self._connect_wifi)
        btn_layout.addWidget(self._connect_wifi_btn)
        
        wifi_layout.addLayout(btn_layout)
        wifi_layout.addStretch()
        
        tabs.addTab(wifi_tab, "📶 Wi-Fi 연결")
        
        # ===== Network Scan Tab =====
        scan_tab = QWidget()
        scan_layout = QVBoxLayout(scan_tab)
        
        # Subnet input
        subnet_group = QGroupBox("네트워크 범위")
        subnet_layout = QHBoxLayout(subnet_group)
        
        subnet_layout.addWidget(QLabel("서브넷:"))
        self._subnet_edit = QLineEdit()
        self._subnet_edit.setPlaceholderText("예: 192.168.1 (자동 감지)")
        subnet_layout.addWidget(self._subnet_edit, 1)
        
        self._scan_btn = QPushButton("🔍 네트워크 스캔")
        self._scan_btn.clicked.connect(self._start_scan)
        subnet_layout.addWidget(self._scan_btn)
        
        scan_layout.addWidget(subnet_group)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        scan_layout.addWidget(self._progress_bar)
        
        # Results list
        results_group = QGroupBox("발견된 카메라")
        results_layout = QVBoxLayout(results_group)
        
        self._results_list = QListWidget()
        self._results_list.itemDoubleClicked.connect(self._on_result_double_clicked)
        results_layout.addWidget(self._results_list)
        
        scan_layout.addWidget(results_group)
        
        # Connect button for scan results
        self._connect_scan_btn = QPushButton("✓ 선택한 카메라 연결")
        self._connect_scan_btn.setEnabled(False)
        self._connect_scan_btn.clicked.connect(self._connect_scanned)
        self._results_list.itemSelectionChanged.connect(
            lambda: self._connect_scan_btn.setEnabled(
                len(self._results_list.selectedItems()) > 0
            )
        )
        scan_layout.addWidget(self._connect_scan_btn)
        
        tabs.addTab(scan_tab, "🔍 네트워크 스캔")
        
        # ===== Help Tab =====
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        
        help_text = QLabel("""
<h3>📱 휴대폰 카메라 연결 가이드</h3>

<h4>🔌 USB 연결 방법 (권장)</h4>
<ol>
<li>Google Play에서 <b>DroidCam</b> 앱 설치</li>
<li>휴대폰 설정 → 휴대전화 정보 → 빌드번호 7번 터치 (개발자 모드 활성화)</li>
<li>설정 → 개발자 옵션 → <b>USB 디버깅</b> 활성화</li>
<li>USB 케이블로 PC에 연결</li>
<li>휴대폰에서 "USB 디버깅 허용" 팝업에서 "확인" 선택</li>
<li>DroidCam 앱 실행</li>
<li>이 프로그램에서 "USB 연결" 탭 → 장치 선택 → 연결</li>
</ol>

<h4>📶 Wi-Fi 연결 방법</h4>
<ol>
<li>PC와 휴대폰이 같은 Wi-Fi에 연결되어 있는지 확인</li>
<li>DroidCam 또는 IP Webcam 앱 실행</li>
<li>앱에 표시된 IP 주소 확인 (예: 192.168.1.100)</li>
<li>"Wi-Fi 연결" 탭에서 IP 주소 입력 후 연결</li>
</ol>

<h4>⚠️ ADB가 없는 경우</h4>
<p>USB 연결을 위해 ADB(Android Debug Bridge)가 필요합니다.<br>
<a href="https://developer.android.com/studio/releases/platform-tools">
Android SDK Platform-Tools</a>를 설치하세요.</p>
        """)
        help_text.setWordWrap(True)
        help_text.setOpenExternalLinks(True)
        help_text.setStyleSheet("padding: 10px;")
        help_layout.addWidget(help_text)
        help_layout.addStretch()
        
        tabs.addTab(help_tab, "❓ 도움말")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)
        
        # Initial URL preview
        self._update_url_preview()
    
    def _load_adb_devices(self):
        """Load ADB-connected devices."""
        self._usb_device_list.clear()
        
        devices = self.manager.detect_phones()
        
        adb_available = self.manager.adb.is_available
        
        if not adb_available:
            item = QListWidgetItem("⚠️ ADB가 설치되지 않았습니다")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._usb_device_list.addItem(item)
            
            item2 = QListWidgetItem("   Android SDK Platform-Tools를 설치하세요")
            item2.setFlags(item2.flags() & ~Qt.ItemIsEnabled)
            self._usb_device_list.addItem(item2)
            
            self._usb_connect_btn.setEnabled(False)
            return
        
        adb_devices = [d for d in devices if d.connection_type == 'adb']
        
        if adb_devices:
            for device in adb_devices:
                item = QListWidgetItem(f"📱 {device.name} ({device.device_id})")
                item.setData(Qt.UserRole, device)
                self._usb_device_list.addItem(item)
            self._usb_connect_btn.setEnabled(True)
        else:
            item = QListWidgetItem("연결된 Android 장치가 없습니다")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self._usb_device_list.addItem(item)
            
            item2 = QListWidgetItem("USB 케이블을 연결하고 USB 디버깅을 활성화하세요")
            item2.setFlags(item2.flags() & ~Qt.ItemIsEnabled)
            self._usb_device_list.addItem(item2)
            
            self._usb_connect_btn.setEnabled(False)
    
    def _connect_usb(self):
        """Connect to phone camera via USB."""
        items = self._usb_device_list.selectedItems()
        if not items:
            # Try to select the first valid item
            for i in range(self._usb_device_list.count()):
                item = self._usb_device_list.item(i)
                if item.flags() & Qt.ItemIsEnabled:
                    item.setSelected(True)
                    items = [item]
                    break
        
        if not items:
            QMessageBox.warning(
                self,
                "장치 없음",
                "연결할 장치를 선택하세요.\n\n"
                "장치가 표시되지 않으면:\n"
                "1. USB 케이블이 연결되어 있는지 확인\n"
                "2. USB 디버깅이 활성화되어 있는지 확인\n"
                "3. '장치 새로고침' 버튼 클릭"
            )
            return
        
        device = items[0].data(Qt.UserRole)
        if not device:
            return
        
        app_type = self._usb_app_combo.currentData()
        
        self._usb_connect_btn.setEnabled(False)
        self._usb_connect_btn.setText("연결 중...")
        
        try:
            # Setup USB camera via ADB port forwarding
            usb_device = self.manager.setup_usb_camera(device.device_id, app_type)
            
            if usb_device:
                QMessageBox.information(
                    self,
                    "연결 성공",
                    f"USB 카메라 연결 준비 완료!\n\n"
                    f"장치: {usb_device.name}\n"
                    f"스트림 URL: {usb_device.stream_url}\n\n"
                    f"휴대폰에서 {app_type.replace('_', ' ').title()} 앱이 "
                    f"실행 중인지 확인하세요."
                )
                self.device_selected.emit(usb_device)
                self.accept()
            else:
                QMessageBox.warning(
                    self,
                    "연결 실패",
                    "USB 카메라 설정에 실패했습니다.\n\n"
                    "다음 사항을 확인하세요:\n"
                    "1. 휴대폰에서 카메라 앱이 실행 중인가요?\n"
                    "2. USB 디버깅이 활성화되어 있나요?\n"
                    "3. PC에서 ADB가 정상적으로 작동하나요?"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                f"연결 중 오류 발생:\n{str(e)}"
            )
        finally:
            self._usb_connect_btn.setEnabled(True)
            self._usb_connect_btn.setText("🔌 USB로 연결")
    
    def _on_preset_changed(self, index: int):
        """Handle app preset change."""
        data = self._preset_combo.currentData()
        if data:
            port, path = data
            self._port_spin.setValue(int(port))
            self._path_edit.setText(path)
    
    def _update_url_preview(self):
        """Update URL preview."""
        ip = self._ip_edit.text().strip()
        port = self._port_spin.value()
        path = self._path_edit.text().strip()
        
        if ip:
            url = f"http://{ip}:{port}{path}"
            self._url_preview.setText(url)
        else:
            self._url_preview.setText("")
    
    def _test_connection(self):
        """Test the connection to the entered IP."""
        ip = self._ip_edit.text().strip()
        port = self._port_spin.value()
        path = self._path_edit.text().strip()
        
        if not ip:
            QMessageBox.warning(self, "오류", "IP 주소를 입력하세요")
            return
        
        url = f"http://{ip}:{port}{path}"
        self._test_btn.setEnabled(False)
        self._test_btn.setText("테스트 중...")
        
        # Test in background
        import cv2
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                
                if ret:
                    QMessageBox.information(
                        self, "성공", 
                        f"✓ 연결 성공!\n\nURL: {url}"
                    )
                else:
                    QMessageBox.warning(
                        self, "실패",
                        f"비디오 스트림을 읽을 수 없습니다:\n{url}"
                    )
            else:
                QMessageBox.warning(
                    self, "실패",
                    f"연결할 수 없습니다:\n{url}\n\n"
                    "휴대폰에서 카메라 앱이 실행 중인지 확인하세요."
                )
        except Exception as e:
            QMessageBox.critical(self, "오류", f"연결 오류:\n{str(e)}")
        finally:
            self._test_btn.setEnabled(True)
            self._test_btn.setText("🔍 연결 테스트")
    
    def _connect_wifi(self):
        """Connect to Wi-Fi IP camera."""
        ip = self._ip_edit.text().strip()
        port = self._port_spin.value()
        path = self._path_edit.text().strip()
        
        if not ip:
            QMessageBox.warning(self, "오류", "IP 주소를 입력하세요")
            return
        
        device = self.manager.add_manual_ip_camera(ip, port, path)
        
        if device:
            self.device_selected.emit(device)
            self.accept()
        else:
            QMessageBox.warning(
                self, "실패",
                "카메라에 연결할 수 없습니다.\n"
                "IP 주소와 카메라 앱 실행 상태를 확인하세요."
            )
    
    def _start_scan(self):
        """Start network scan for IP cameras."""
        subnet = self._subnet_edit.text().strip() or None
        
        self._scan_btn.setEnabled(False)
        self._scan_btn.setText("스캔 중...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._results_list.clear()
        
        # Start scan thread
        self._scan_thread = ScanThread(
            self.manager.scanner, 
            subnet
        )
        self._scan_thread.progress.connect(self._progress_bar.setValue)
        self._scan_thread.finished.connect(self._on_scan_finished)
        self._scan_thread.start()
    
    def _on_scan_finished(self, devices: List[PhoneDevice]):
        """Handle scan completion."""
        self._scan_btn.setEnabled(True)
        self._scan_btn.setText("🔍 네트워크 스캔")
        self._progress_bar.setVisible(False)
        
        if devices:
            for device in devices:
                item = QListWidgetItem(str(device))
                item.setData(Qt.UserRole, device)
                self._results_list.addItem(item)
            
            QMessageBox.information(
                self, "스캔 완료",
                f"{len(devices)}개의 IP 카메라를 발견했습니다"
            )
        else:
            QMessageBox.information(
                self, "스캔 완료",
                "네트워크에서 IP 카메라를 찾지 못했습니다.\n\n"
                "휴대폰의 카메라 앱이 실행 중이고 "
                "같은 네트워크에 연결되어 있는지 확인하세요."
            )
    
    def _on_result_double_clicked(self, item: QListWidgetItem):
        """Handle double-click on scan result."""
        device = item.data(Qt.UserRole)
        if device:
            self.device_selected.emit(device)
            self.accept()
    
    def _connect_scanned(self):
        """Connect to selected scanned device."""
        items = self._results_list.selectedItems()
        if items:
            device = items[0].data(Qt.UserRole)
            if device:
                self.device_selected.emit(device)
                self.accept()
    
    def closeEvent(self, event):
        """Handle dialog close."""
        # Cleanup if needed
        if self._scan_thread and self._scan_thread.isRunning():
            self._scan_thread.wait(1000)
        super().closeEvent(event)
