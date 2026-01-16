# USB Camera Viewer

USB를 통해 연결된 스마트폰 또는 카메라의 영상을 컴퓨터 화면에 실시간으로 표시하는 데스크톱 애플리케이션입니다.

## 📋 기능

### MVP (Phase 1)
- ✅ USB 장치 자동 감지
- ✅ 실시간 영상 스트리밍
- ✅ 장치 선택 UI
- ✅ 시작/중지 제어
- ✅ 전체 화면 모드

### 📱 휴대폰 카메라 연결
- ✅ IP 카메라 스트림 지원 (DroidCam, IP Webcam)
- ✅ 수동 IP 주소 입력
- ✅ 네트워크 스캔으로 자동 검색
- ✅ ADB 연결 장치 탐지

### 👤 YOLO 사람 감지 (NEW!)
- ✅ YOLOv8 기반 실시간 사람 감지 (ultralytics 사용 시)
- ✅ YOLOv4-tiny OpenCV DNN 폴백 (torch 설치 실패 시 자동)
- ✅ 바운딩 박스 및 중심점 표시
- ✅ 사람 좌표 추출 (bbox, center)
- ✅ 감지 신뢰도 표시
- ✅ 감지 활성화/비활성화 토글

### 추가 기능 (Phase 2)
- 📷 스크린샷 캡처
- 🎥 영상 녹화
- 🔧 밝기/대비/채도 조절
- 🔄 영상 회전/반전

## 🛠️ 설치 방법

### 1. 요구 사항
- Python 3.10 이상
- Windows 10/11
- USB 2.0 이상 포트

### 2. 의존성 설치

```bash
# 가상 환경 생성 (권장)
python -m venv venv

# 가상 환경 활성화 (Windows)
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 🚀 실행 방법

```bash
python main.py
```

## 📁 프로젝트 구조

```
camera_test/
├── PRD.md                 # 제품 요구사항 문서
├── README.md              # 프로젝트 설명서
├── requirements.txt       # Python 의존성
├── main.py               # 애플리케이션 진입점
├── src/
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── detector.py    # 장치 감지 모듈
│   │   ├── streamer.py    # 영상 스트리밍 모듈
│   │   ├── controller.py  # 카메라 제어 모듈
│   │   └── phone_camera.py# 휴대폰 카메라 연결 모듈
│   ├── detection/
│   │   ├── __init__.py
│   │   └── person_detector.py  # YOLO 사람 감지 모듈
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py # 메인 윈도우
│   │   ├── video_widget.py# 영상 표시 위젯
│   │   ├── controls.py    # 제어 UI 컴포넌트
│   │   └── phone_dialog.py# 휴대폰 연결 대화상자
│   └── utils/
│       ├── __init__.py
│       ├── config.py      # 설정 관리
│       └── logger.py      # 로깅 유틸리티
├── resources/
│   ├── icons/            # 아이콘 리소스
│   └── styles/           # 스타일 시트
└── tests/
    └── __init__.py
```

## 📱 지원 장치

| 장치 유형 | 지원 여부 | 비고 |
|----------|----------|------|
| 웹캠 (UVC) | ✅ 완전 지원 | USB 연결 웹캠 |
| Android 스마트폰 | ✅ 지원 | USB 디버깅 활성화 필요 |
| iPhone | ⚠️ 제한적 | 별도 드라이버 필요 |
| DSLR/미러리스 | ✅ 지원 | UVC 호환 모델 |

## 🎮 사용 방법

### USB 웹캠/카메라 사용
1. **장치 연결**: USB 케이블로 카메라/스마트폰을 PC에 연결합니다.
2. **프로그램 실행**: `python main.py`를 실행합니다.
3. **장치 선택**: 드롭다운 메뉴에서 연결된 장치를 선택합니다.
4. **스트리밍 시작**: "시작" 버튼을 클릭합니다.
5. **전체 화면**: 영상을 더블클릭하거나 전체화면 버튼을 누릅니다.

### 📱 휴대폰 카메라 연결 방법

#### 방법 1: DroidCam 앱 사용 (권장)
1. **앱 설치**: Google Play에서 "DroidCam" 앱 설치
2. **같은 Wi-Fi 연결**: PC와 휴대폰이 같은 Wi-Fi에 연결되어 있는지 확인
3. **앱 실행**: DroidCam 앱을 실행하면 IP 주소가 표시됨 (예: 192.168.1.100:4747)
4. **PC에서 연결**:
   - "📱 Phone Camera" 버튼 클릭
   - IP 주소 입력 (예: 192.168.1.100)
   - 포트: 4747, 경로: /video
   - "Connect" 클릭

#### 방법 2: IP Webcam 앱 사용
1. **앱 설치**: Google Play에서 "IP Webcam" 앱 설치
2. **앱 실행**: 하단의 "Start server" 클릭
3. **IP 확인**: 화면에 표시된 IP 주소 확인 (예: 192.168.1.100:8080)
4. **PC에서 연결**:
   - "📱 Phone Camera" 버튼 클릭
   - App Preset에서 "IP Webcam" 선택
   - IP 주소 입력
   - "Connect" 클릭

#### 방법 3: USB 직접 연결 (Android) - 권장! 🔌
Wi-Fi 없이 USB 케이블만으로 휴대폰 카메라를 사용할 수 있습니다.

**사전 준비:**
1. Google Play에서 **DroidCam** 앱 설치
2. 휴대폰 설정 → 휴대전화 정보 → 빌드번호 7번 터치 (개발자 모드 활성화)
3. 설정 → 개발자 옵션 → **USB 디버깅** 활성화
4. [Android SDK Platform-Tools](https://developer.android.com/studio/releases/platform-tools) 설치 (ADB 포함)

**연결 방법:**
1. **USB 케이블 연결**: PC와 휴대폰을 USB 케이블로 연결
2. **USB 디버깅 허용**: 휴대폰에 팝업이 뜨면 "확인" 선택
3. **DroidCam 실행**: 휴대폰에서 DroidCam 앱 실행
4. **PC 프로그램에서 연결**:
   - "📱 Phone Camera" 버튼 클릭
   - "🔌 USB 연결" 탭 선택
   - 장치 목록에서 휴대폰 선택
   - "USB로 연결" 클릭

**작동 원리:**
- ADB(Android Debug Bridge)의 포트 포워딩 기능 사용
- 휴대폰의 카메라 앱 서버를 USB를 통해 PC로 터널링
- Wi-Fi 연결 없이도 안정적인 스트리밍 가능

### 👤 YOLO 사람 감지 사용 방법

1. **스트리밍 시작**: 카메라 또는 휴대폰 연결 후 스트리밍 시작
2. **감지 활성화**: "Person Detection" 체크박스 활성화 또는 `D` 키 누르기
3. **첫 실행 시**: YOLO 모델이 자동으로 다운로드됨 (약 6MB)
4. **결과 확인**:
   - 초록색 박스: 감지된 사람의 바운딩 박스
   - 빨간색 점: 사람의 중심점 좌표
   - 상태바: 감지된 사람 수 표시

**좌표 출력 형식:**
```python
{
    "id": -1,                    # 트래킹 ID (트래킹 비활성화 시 -1)
    "center_x": 640,             # 중심점 X 좌표
    "center_y": 360,             # 중심점 Y 좌표
    "bbox": {
        "x1": 500, "y1": 200,    # 좌상단
        "x2": 780, "y2": 520,    # 우하단
        "width": 280,
        "height": 320
    },
    "confidence": 0.85           # 감지 신뢰도
}
```

### 키보드 단축키

| 단축키 | 기능 |
|--------|------|
| `Space` | 스트리밍 시작/중지 |
| `F11` | 전체 화면 전환 |
| `Esc` | 전체 화면 종료 |
| `S` | 스크린샷 저장 |
| `R` | 녹화 시작/중지 |
| `P` | 휴대폰 카메라 연결 |
| `D` | 사람 감지 켜기/끄기 |
| `Q` | 프로그램 종료 |

## ⚙️ 설정

설정 파일은 `config.json`에 저장됩니다:

```json
{
    "default_resolution": [1920, 1080],
    "default_fps": 30,
    "save_path": "./captures",
    "auto_connect": true
}
```

## 🐛 문제 해결

### 카메라가 감지되지 않을 때
1. USB 케이블이 데이터 전송을 지원하는지 확인
2. 장치 관리자에서 카메라 드라이버 상태 확인
3. 다른 USB 포트에 연결 시도

### Android 스마트폰 연결 문제
1. USB 디버깅 활성화 확인
2. MTP/PTP 모드 대신 "파일 전송" 또는 "웹캠" 모드 선택
3. ADB 드라이버 설치 확인

### 영상이 끊기거나 지연될 때
1. 해상도를 낮춤 (1080p → 720p)
2. 다른 USB 포트 (USB 3.0 권장) 사용
3. 백그라운드 프로그램 종료

### YOLO 감지가 느릴 때
1. GPU 가속 확인 (NVIDIA CUDA 지원 시 자동 사용)
2. 작은 모델 사용 (기본값: yolov8n - 가장 빠름)
3. 해상도를 낮춰서 테스트

### YOLO 모델 로드 실패 (DLL 오류)
**증상**: `DLL 초기화 루틴을 실행할 수 없습니다` 또는 `c10.dll` 오류

**해결 방법**:
1. **자동 폴백**: OpenCV DNN (YOLOv4-tiny)로 자동 전환됩니다
   - 첫 실행 시 모델 파일이 자동 다운로드됩니다 (~25MB)
   - PyTorch 없이도 작동합니다

2. **YOLOv8 사용을 원할 경우**:
   ```bash
   # 기존 torch 제거 후 재설치
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install ultralytics
   ```

3. **Visual C++ 재배포 패키지 설치**:
   - [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 설치

## 📄 라이선스

MIT License

## 🤝 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해 주세요.

---

**Version**: 1.1.0
**Last Updated**: 2026-01-16
**New in 1.1.0**: YOLO 기반 사람 감지 기능 추가
