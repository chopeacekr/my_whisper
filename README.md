# Whisper STT Server

> 🎤 **Fast and accurate Speech-to-Text using faster-whisper**

OpenAI의 Whisper 모델을 기반으로 한 고성능 음성 인식 서버입니다. **faster-whisper**를 사용하여 CPU 환경에서도 실용적인 속도를 제공하며, 최신 Python 패키지 매니저인 **uv**를 통해 쉽고 빠르게 설치할 수 있습니다.

---

## 📋 목차

- [특징](#-특징)
- [시스템 요구사항](#-시스템-요구사항)
- [설치 (uv)](#-설치-uv)
- [사용법](#-사용법)
- [API 문서](#-api-문서)
- [성능](#-성능)
- [문제 해결](#-문제-해결)
- [라이센스](#-라이센스)

---

## ✨ 특징

- **초고속 환경 설정**: `uv`를 통해 의존성 설치 및 가상환경 구성 속도 극대화
- **높은 정확도**: OpenAI Whisper 모델 기반 (한국어 90%+ 정확도)
- **빠른 처리**: faster-whisper 최적화로 실시간 대비 4배 빠른 처리
- **다국어 지원**: 99개 언어 즉시 사용 가능 (추가 모델 다운로드 불필요)
- **CPU 최적화**: GPU 없이도 실용적인 속도
- **RESTful API**: FastAPI 기반 표준 HTTP API
- **자동 전처리**: 스테레오→모노, 샘플레이트 변환 자동 처리
- **VAD 내장**: Voice Activity Detection으로 정확도 향상

---

## 💻 시스템 요구사항

### 최소 사양
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+
- **Python**: 3.8 - 3.12 (uv가 자동 관리)
- **RAM**: 1GB (Base 모델 기준)
- **디스크**: 500MB (모델 캐시 포함)

### 권장 사양
- **RAM**: 2GB+
- **CPU**: 4 코어 이상
- **디스크**: 1GB (여러 모델 사용 시)

### GPU 사용 시 (선택)
- **CUDA**: 11.2+
- **GPU RAM**: 2GB+ (Base 모델)

---

## 🚀 설치 (uv)

이 프로젝트는 `uv`를 사용하여 의존성을 관리합니다. 복잡한 가상환경 설정 없이 단 두 줄로 설치가 완료됩니다.

### 1. uv 설치 (최초 1회)

이미 `uv`가 설치되어 있다면 건너뛰세요.

**macOS / Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

### 2. 저장소 클론 및 동기화

프로젝트를 다운로드하고 `uv sync` 명령어를 실행하면, **가상환경 생성부터 라이브러리 설치까지 자동으로 완료**됩니다.

```bash
git clone [https://github.com/chopeacekr/my-whisper.git](https://github.com/chopeacekr/my-whisper.git)
cd my-whisper

# 가상환경 생성 및 의존성 설치 (자동)
uv sync
```

### 3. 모델 다운로드 (자동)

서버 첫 실행 시 모델이 자동으로 다운로드됩니다. 수동 다운로드는 불필요합니다.

---

## 🎯 사용법

### 서버 실행

`uv run` 명령어를 사용하면 가상환경을 별도로 활성화할 필요 없이 바로 서버가 실행됩니다.

```bash
uv run python server_stt.py
```

서버가 `http://localhost:8300`에서 시작됩니다.

### 로그 레벨 조정

`server_stt.py` 파일 상단에서 설정:

```python
VERBOSE = True   # False로 설정하면 최소 로그만
DEBUG = True     # False로 설정하면 상세 정보 숨김
```

### 모델 변경

`server_stt.py`에서 모델 크기 변경:

```python
# tiny: 가장 빠름, 낮은 정확도
# base: 권장 (기본값)
# small: 더 정확, 느림
# medium/large: 최고 정확도, 매우 느림

model = WhisperModel("base", device=device, compute_type=compute_type)
```

### GPU 사용

```python
device = "cuda"  # CPU → CUDA로 변경
compute_type = "float16"  # int8 → float16으로 변경
```

---

## 📡 API 문서

### 1. Health Check

서버 상태를 확인합니다.

**Endpoint**: `GET /health`

**응답 예시**:
```json
{
  "status": "ok",
  "device": "cpu",
  "model": "base",
  "loaded_languages": ["KR", "EN", "JP", "ZH", "FR", "DE", "ES", "RU"]
}
```

**cURL 예시**:
```bash
curl http://localhost:8300/health
```

---

### 2. 음성 인식

오디오를 텍스트로 변환합니다.

**Endpoint**: `POST /recognize`

**요청 본문**:
```json
{
  "audio_b64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "lang": "KR",
  "sample_rate": 16000
}
```

**파라미터**:
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `audio_b64` | string | ✅ | Base64 인코딩된 WAV 오디오 |
| `lang` | string | ✅ | 언어 코드 (KR, EN, JP, ZH, FR, DE, ES, RU) |
| `sample_rate` | integer | ❌ | 샘플링 레이트 (기본값: 16000) |

**응답 예시**:
```json
{
  "text": "음성을 텍스트로 변환해주는 모델 추천해줘",
  "language": "ko",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": " 음성을 텍스트로 변환해주는 모델 추천해줘"
    }
  ]
}
```

**Python 예시**:
```python
import requests
import base64

# 오디오 파일 읽기
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

# Base64 인코딩
audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

# API 요청
response = requests.post(
    "http://localhost:8300/recognize",
    json={
        "audio_b64": audio_b64,
        "lang": "KR",
        "sample_rate": 16000
    },
    timeout=60
)

result = response.json()
print(f"인식된 텍스트: {result['text']}")
```

---

## 📊 성능

### 처리 속도 (Base 모델, CPU)

| 음성 길이 | 처리 시간 | 실시간 배율 |
|-----------|-----------|-------------|
| 5초 | 1.3초 | 3.8x |
| 10초 | 2.5초 | 4.0x |
| 30초 | 7.2초 | 4.2x |
| 60초 | 14.8초 | 4.1x |

### 모델별 비교

| 모델 | 파라미터 | 크기 | 처리 속도 | 정확도 (한국어) |
|------|----------|------|-----------|-----------------|
| Tiny | 39M | 74MB | 6x 실시간 | 89.3% |
| **Base** | 74M | 142MB | **4x 실시간** | **90%+** |
| Small | 244M | 466MB | 2.5x 실시간 | 97.8% |
| Medium | 769M | 1.5GB | 1.2x 실시간 | 98.5% |
| Large-v3 | 1550M | 2.9GB | 0.8x 실시간 | 99.1% |

> 💡 **권장**: Base 모델이 속도와 정확도의 최적 균형점

---

## 🔧 설정

### 1. 포트 변경

`server_stt.py` 마지막 줄:

```python
uvicorn.run(app, host="0.0.0.0", port=8300)  # 원하는 포트로 변경
```

### 2. VAD 파라미터 조정

`server_stt.py`의 `transcribe` 함수:

```python
segments, info = model.transcribe(
    audio_data,
    language=whisper_lang,
    vad_filter=True,
    vad_parameters={
        "threshold": 0.5,      # 0.3-0.7 (낮을수록 민감)
        "min_speech_duration_ms": 250,  # 최소 음성 길이
        "min_silence_duration_ms": 100,  # 최소 무음 길이
    }
)
```

---

## 🐛 문제 해결

### 1. 'uv' 명령어를 찾을 수 없음

설치 후 터미널을 재시작하지 않아서 발생합니다. 터미널을 껐다 켜거나 `source $HOME/.cargo/env` (Linux/Mac)를 실행하세요.

### 2. 패키지 설치 오류

`uv sync` 실행 중 오류가 발생하면 캐시를 초기화해 보세요:

```bash
uv cache clean
uv sync
```

### 3. GPU 인식 불가

CUDA가 설치되어 있음에도 인식이 안 된다면, PyTorch를 CUDA 버전으로 명시적 설치가 필요할 수 있습니다. `pyproject.toml` 설정을 확인하거나 아래 명령어로 재설치합니다:

```bash
uv pip install torch --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

---

## 📁 프로젝트 구조

```
my-whisper/
├── server_stt.py           # FastAPI 서버 메인
├── pyproject.toml          # 프로젝트 및 의존성 설정 (uv)
├── uv.lock                 # 의존성 버전 잠금 파일
├── README.md               # 이 문서
├── .python-version         # Python 버전 명시
└── models/                 # 모델 캐시 (자동 생성)
    └── base/
```

---

## 🤝 기여

버그 리포트, 기능 제안, Pull Request 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

Whisper 모델 자체는 OpenAI의 라이센스를 따릅니다.

---

## 👤 작성자

**조화평**

- GitHub: [@chopeacekr](https://github.com/chopeacekr)
- Email: chopeacekr@gmail.com

---

> 📅 최종 업데이트: 2024.12.04
>
> 🏷️ **태그**: #STT #Whisper #FastAPI #uv #음성인식 #AI