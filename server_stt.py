"""
Whisper STT Server
Fast and accurate speech-to-text using faster-whisper
"""

import base64
import io
import time
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from faster_whisper import WhisperModel
from pydantic import BaseModel

# ================================
# 설정
# ================================
VERBOSE = True   # False: 최소 로그만
DEBUG = True     # False: 상세 정보 숨김

# ================================
# FastAPI 앱 초기화
# ================================
app = FastAPI(
    title="Whisper STT Server",
    description="Speech-to-Text using faster-whisper",
    version="1.0.0"
)

# ================================
# 전역 변수
# ================================
models = {}  # 언어별 모델 캐시 (여기서는 단일 모델 사용)
device = "cpu"  # "cuda" or "cpu"
compute_type = "int8"  # "float16" (GPU) or "int8" (CPU)

# ================================
# 언어 코드 매핑
# ================================
LANGUAGE_MAP = {
    "KR": "ko",
    "EN": "en",
    "JP": "ja",
    "ZH": "zh",
    "FR": "fr",
    "DE": "de",
    "ES": "es",
    "RU": "ru",
}

# ================================
# 모델 로딩
# ================================
def load_model():
    """Whisper 모델 로드 (base 모델 사용 - tiny보다 정확함)"""
    global models
    
    if VERBOSE:
        print("=" * 60)
        print("🚀 Whisper STT Server Starting...")
        print(f"ℹ️  Device: {device}")
        print(f"ℹ️  Compute Type: {compute_type}")
        print("=" * 60)
    
    try:
        if VERBOSE:
            print("📦 Loading Whisper base model...")
        
        start_time = time.time()
        
        # ✅ base 모델 로드 (tiny → base로 변경하여 정확도 향상)
        # tiny: 가장 빠르지만 정확도 낮음
        # base: 속도와 정확도의 균형
        # small/medium/large: 더 정확하지만 느림
        model = WhisperModel("base", device=device, compute_type=compute_type)
        models["default"] = model
        
        elapsed = time.time() - start_time
        
        if VERBOSE:
            print(f"✅ Model loaded successfully in {elapsed:.2f}s")
            print("=" * 60)
            print("✅ Server ready to transcribe speech!")
            print("=" * 60)
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

# ================================
# API 모델
# ================================
class RecognizeRequest(BaseModel):
    audio_b64: str
    lang: str = "KR"
    sample_rate: int = 16000

class RecognizeResponse(BaseModel):
    text: str
    language: str
    segments: Optional[list] = None

class HealthResponse(BaseModel):
    status: str
    device: str
    model: str
    loaded_languages: list

# ================================
# API 엔드포인트
# ================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "ok",
        "device": device,
        "model": "base",  # tiny → base
        "loaded_languages": list(LANGUAGE_MAP.keys())
    }

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(request: RecognizeRequest):
    """
    음성 인식 수행
    
    Args:
        request: RecognizeRequest
            - audio_b64: Base64 인코딩된 WAV 오디오
            - lang: 언어 코드 (KR, EN, JP, ZH, FR, DE, ES, RU)
            - sample_rate: 샘플링 레이트 (기본 16000)
    
    Returns:
        RecognizeResponse
            - text: 인식된 텍스트
            - language: 감지된 언어
            - segments: 세그먼트 정보 (선택)
    """
    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"🎤 New recognition request: lang={request.lang}")
    
    start_time = time.time()
    
    try:
        # 1. Base64 디코딩
        audio_bytes = base64.b64decode(request.audio_b64)
        
        if DEBUG:
            print(f"📊 Audio size: {len(audio_bytes)} bytes")
        
        # 2. 오디오 데이터 로드
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        if DEBUG:
            print(f"📊 Raw audio dtype: {audio_data.dtype}, shape: {audio_data.shape}")
            print(f"📊 Raw audio info: sr={sr}Hz, duration={len(audio_data)/sr:.2f}s")

        # ✅ 스테레오 → 모노 변환 (채널이 여러 개인 경우 평균)
        if audio_data.ndim > 1:
            if DEBUG:
                print(f"🔄 Converting stereo ({audio_data.shape[1]} channels) to mono")
            audio_data = audio_data.mean(axis=1)

        # ✅ float32로 캐스팅 (Whisper는 float32를 기대함)
        if audio_data.dtype != np.float32:
            if DEBUG:
                print(f"🔄 Converting {audio_data.dtype} to float32")
            audio_data = audio_data.astype(np.float32)

        # ✅ 샘플레이트 확인 (Whisper는 내부적으로 16kHz를 사용)
        if sr != 16000:
            if DEBUG:
                print(f"⚠️  Sample rate is {sr}Hz (Whisper expects 16kHz)")
                print(f"    Audio will be resampled internally by Whisper")

        if DEBUG:
            print(f"📊 After convert: dtype={audio_data.dtype}, shape={audio_data.shape}")
            print(f"📊 Audio stats: min={audio_data.min():.4f}, max={audio_data.max():.4f}, mean={audio_data.mean():.4f}")

        # 3. 언어 코드 변환
        whisper_lang = LANGUAGE_MAP.get(request.lang, "ko")
        
        if DEBUG:
            print(f"🌍 Language: {request.lang} -> {whisper_lang}")
        
        # 4. 모델 가져오기
        model = models.get("default")
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # 5. 음성 인식 수행
        if DEBUG:
            print("🎯 Starting transcription...")
        
        transcribe_start = time.time()
        
        # ✅ 개선된 파라미터 설정
        segments, info = model.transcribe(
            audio_data,
            language=whisper_lang,
            vad_filter=True,           # Voice Activity Detection
            vad_parameters={
                "threshold": 0.5,      # VAD threshold (낮을수록 민감)
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
            },
            beam_size=5,               # Beam search 크기
            best_of=5,                 # 후보 개수
            temperature=0.0,           # 결정적 출력
            condition_on_previous_text=False,  # 이전 텍스트에 의존하지 않음
            initial_prompt=None,       # 초기 프롬프트 없음
            word_timestamps=False,     # 단어별 타임스탬프 불필요
        )
        
        # 6. 결과 수집
        full_text = ""
        segment_list = []
        
        for segment in segments:
            full_text += segment.text
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        
        transcribe_time = time.time() - transcribe_start
        
        if DEBUG:
            print(f"✅ Transcription completed in {transcribe_time:.2f}s")
            print(f"📝 Detected language: {info.language} (probability: {info.language_probability:.2f})")
            print(f"📝 Number of segments: {len(segment_list)}")
            print(f"📝 Result: '{full_text.strip()}'")
        
        # ✅ 빈 결과 경고
        if not full_text.strip():
            if VERBOSE:
                print("⚠️  Warning: Empty transcription result")
                print("    This may happen if:")
                print("      - Audio is too quiet")
                print("      - Audio contains no speech")
                print("      - Audio quality is too poor")
        
        # 7. 응답 생성
        total_time = time.time() - start_time
        
        if VERBOSE:
            print(f"✅ Request completed in {total_time:.2f}s")
            if DEBUG:
                print(f"   Breakdown:")
                print(f"     Transcription: {transcribe_time:.2f}s")
            print("="*60)
        
        return RecognizeResponse(
            text=full_text.strip(),
            language=info.language,
            segments=segment_list if DEBUG else None
        )
    
    except Exception as e:
        if VERBOSE:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("="*60)
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# 서버 시작 이벤트
# ================================
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_model()

# ================================
# 메인
# ================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)