"""
Voice input — Week 3.

Uses faster-whisper (CTranslate2-based) with the tiny.en model, which is the
right choice for 8GB-RAM CPU systems:
- ~75MB model download
- ~1-3 second transcription latency for short utterances
- ~150MB peak RAM during transcription
- English-only (sufficient for our demo)

Recording uses sounddevice — cross-platform, simple. Press to start, press to
stop. No streaming for now; we record the whole utterance, then transcribe.
"""
from __future__ import annotations

import io
import threading
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class VoiceInput:
    """Captures microphone audio and transcribes via faster-whisper.

    Lazy-loads everything so this module can be imported even if the user
    hasn't installed sounddevice / faster-whisper yet.
    """

    SAMPLE_RATE = 16000     # whisper expects 16kHz mono
    CHANNELS = 1
    DTYPE = "float32"

    def __init__(
        self,
        model_name: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._recording = False

    # ---------------------- model loading ----------------------

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            ) from e
        # int8 quantization keeps RAM use low on CPU
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    # ---------------------- recording ----------------------

    def start_recording(self):
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError(
                "sounddevice not installed. Run: pip install sounddevice"
            ) from e

        if self._recording:
            return

        self._frames = []
        self._recording = True

        def callback(indata, frames, time, status):
            if status:
                print(f"[VoiceInput] {status}")
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            callback=callback,
        )
        self._stream.start()

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the captured audio as a float32 numpy array."""
        if not self._recording or self._stream is None:
            return np.zeros(0, dtype=np.float32)

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False

        if not self._frames:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(self._frames, axis=0).flatten().astype(np.float32)
        self._frames = []
        return audio

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ---------------------- transcription ----------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 16kHz float32 audio array. Returns the recognised text."""
        if audio is None or len(audio) < self.SAMPLE_RATE // 4:    # < 0.25s
            return ""
        self._load_model()

        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,            # greedy = fast on CPU
            vad_filter=True,        # skip silent regions
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text

    def record_and_transcribe(self) -> str:
        """Convenience: stop the running recording and return the transcript."""
        audio = self.stop_recording()
        return self.transcribe(audio)


# ----------------------------------------------------------------------------
# Async helper: run transcription in a background thread to keep UI responsive
# ----------------------------------------------------------------------------

class TranscribeThread(threading.Thread):
    """Runs voice.transcribe(audio) off the GUI thread and calls back with the result."""

    def __init__(self, voice: VoiceInput, audio: np.ndarray, callback: Callable[[str], None]):
        super().__init__(daemon=True)
        self.voice = voice
        self.audio = audio
        self.callback = callback

    def run(self):
        try:
            text = self.voice.transcribe(self.audio)
        except Exception as e:
            text = f"[transcription error: {e}]"
        self.callback(text)
