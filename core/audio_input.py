import ctypes
import numpy as np
import speech_recognition as sr
import ctypes, glob, os
import nvidia.cublas.lib, nvidia.cudnn.lib
# Shit to load it on GPU :/
for pkg in (nvidia.cublas.lib, nvidia.cudnn.lib):
    for so in sorted(glob.glob(os.path.join(list(pkg.__path__)[0], "*.so*"))):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

from faster_whisper import WhisperModel

print("Ładowanie modelu")
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.9         
recognizer.non_speaking_duration = 0.6   
recognizer.phrase_threshold = 0.1
recognizer.dynamic_energy_threshold = True

_mic = None


def _get_mic():
    """Open the mic and calibrate once, reuse afterwards."""
    global _mic
    if _mic is None:
        mic = sr.Microphone(device_index=10, sample_rate=16000)
        mic.__enter__()
        print("Calibrating background noise...")
        recognizer.adjust_for_ambient_noise(mic, duration=1.0)
        print("Done. Start talking")
        _mic = mic
    return _mic


def _transcribe(audio_data):
    raw = audio_data.get_raw_data()  # 16 kHz, 16-bit
    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_np,
        beam_size=2,
        language="pl",
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="Rozmowa po polsku, Kalman, łazik, AGH Space Systems",
    )
    return " ".join(s.text for s in segments).strip()


def listen():
    """Block until a phrase is heard, return its transcription"""
    mic = _get_mic()
    try:
        audio = recognizer.listen(mic, timeout=10, phrase_time_limit=15)
    except sr.WaitTimeoutError:
        return ""
    return _transcribe(audio)


if __name__ == "__main__":
    while True:
        text = listen()
        if text:
            print(f"-> {text} <-")