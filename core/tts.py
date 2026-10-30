import queue, re, threading, time

import numpy as np
import sounddevice as sd
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"      # George
# MODEL_ID = "eleven_flash_v2_5"
MODEL_ID = "eleven_multilingual_v2"
# MODEL_ID = "eleven_v3"
SR = 24000                              

client = ElevenLabs()                  
SETTINGS = VoiceSettings(
    stability=0.2,          # more emotions 
    similarity_boost=0.75,
    speed=1.05,
)

# PRONUNCIATION = {
#     r"\bAGH\b": "a gie ha",
#     r"\bURC\b": "u er ce",
#     r"\bERC\b": "e er ce",
# }

_text_q = queue.Queue()
_audio_q = queue.Queue()
_prev_text = ""

_stream = sd.OutputStream(samplerate=SR, channels=1, dtype="int16", device="pipewire")
_stream.start()


def _clean(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[*_#`~]", "", text)
    # for pattern, spoken in PRONUNCIATION.items():
    #     text = re.sub(pattern, spoken, text)
    return text.strip()


def _trim(a, thr=300):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    return a[max(idx[0] - int(0.03 * SR), 0): idx[-1] + int(0.08 * SR)]


def _synth(text):
    global _prev_text
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format="pcm_24000",
        # language_code="pl",             # wymusza polski (usuń, jeśli SDK zgłosi błąd)
        voice_settings=SETTINGS,
        previous_text=_prev_text,       # płynniejsza intonacja między zdaniami
    )
    _prev_text = text
    return np.frombuffer(b"".join(audio), dtype=np.int16)


def _synth_worker():
    while True:
        text = _text_q.get()
        try:
            if text:
                pcm = _synth(text)
                if len(pcm):
                    _audio_q.put(pcm)
        except Exception as e:
            print(f"[TTS error] {e}")
        finally:
            _text_q.task_done()


def _play_worker():
    while True:
        audio = _audio_q.get()
        try:
            _stream.write(_trim(audio).reshape(-1, 1))
        finally:
            _audio_q.task_done()


threading.Thread(target=_synth_worker, daemon=True).start()
threading.Thread(target=_play_worker, daemon=True).start()


def speak(text):
    _text_q.put(_clean(text))


def wait_done():
    """Block until everything queued has been spoken."""
    _text_q.join()
    _audio_q.join()
    time.sleep(0.3)