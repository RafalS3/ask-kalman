import asyncio, queue, re, subprocess, threading

import edge_tts
import numpy as np
import sounddevice as sd

VOICE = "pl-PL-MarekNeural"   
RATE = "+0%"                  # tempo, np. "+10%"
SR = 48000

_text_q = queue.Queue()
_audio_q = queue.Queue()


def _clean(text):
    text = re.sub(r"\[.*?\]", "", text)   # np. [smile]
    text = re.sub(r"[*_#`~]", "", text)   # markdown
    return text.strip()


async def _fetch(text):
    data = b""
    async for c in edge_tts.Communicate(text, VOICE, rate=RATE).stream():
        if c["type"] == "audio":
            data += c["data"]
    return data


def _to_pcm(mp3):
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
        input=mp3, capture_output=True,
    )
    return np.frombuffer(p.stdout, dtype=np.int16)


def _synth_worker():
    loop = asyncio.new_event_loop()
    while True:
        text = _text_q.get()
        try:
            if text:
                mp3 = loop.run_until_complete(_fetch(text))
                if mp3:
                    _audio_q.put(_to_pcm(mp3))
        except Exception as e:
            print(f"[TTS error] {e}")
        finally:
            _text_q.task_done()


def _play_worker():
    while True:
        audio = _audio_q.get()
        try:
            sd.play(audio, SR, device="pipewire")
            sd.wait()
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