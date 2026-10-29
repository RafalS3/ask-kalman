from core.audio_input import listen
from core.vision import get_frame   
from core.brain import ask_kalman
from core.tts import wait_done

VISION_WORDS = ("widzisz", "zobacz", "patrz", "spójrz", "co to", "opisz", "na obrazie", "przede mną")

def wants_vision(text):
    t = text.lower()
    return any(w in t for w in VISION_WORDS)

while True:
    text = listen()
    print(text)
    if not text:
        continue
    image = get_frame() if wants_vision(text) else None
    ask_kalman(prompt_tekst=text, image_b64=image)
    wait_done()