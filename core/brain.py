import base64
import re

from google import genai
from google.genai import types

from core.tts import speak

SENT_END = re.compile(r"(?<=[.!?…])\s+")
MODEL = "gemini-3.5-flash-lite"   

client = genai.Client()   # czyta GEMINI_API_KEY


def ask_kalman(prompt_tekst, image_b64=None, persona_path="kalman_persona.txt"):
    try:
        with open(persona_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "Jesteś łazikiem Kalmanem. Odpowiadaj krótko, po polsku."

    parts = []
    if image_b64:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(image_b64), mime_type="image/jpeg"))
    parts.append(prompt_tekst)

    stream = client.models.generate_content_stream(
        model=MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=400,   # z zapasem, bo "myślenie" wlicza się do limitu
            temperature=0.4,
        ),
    )

    FIRST_CHUNK = re.compile(r"^(.{25,}?[,;:])\s")   # pierwszy przecinek po min. 25 znakach
    END = re.compile(r"^(.{8,}?[.!?…]+)(?:\s+|$)", re.S)

    print("[LLM]: ", end="", flush=True)
    full_text, buffer = "", ""
    first_sent = False

    for chunk in stream:
        token = chunk.text or ""
        print(token, end="", flush=True)
        full_text += token
        buffer += token

        # 1) tylko na początku odpowiedzi: wyślij fragment do pierwszego przecinka
        if not first_sent:
            m = FIRST_CHUNK.match(buffer)
            if m:
                speak(m.group(1))
                buffer = buffer[m.end():]
                first_sent = True

        # 2) potem wysyłaj każde skończone zdanie od razu po kropce
        while (m := END.match(buffer)):
            speak(m.group(1))
            buffer = buffer[m.end():]
            first_sent = True

    if buffer.strip():
        speak(buffer)


    print("\n")
    return full_text