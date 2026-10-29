import re
import ollama
from core.tts import speak

SENT_END = re.compile(r"(?<=[.!?…])\s+")

def ask_kalman(prompt_tekst, image_b64=None, persona_path="kalman_persona.txt"):
    try:
        with open(persona_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "Jesteś łazikiem Kalmanem. Odpowiadaj krótko, po polsku."

    user_msg = {"role": "user", "content": prompt_tekst}
    if image_b64:
        user_msg["images"] = [image_b64]

    stream = ollama.chat(
        model="qwen3.5",
        messages=[{"role": "system", "content": system_prompt}, user_msg],
        stream=True,
        think=False,
        options={
        "num_ctx": 2048,               
        "num_predict": 120,            
        "temperature": 0.4,
        },

    )

    print("[LLM]: ", end="", flush=True)
    full_text, buffer = "", ""
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_text += token
        buffer += token

        parts = SENT_END.split(buffer)
        for sentence in parts[:-1]:      
            speak(sentence)
        buffer = parts[-1]

    if buffer.strip():
        speak(buffer)

    print("\n")
    return full_text