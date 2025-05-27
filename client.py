import base64, requests, json

API_URL = "http://10.2.6.13:5000/generate"
WAV_IN  = "recording.wav"
WAV_OUT = "reply.wav"

def ask_kimi(audio_path: str,
             prompt: str = "",
             max_tokens: int = 40,
             response_type: str = "both") -> str | None:
    """Возвращает текст (str) или None, если текста нет."""
    with open(audio_path, "rb") as f:
        files = {"audio_file": ("voice.wav", f, "audio/wav")}
        data  = {
            "prompt": prompt,
            "response_type": response_type,  # text | both
            "max_new_tokens": str(max_tokens),
        }
        resp = requests.post(API_URL, files=files, data=data, timeout=300)
        resp.raise_for_status()

    # --- если запрашивали аудио ---
    if response_type in ("audio", "both"):
        with open(WAV_OUT, "wb") as out:
            out.write(resp.content)
        print(f"Aудио-ответ сохранён в {WAV_OUT}")

    # --- если запрашивали текст ---
    if response_type == "text":
        return resp.json().get("text")

    if response_type == "both":
        b64 = resp.headers.get("X-Generated-Text-B64")
        if b64:
            return base64.b64decode(b64).decode("utf-8")
    return None

if __name__ == "__main__":
    text = ask_kimi(
        audio_path=WAV_IN,
        prompt="ответь на вопрос",
        max_tokens=60,
        response_type="both"   # поменяйте на "text" или "both" когда нужно
    )
    print("Модель ответила текстом:\n", text)