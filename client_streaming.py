import time

def countdown():
    print("Начинайте говорить через...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Говорите!")

# Обратный отсчёт
countdown()


import io, base64, requests, sounddevice as sd, soundfile as sf
from scipy.io.wavfile import write
import numpy as np
from tempfile import NamedTemporaryFile

# ----------------------------------------------------------------------
API_URL       = "http://10.2.6.13:5000/generate"
REC_SECONDS   = 4
REC_RATE      = 24_000         
PROMPT_TEXT   = "ответь на вопрос на русском"
RESPONSE_TYPE = "both"          # "text" | "both"
# ----------------------------------------------------------------------


def record_wav(rate: int = REC_RATE, seconds: int = REC_SECONDS) -> str:
    print(f"Говорите ({seconds} с)…")
    data = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                  dtype="int16")
    sd.wait()
    tmp = NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp.name, rate, data)        
    print(f"Записано: {tmp.name}")
    return tmp.name


def query_kimi(wav_path: str, prompt: str, resp_type: str):
    with open(wav_path, "rb") as f:
        files = {"audio_file": ("voice.wav", f, "audio/wav")}
        data  = {"prompt": prompt, "response_type": resp_type}
        r = requests.post(API_URL, files=files, data=data, timeout=300)
        r.raise_for_status()

    wav_bytes = r.content if resp_type in ("audio", "both") else None
    text      = None

    if resp_type == "text":
        text = r.json().get("text")

    if resp_type == "both":
        b64 = r.headers.get("X-Generated-Text-B64")
        if b64:
            text = base64.b64decode(b64).decode("utf-8")

    return wav_bytes, text


def play_wav_bytes(wav_bytes: bytes):
    buf = io.BytesIO(wav_bytes)
    data, rate = sf.read(buf, dtype="float32")
    sd.play(data, rate)
    sd.wait()


if __name__ == "__main__":
    in_wav = record_wav()
    wav_ans, txt_ans = query_kimi(in_wav, PROMPT_TEXT, RESPONSE_TYPE)

    if txt_ans:
        print("\nМодель ответила текстом:\n", txt_ans)

    if wav_ans:
        print("Воспроизвожу аудио-ответ…")
        play_wav_bytes(wav_ans)
        # если нужно сохранить:
        # with open("reply.wav", "wb") as f: f.write(wav_ans)

