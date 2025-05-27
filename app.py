import os, io, gc, uuid, tempfile, logging, time, base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import torch, soundfile as sf
from kimia_infer.api.kimia import KimiAudio

# ──────────── базовая настройка логгера ────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("kimi_api")

# ──────────── системные / CUDA настройки ────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ──────────── FastAPI ────────────
app = FastAPI(
    title="Kimi-Audio API",
    description="Принимает WAV, возвращает текст, WAV или оба формата",
    version="0.1",
)

# ──────────── модель и дефолтные параметры ────────────
MODEL_PATH = (
    "/root/.cache/huggingface/hub/"
    "models--moonshotai--Kimi-Audio-7B-Instruct/"
    "snapshots/a574f67664cb0443ce08fd6827eb7e2170c94140"
)
SAMPLING_DEFAULT = dict(
    audio_temperature=0.8,
    audio_top_k=10,
    text_temperature=0.0,
    text_top_k=5,
    audio_repetition_penalty=1.0,
    audio_repetition_window_size=64,
    text_repetition_penalty=1.0,
    text_repetition_window_size=16,
)

@app.on_event("startup")
def _load_model() -> None:
    log.info("Начало загрузки модели из %s", MODEL_PATH)
    t0 = time.perf_counter()
    try:
        app.state.model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=True)
        app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        log.exception("Ошибка при загрузке модели")
        raise
    log.info("Модель загружена за %.2f c, устройство: %s",
             time.perf_counter() - t0, app.state.device)

# ──────────── единый эндпоинт ────────────
@app.post("/generate")
async def generate(
    request: Request,
    audio_file: UploadFile = File(..., description="WAV 24 kHz, 1 канал"),
    prompt: str = Form("", description="Опциональный текст-инструкция"),
    response_type: str = Form(
        "both", description='text | audio | both', regex="^(text|audio|both)$"
    ),
    max_new_tokens: int = Form(20, ge=1, le=128),
):
    req_id = uuid.uuid4().hex[:8]
    log.info("[%s] Запрос от %s | response_type=%s | prompt_len=%d",
             req_id, request.client.host, response_type, len(prompt))

    # —–– сохранение входное аудио во временный файл –––
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio_file.read())
        audio_path = tmp.name
    log.debug("[%s] Входной WAV сохранён во %s (%.1f KB)",
              req_id, audio_path, os.path.getsize(audio_path)/1024)

    sampling = SAMPLING_DEFAULT | {"max_new_tokens": max_new_tokens}
    messages = []
    if prompt:
        messages.append({"role": "user", "message_type": "text", "content": prompt})
    messages.append({"role": "user", "message_type": "audio", "content": audio_path})

    try:
        t0 = time.perf_counter()

        # --- только текст ---
        if response_type == "text":
            _, text_out = app.state.model.generate(
                messages, **sampling, output_type="text"
            )
            log.info("[%s] Сгенерирован текст за %.2f c", req_id, time.perf_counter()-t0)
            return JSONResponse({"text": text_out})

        # --- аудио / оба ---
        wav, text_out = app.state.model.generate(
            messages, **sampling, output_type="both"
        )
        # кодируем UTF-8 - Base64 - ASCII-строка
        text_b64 = base64.b64encode(text_out.encode("utf-8")).decode("ascii")

        log.info("[%s] Сгенерирован WAV за %.2f c (%.2f MB)",
                 req_id, time.perf_counter()-t0, wav.numel()*4/1e6)

        buf = io.BytesIO()
        sf.write(buf, wav.detach().cpu().view(-1).numpy(), 24_000, format="WAV")
        buf.seek(0)

        headers = {
    "Content-Disposition": 'attachment; filename="kimi_output.wav"',
    # Текст закодирован в Base64, поэтому полностью ASCII
    "X-Generated-Text-B64": text_b64}

        return StreamingResponse(buf, media_type="audio/wav", headers=headers)

    except Exception as e:
        log.exception("[%s] Ошибка во время генерации", req_id)
        raise HTTPException(status_code=500, detail="Generation failed") from e

    finally:
        # очистка
        try:
            os.remove(audio_path)
            log.debug("[%s] Временный входной WAV удалён", req_id)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()