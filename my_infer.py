import soundfile as sf
# Assuming the KimiAudio class is available after installation
from kimia_infer.api.kimia import KimiAudio
import torch # Ensure torch is imported if needed for device placement

device = "cuda" if torch.cuda.is_available() else "cpu"

# Локальный путь к модели 
model_path = "/root/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/a574f67664cb0443ce08fd6827eb7e2170c94140/"

# Загружаем модель из локальной папки
model = KimiAudio(model_path=model_path, load_detokenizer=True)
model.to(device)  # Пример размещения на устройстве (GPU или CPU)

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

print("Starting inference...")