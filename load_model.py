import os
os.environ["HF_HOME"] = "/app/huggingface_cache"

from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf

print("началась загрузка модели!")
model = KimiAudio(
    model_path="moonshotai/Kimi-Audio-7B-Instruct",
    load_detokenizer=True,
)

print("модель загружена!")

