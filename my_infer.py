import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"


import soundfile as sf
# Assuming the KimiAudio class is available after installation
from kimia_infer.api.kimia import KimiAudio
import torch # Ensure torch is imported if needed for device placement

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device("cpu")
# Локальный путь к модели 
model_path = "/root/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/a574f67664cb0443ce08fd6827eb7e2170c94140/"

# Загружаем модель из локальной папки
model = KimiAudio(model_path=model_path, load_detokenizer=True)
#model.to(device)  # Пример размещения на устройстве (GPU или CPU)

print("Starting inference...")

import os
sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
        "max_new_tokens" : 5
    }

output_dir = "test_audios/output"
os.makedirs(output_dir, exist_ok=True)
# audio2audio
messages = [
        {
            "role": "user",
            "message_type": "audio",
            "content": "test_audios/privet.wav",
        }
    ]

wav, text = model.generate(messages, **sampling_params, output_type="both")
sf.write(
        os.path.join(output_dir, "output.wav"),
        wav.detach().cpu().view(-1).numpy(),
        24000,
    )
print(">>> output text: ", text)