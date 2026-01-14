from TTS.api import TTS
import uuid

class TTSEngine:
    def __init__(self):
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=True
        )

    def speak(self, text, speaker_wav, out_dir="outputs"):
        out_path = f"{out_dir}/{uuid.uuid4()}.wav"
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            file_path=out_path
        )
        return out_path
