from faster_whisper import WhisperModel

class WhisperASR:
    def __init__(self):
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )

    def transcribe(self, audio_path, language=None):
        segments, _ = self.model.transcribe(
            audio_path,
            language=language
        )
        return " ".join([seg.text for seg in segments])
