from faster_whisper import WhisperModel

class LiveWhisperASR:
    def __init__(self):
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )

    def transcribe(self, audio_path, language):
        segments, _ = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments)
