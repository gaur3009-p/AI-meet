from faster_whisper import WhisperModel

class StreamingASR:
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
        return " ".join(seg.text.strip() for seg in segments)
