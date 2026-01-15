from faster_whisper import WhisperModel

class StreamingASR:
    def __init__(self):
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )

    def transcribe_chunk(self, audio_path, lang):
        segments, _ = self.model.transcribe(
            audio_path,
            language=lang,
            vad_filter=True
        )
        return " ".join(seg.text for seg in segments)
