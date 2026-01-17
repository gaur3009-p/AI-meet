import torch
import soundfile as sf
import tempfile
import os
import numpy as np

from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)

class StreamingTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models from HuggingFace (Kaggle-safe)
        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts"
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)

        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)

        # Use a default speaker embedding (generic voice)
        self.speaker_embeddings = torch.zeros((1, 512)).to(self.device)

    def speak(self, text: str):
        if not text or not text.strip():
            return None

        inputs = self.processor(
            text=text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embeddings,
                vocoder=self.vocoder
            )

        # Save audio
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        sf.write(
            out_path,
            speech.cpu().numpy(),
            samplerate=16000
        )

        return out_path
