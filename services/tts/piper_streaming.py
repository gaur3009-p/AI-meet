import torch
import soundfile as sf
import tempfile
import os

from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)

class StreamingTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts"
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)

        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)

        self.speaker = torch.zeros((1, 512)).to(self.device)

    def speak(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker,
                vocoder=self.vocoder
            )

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, speech.cpu().numpy(), 16000)

        return path
