import numpy as np
import time

class UtteranceBuffer:
    def __init__(
        self,
        silence_threshold=0.02,     # higher for Gradio
        silence_duration=0.7,       # seconds
        max_utterance_duration=6.0  # safety flush
    ):
        self.buffer = []
        self.start_time = None
        self.last_voice_time = time.time()

        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_utterance_duration = max_utterance_duration

    def add_chunk(self, audio_chunk):
        if self.start_time is None:
            self.start_time = time.time()

        # RMS energy (better than mean abs)
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        now = time.time()

        # Voice detected
        if energy > self.silence_threshold:
            self.last_voice_time = now
            self.buffer.append(audio_chunk)
            return None

        # Silence detected
        silence_time = now - self.last_voice_time
        utterance_time = now - self.start_time

        if self.buffer and (
            silence_time >= self.silence_duration
            or utterance_time >= self.max_utterance_duration
        ):
            utterance = np.concatenate(self.buffer)
            self.buffer = []
            self.start_time = None
            return utterance

        return None
