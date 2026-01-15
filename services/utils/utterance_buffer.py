import numpy as np
import time

class UtteranceBuffer:
    def __init__(self, silence_threshold=0.01, silence_duration=0.6):
        self.buffer = []
        self.last_voice_time = time.time()
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

    def add_chunk(self, audio_chunk):
        energy = np.mean(np.abs(audio_chunk))
        now = time.time()

        if energy > self.silence_threshold:
            self.last_voice_time = now
            self.buffer.append(audio_chunk)
            return None

        if now - self.last_voice_time >= self.silence_duration and self.buffer:
            utterance = np.concatenate(self.buffer)
            self.buffer = []
            return utterance

        return None
