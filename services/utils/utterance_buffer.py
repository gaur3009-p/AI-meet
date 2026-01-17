import numpy as np
import time

class UtteranceBuffer:
    def __init__(self, silence_threshold=0.02, silence_duration=0.6):
        self.audio_chunks = []
        self.last_voice_time = time.time()
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

    def add_chunk(self, chunk):
        energy = np.sqrt(np.mean(chunk ** 2))
        now = time.time()

        if energy > self.silence_threshold:
            self.last_voice_time = now
            self.audio_chunks.append(chunk)
            return None

        if now - self.last_voice_time >= self.silence_duration:
            utterance = np.concatenate(self.audio_chunks)
            self.audio_chunks = []
            return utterance

        return None
