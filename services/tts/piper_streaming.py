import subprocess
import tempfile
import os

class StreamingTTS:
    def __init__(self):
        self.model = "models/en_US-lessac-medium.onnx"

    def speak(self, text):
        if not text.strip():
            return None

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        p = subprocess.Popen(
            ["piper", "--model", self.model, "--output_file", path],
            stdin=subprocess.PIPE,
            text=True
        )
        p.stdin.write(text)
        p.stdin.close()
        p.wait()

        return path
