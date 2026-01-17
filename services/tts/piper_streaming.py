import subprocess
import tempfile
import os

class StreamingTTS:
    def __init__(self):
        self.model_path = "models/en_US-lessac-medium.onnx"

    def speak(self, text):
        if not text.strip():
            return None

        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        p = subprocess.Popen(
            ["piper", "--model", self.model_path, "--output_file", out_path],
            stdin=subprocess.PIPE,
            text=True
        )
        p.stdin.write(text)
        p.stdin.close()
        p.wait()

        return out_path
