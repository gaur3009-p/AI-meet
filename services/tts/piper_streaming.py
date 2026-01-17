import subprocess
import tempfile
import os

class StreamingTTS:
    def __init__(self):
        self.model_path = os.path.expanduser(
            "~/.local/share/piper/voices/en_US-lessac-medium.onnx"
        )

        if not os.path.exists(self.model_path):
            raise RuntimeError(
                "Piper voice not found. Run `piper.download_voices`"
            )

    def speak(self, text: str):
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
