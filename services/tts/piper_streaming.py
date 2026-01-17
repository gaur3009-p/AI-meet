import subprocess
import tempfile
import os
import urllib.request

class StreamingTTS:
    def __init__(self):
        # Colab-safe local directory
        self.voice_dir = "/content/piper_voices"
        self.voice_name = "en_US-lessac-medium.onnx"
        self.model_path = os.path.join(self.voice_dir, self.voice_name)

        os.makedirs(self.voice_dir, exist_ok=True)

        # Auto-download voice if missing (NO piper.download_voices)
        if not os.path.exists(self.model_path):
            self._download_voice_direct()

        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"Piper voice still missing at {self.model_path}"
            )

    def _download_voice_direct(self):
        """
        Direct download from official Piper GitHub.
        This works reliably in Google Colab.
        """
        print("🔽 Downloading Piper voice directly (Colab-safe)...")

        url = (
            "https://github.com/rhasspy/piper/releases/download/"
            "v1.2.0/en_US-lessac-medium.onnx"
        )

        try:
            urllib.request.urlretrieve(url, self.model_path)
        except Exception as e:
            raise RuntimeError(
                "Failed to download Piper voice via direct URL"
            ) from e

        print("✅ Piper voice downloaded successfully.")

    def speak(self, text: str):
        if not text or not text.strip():
            return None

        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        process = subprocess.Popen(
            [
                "piper",
                "--model", self.model_path,
                "--output_file", out_path
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process.stdin.write(text)
        process.stdin.close()
        process.wait()

        # Safety check
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print("⚠️ Piper produced empty audio.")
            return None

        return out_path
