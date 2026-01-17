import subprocess
import tempfile
import os
import shutil

class StreamingTTS:
    def __init__(self):
        # Piper voice directory (Colab-safe)
        self.voice_dir = os.path.expanduser("~/.local/share/piper/voices")
        self.voice_name = "en_US-lessac-medium.onnx"
        self.model_path = os.path.join(self.voice_dir, self.voice_name)

        os.makedirs(self.voice_dir, exist_ok=True)

        # Auto-download voice if missing (COLAB FIX)
        if not os.path.exists(self.model_path):
            self._download_voice()

        # Final safety check
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"Piper voice not found even after download: {self.model_path}"
            )

    def _download_voice(self):
        """
        Downloads Piper voices using official CLI.
        Works reliably in Google Colab.
        """
        print("🔽 Piper voice not found. Downloading voices...")

        try:
            subprocess.run(
                ["piper.download_voices"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to download Piper voices. "
                "Try running `piper.download_voices` manually."
            ) from e

        print("✅ Piper voice download completed.")

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

        # Safety: ensure audio file is valid
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print("⚠️ Piper produced empty audio.")
            return None

        return out_path
