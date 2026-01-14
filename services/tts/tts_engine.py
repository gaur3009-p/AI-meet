import subprocess
import uuid
import os

class TTSEngine:
    def __init__(self):
        self.model_path = "models/en_US-lessac-medium.onnx"
        os.makedirs("outputs", exist_ok=True)

    def speak(self, text: str) -> str:
        output_path = f"outputs/{uuid.uuid4()}.wav"

        process = subprocess.Popen(
            [
                "piper",
                "--model", self.model_path,
                "--output_file", output_path
            ],
            stdin=subprocess.PIPE,
            text=True
        )

        process.stdin.write(text)
        process.stdin.close()
        process.wait()

        return output_path
