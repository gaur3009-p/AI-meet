import subprocess
import uuid
import os

class TTSEngine:
    def __init__(self, model_path="models/en_US-lessac-medium.onnx"):
        self.model_path = model_path
        os.makedirs("outputs", exist_ok=True)

    def speak(self, text):
        output_path = f"outputs/{uuid.uuid4()}.wav"

        cmd = [
            "piper",
            "--model", self.model_path,
            "--output_file", output_path
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process.stdin.write(text)
        process.stdin.close()
        process.wait()

        return output_path
