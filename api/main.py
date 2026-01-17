import os
import sys
import uuid
import time
import numpy as np
import soundfile as sf
import gradio as gr

# ===============================
# FIX PROJECT ROOT (KAGGLE SAFE)
# ===============================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===============================
# IMPORT SERVICES
# ===============================
from services.asr.whisper_streaming import LiveWhisperASR
from services.translation.nllb_translate import Translator
from services.tts.piper_streaming import StreamingTTS

# ===============================
# INIT MODELS
# ===============================
asr = LiveWhisperASR()
translator = Translator()
tts = StreamingTTS()

LANG_MAP = {
    "hin_Deva": "hi",
    "eng_Latn": "en"
}

# ===============================
# STREAMING STATE
# ===============================
audio_buffer = []
translated_history = []

SILENCE_THRESHOLD = 0.015   # energy threshold
SILENCE_TIME = 0.6          # seconds

last_voice_time = time.time()

# ===============================
# SIMPLE VAD
# ===============================
def is_speech(chunk):
    global last_voice_time
    energy = np.sqrt(np.mean(chunk ** 2))

    if energy > SILENCE_THRESHOLD:
        last_voice_time = time.time()
        return True

    return False

# ===============================
# MAIN LIVE PIPELINE
# ===============================
def live_pipeline(audio, src_lang, tgt_lang):
    global audio_buffer, translated_history, last_voice_time

    if audio is None:
        return "", " ".join(translated_history), None

    sr, chunk = audio

    # ---- VAD gate ----
    speaking = is_speech(chunk)

    if speaking:
        audio_buffer.append(chunk)

        # keep ~2.5 seconds of audio
        if len(audio_buffer) > 25:
            audio_buffer = audio_buffer[-25:]

        full_audio = np.concatenate(audio_buffer)
        temp_path = f"/tmp/live_{uuid.uuid4()}.wav"
        sf.write(temp_path, full_audio, sr)

        # LIVE captions (unstable)
        live_text = asr.transcribe(
            temp_path,
            LANG_MAP[src_lang]
        )
    else:
        live_text = ""  # freeze captions on silence

    # ---- Commit on pause ----
    pause_duration = time.time() - last_voice_time

    if pause_duration >= SILENCE_TIME and audio_buffer:
        full_audio = np.concatenate(audio_buffer)
        commit_path = f"/tmp/commit_{uuid.uuid4()}.wav"
        sf.write(commit_path, full_audio, sr)

        final_text = asr.transcribe(
            commit_path,
            LANG_MAP[src_lang]
        )

        audio_buffer = []  # reset buffer

        if final_text.strip():
            translated = translator.translate(
                final_text,
                src_lang,
                tgt_lang
            )

            translated_history.append(translated)
            audio_out = tts.speak(translated)

            return (
                final_text,
                " ".join(translated_history),
                audio_out
            )

    return live_text, " ".join(translated_history), None


# ===============================
# GRADIO UI
# ===============================
with gr.Blocks() as demo:
    gr.Markdown("## 🔵 Level 2.7 — Live Captions + Translation")

    mic = gr.Audio(
        type="numpy",
        streaming=True,
        label="🎙️ Speak"
    )

    src = gr.Dropdown(
        ["hin_Deva", "eng_Latn"],
        value="hin_Deva",
        label="Source Language"
    )

    tgt = gr.Dropdown(
        ["eng_Latn", "hin_Deva"],
        value="eng_Latn",
        label="Target Language"
    )

    live_txt = gr.Textbox(label="Live Captions")
    trans_txt = gr.Textbox(label="Committed Translation")
    out_audio = gr.Audio(label="Translated Speech")

    mic.stream(
        live_pipeline,
        inputs=[mic, src, tgt],
        outputs=[live_txt, trans_txt, out_audio]
    )

demo.launch(share=True)
