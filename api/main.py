# =========================
# FIX PYTHON PATH (COLAB)
# =========================
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================
# STANDARD IMPORTS
# =========================
import gradio as gr
import shutil
import uuid
import soundfile as sf
import tempfile

# =========================
# PROJECT IMPORTS
# =========================
from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translate import Translator
from services.tts.tts_engine import TTSEngine

# =========================
# INITIALIZE SERVICES
# =========================
asr = WhisperASR()
translator = Translator()
tts = TTSEngine()

os.makedirs("outputs", exist_ok=True)

# =========================
# CORE PIPELINE FUNCTION
# =========================
def speech_to_speech(
    input_audio,
    src_lang,
    tgt_lang
):
    """
    input_audio: (sample_rate, numpy_array)
    """

    if input_audio is None:
        return "", "", None

    # -------------------------
    # Save input audio
    # -------------------------
    sample_rate, audio_data = input_audio
    temp_audio_path = f"/tmp/input_{uuid.uuid4()}.wav"
    sf.write(temp_audio_path, audio_data, sample_rate)

    # -------------------------
    # ASR
    # -------------------------
    original_text = asr.transcribe(
        audio_path=temp_audio_path,
        language=src_lang
    )

    if not original_text.strip():
        return "", "", None

    # -------------------------
    # TRANSLATION
    # -------------------------
    translated_text = translator.translate(
        text=original_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    # -------------------------
    # TTS (Piper)
    # -------------------------
    output_audio_path = tts.speak(translated_text)

    return original_text, translated_text, output_audio_path


# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="Level 1: Multilingual Speech-to-Speech") as demo:
    gr.Markdown(
        """
        ## 🌍 Level 1 — Multilingual Speech-to-Speech (Python 3.12)

        **Pipeline:**  
        Speech → Text (Whisper) → Translation (NLLB) → Speech (Piper)

        ✔ Offline  
        ✔ Python 3.12 compatible  
        ✔ Enterprise-safe foundation
        """
    )

    with gr.Row():
        mic = gr.Audio(
            source="microphone",
            type="numpy",
            label="🎙️ Speak Here"
        )

    with gr.Row():
        src_lang = gr.Dropdown(
            choices=[
                "eng_Latn",
                "hin_Deva",
                "kan_Knda",
                "tam_Taml"
            ],
            value="hin_Deva",
            label="Source Language"
        )

        tgt_lang = gr.Dropdown(
            choices=[
                "eng_Latn",
                "hin_Deva",
                "kan_Knda",
                "tam_Taml"
            ],
            value="eng_Latn",
            label="Target Language"
        )

    run_btn = gr.Button("▶️ Translate & Speak")

    original_text = gr.Textbox(label="📝 Transcribed Text")
    translated_text = gr.Textbox(label="🌐 Translated Text")
    output_audio = gr.Audio(label="🔊 Output Speech")

    run_btn.click(
        fn=speech_to_speech,
        inputs=[mic, src_lang, tgt_lang],
        outputs=[original_text, translated_text, output_audio]
    )

# =========================
# LAUNCH APP
# =========================
demo.launch()
