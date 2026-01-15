# =========================
# PYTHON PATH FIX (COLAB)
# =========================
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================
# IMPORTS
# =========================
import gradio as gr
import uuid
import soundfile as sf

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
# CORE FUNCTION
# =========================
def speech_to_speech(audio, src_lang, tgt_lang):
    if audio is None:
        return "", "", None

    sample_rate, audio_data = audio
    temp_audio_path = f"/tmp/input_{uuid.uuid4()}.wav"
    sf.write(temp_audio_path, audio_data, sample_rate)

    # ASR
    original_text = asr.transcribe(
        audio_path=temp_audio_path,
        language=src_lang
    )

    if not original_text.strip():
        return "", "", None

    # TRANSLATION
    translated_text = translator.translate(
        text=original_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    # TTS (Python 3.12 safe)
    output_audio = tts.speak(translated_text)

    return original_text, translated_text, output_audio


# =========================
# GRADIO UI (v4+ CORRECT)
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 Level 1 — Multilingual Speech-to-Speech (Python 3.12)")

    audio_input = gr.Audio(
        type="numpy",
        label="🎙️ Speak"
    )

    src_lang = gr.Dropdown(
        choices=["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml"],
        value="hin_Deva",
        label="Source Language"
    )

    tgt_lang = gr.Dropdown(
        choices=["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml"],
        value="eng_Latn",
        label="Target Language"
    )

    run_btn = gr.Button("Translate & Speak")

    original_text = gr.Textbox(label="Transcription")
    translated_text = gr.Textbox(label="Translation")
    output_audio = gr.Audio(label="Output Audio")

    run_btn.click(
        speech_to_speech,
        inputs=[audio_input, src_lang, tgt_lang],
        outputs=[original_text, translated_text, output_audio]
    )

demo.launch(share = True)
