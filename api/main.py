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
# LANGUAGE MAPS
# =========================
WHISPER_LANG_MAP = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "tam_Taml": "ta",
    "kan_Knda": "kn"
}

# =========================
# CORE FUNCTION
# =========================
def speech_to_speech(audio, src_lang, tgt_lang):
    if audio is None:
        return "", "", None

    sample_rate, audio_data = audio
    temp_audio_path = f"/tmp/input_{uuid.uuid4()}.wav"
    sf.write(temp_audio_path, audio_data, sample_rate)

    # -------------------------
    # ASR (Whisper uses ISO codes)
    # -------------------------
    whisper_lang = WHISPER_LANG_MAP[src_lang]

    original_text = asr.transcribe(
        audio_path=temp_audio_path,
        language=whisper_lang
    )

    if not original_text.strip():
        return "", "", None

    # -------------------------
    # TRANSLATION (NLLB codes)
    # -------------------------
    translated_text = translator.translate(
        text=original_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    # -------------------------
    # TTS
    # -------------------------
    output_audio = tts.speak(translated_text)

    return original_text, translated_text, output_audio


# =========================
# GRADIO UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 Level 1 — Multilingual Speech-to-Speech")

    audio_input = gr.Audio(
        type="numpy",
        label="🎙️ Speak"
    )

    src_lang = gr.Dropdown(
        choices=["eng_Latn", "hin_Deva", "tam_Taml", "kan_Knda"],
        value="hin_Deva",
        label="Source Language"
    )

    tgt_lang = gr.Dropdown(
        choices=["eng_Latn", "hin_Deva", "tam_Taml", "kan_Knda"],
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

# =========================
# LAUNCH (COLAB)
# =========================
demo.launch(share=True)
