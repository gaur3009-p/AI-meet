import sys, os, uuid, tempfile, soundfile as sf
import gradio as gr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from services.asr.whisper_streaming import StreamingASR
from services.translation.nllb_translate import Translator
from services.tts.piper_streaming import StreamingTTS

asr = StreamingASR()
translator = Translator()
tts = StreamingTTS()

LANG_MAP = {
    "hin_Deva": "hi",
    "eng_Latn": "en"
}

def stream_pipeline(audio, src_lang, tgt_lang):
    if audio is None:
        return None, None, None

    sr, data = audio
    path = f"/tmp/chunk_{uuid.uuid4()}.wav"
    sf.write(path, data, sr)

    text = asr.transcribe_chunk(path, LANG_MAP[src_lang])
    if not text.strip():
        return None, None, None

    translated = translator.translate(text, src_lang, tgt_lang)
    audio_out = tts.speak(translated)

    return text, translated, audio_out


with gr.Blocks() as demo:
    gr.Markdown("## 🔴 Level 2 — Real-Time Multilingual Speech")

    mic = gr.Audio(type="numpy", label="Live microphone")

    src = gr.Dropdown(["hin_Deva", "eng_Latn"], value="hin_Deva")
    tgt = gr.Dropdown(["eng_Latn", "hin_Deva"], value="eng_Latn")

    txt = gr.Textbox(label="Live text")
    trn = gr.Textbox(label="Live translation")
    aud = gr.Audio(label="Live speech")

    mic.stream(
        stream_pipeline,
        inputs=[mic, src, tgt],
        outputs=[txt, trn, aud]
    )

demo.launch(share=True)
