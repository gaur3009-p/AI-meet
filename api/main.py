import sys, os, uuid
import numpy as np
import soundfile as sf
import gradio as gr

sys.path.insert(0, os.path.abspath(".."))

from services.asr.whisper_live import LiveWhisperASR
from services.translation.nllb_translate import Translator
from services.tts.tts_streaming import StreamingTTS
from utils.phrase_commit import PhraseCommitter

asr = LiveWhisperASR()
translator = Translator()
tts = StreamingTTS()
committer = PhraseCommitter()

LANG_MAP = {
    "hin_Deva": "hi",
    "eng_Latn": "en"
}

audio_buffer = []

def live_pipeline(audio, src_lang, tgt_lang):
    global audio_buffer

    if audio is None:
        return "", "", None

    sr, chunk = audio
    audio_buffer.append(chunk)

    if len(audio_buffer) > 40:  # ~4 seconds
        audio_buffer = audio_buffer[-40:]

    full_audio = np.concatenate(audio_buffer)
    path = f"/tmp/live_{uuid.uuid4()}.wav"
    sf.write(path, full_audio, sr)

    live_text = asr.transcribe(path, LANG_MAP[src_lang])
    phrase = committer.process(live_text)

    if phrase:
        translated = translator.translate(phrase, src_lang, tgt_lang)
        audio_out = tts.speak(translated)
        return live_text, translated, audio_out

    return live_text, "", None


with gr.Blocks() as demo:
    gr.Markdown("## 🔵 Level 2.7 — Live Captions + Translation")

    mic = gr.Audio(type="numpy", streaming=True)
    src = gr.Dropdown(["hin_Deva", "eng_Latn"], value="hin_Deva")
    tgt = gr.Dropdown(["eng_Latn", "hin_Deva"], value="eng_Latn")

    live_txt = gr.Textbox(label="Live Captions")
    trans_txt = gr.Textbox(label="Committed Translation")
    out_audio = gr.Audio(label="Translated Speech")

    mic.stream(
        live_pipeline,
        inputs=[mic, src, tgt],
        outputs=[live_txt, trans_txt, out_audio]
    )

demo.launch(share=True)
