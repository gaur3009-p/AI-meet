import sys, os, uuid, time
import numpy as np
import soundfile as sf
import gradio as gr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from services.asr.whisper_streaming import StreamingASR
from services.translation.nllb_translate import Translator
from services.tts.piper_streaming import StreamingTTS
from services.utils.phrase_committer import PhraseCommitter

# INIT
asr = StreamingASR()
translator = Translator()
tts = StreamingTTS()
committer = PhraseCommitter(min_words=8)

LANG_MAP = {
    "hin_Deva": "hi",
    "eng_Latn": "en"
}

rolling_audio = []

def stream_pipeline(audio, src_lang, tgt_lang):
    global rolling_audio

    if audio is None:
        return "", "", None

    sr, chunk = audio
    rolling_audio.append(chunk)

    # keep last ~3 seconds (30 x 100ms chunks approx)
    if len(rolling_audio) > 30:
        rolling_audio = rolling_audio[-30:]

    full_audio = np.concatenate(rolling_audio)
    wav_path = f"/tmp/roll_{uuid.uuid4()}.wav"
    sf.write(wav_path, full_audio, sr)

    # ASR on rolling buffer
    full_text = asr.transcribe(
        wav_path,
        LANG_MAP[src_lang]
    )

    phrase = committer.extract_phrase(full_text)

    if phrase is None:
        return full_text, "", None

    # TRANSLATION (PHRASE ONLY)
    translated = translator.translate(
        phrase,
        src_lang,
        tgt_lang
    )

    # TTS
    audio_out = tts.speak(translated)

    return full_text, translated, audio_out


with gr.Blocks() as demo:
    gr.Markdown("## 🔵 Level 2.5 — Phrase-Level Live Translation")

    mic = gr.Audio(
        type="numpy",
        streaming=True,
        label="🎙️ Speak continuously"
    )

    src = gr.Dropdown(
        ["hin_Deva", "eng_Latn"],
        value="hin_Deva",
        label="Source language"
    )

    tgt = gr.Dropdown(
        ["eng_Latn", "hin_Deva"],
        value="eng_Latn",
        label="Target language"
    )

    live_txt = gr.Textbox(label="Live transcription")
    trans_txt = gr.Textbox(label="Committed translation")
    out_audio = gr.Audio(label="Translated speech")

    mic.stream(
        stream_pipeline,
        inputs=[mic, src, tgt],
        outputs=[live_txt, trans_txt, out_audio]
    )

demo.launch(share=True)
