
import sys, os, uuid, time
import soundfile as sf
import gradio as gr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
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

# FORCE FLUSH TIMER
last_flush_time = 0.0
FLUSH_INTERVAL = 2.0   # seconds

def stream_pipeline(audio, src_lang, tgt_lang):
    global last_flush_time

    print("🔥 stream_pipeline called")

    if audio is None:
        print("❌ audio is None")
        return "[waiting for audio]", "", None

    sr, chunk = audio
    print("✅ audio received | sr:", sr, "| samples:", len(chunk))

    now = time.time()
    if now - last_flush_time < FLUSH_INTERVAL:
        return "[buffering utterance...]", "", None

    last_flush_time = now

    # save chunk
    wav_path = f"/tmp/utt_{uuid.uuid4()}.wav"
    sf.write(wav_path, chunk, sr)

    # ASR
    text = asr.transcribe_chunk(wav_path, LANG_MAP[src_lang])
    if not text.strip():
        return "[no speech detected]", "", None

    # TRANSLATION
    translated = translator.translate(text, src_lang, tgt_lang)

    # TTS
    audio_out = tts.speak(translated)

    return text, translated, audio_out

with gr.Blocks() as demo:
    gr.Markdown("## 🔵 Level 2.5 — Utterance Buffered Streaming (Debug Mode)")

    mic = gr.Audio(
        type="numpy",
        streaming=True,        
        label="🎙️ Live Microphone"
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

    utter_txt = gr.Textbox(label="Utterance Text")
    trans_txt = gr.Textbox(label="Translated Text")
    out_audio = gr.Audio(label="Translated Speech")

    mic.stream(
        fn=stream_pipeline,
        inputs=[mic, src, tgt],
        outputs=[utter_txt, trans_txt, out_audio]
    )

demo.launch(share=True)
