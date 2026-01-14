import gradio as gr
import shutil
import os

from services.asr.whisper_asr import WhisperASR
from services.translation.nllb_translate import Translator
from services.tts.tts_engine import TTSEngine

os.makedirs("outputs", exist_ok=True)

asr = WhisperASR()
translator = Translator()
tts = TTSEngine()

def speech_to_speech(
    input_audio,
    speaker_voice,
    src_lang,
    tgt_lang
):
    # Save input audio
    input_path = "temp_input.wav"
    shutil.copy(input_audio, input_path)

    # Save speaker reference
    speaker_path = "temp_speaker.wav"
    shutil.copy(speaker_voice, speaker_path)

    # ASR
    text = asr.transcribe(input_path, src_lang)

    # Translation
    translated_text = translator.translate(
        text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    # TTS (same voice)
    output_audio = tts.speak(
        text=translated_text,
        speaker_wav=speaker_path
    )

    return text, translated_text, output_audio


with gr.Blocks(title="Multilingual Speech-to-Speech AI") as demo:
    gr.Markdown("## 🌍 Multilingual Speech-to-Speech (Same Voice)")

    with gr.Row():
        input_audio = gr.Audio(
            label="Input Speech",
            type="filepath"
        )
        speaker_voice = gr.Audio(
            label="Speaker Voice Reference",
            type="filepath"
        )

    with gr.Row():
        src_lang = gr.Dropdown(
            label="Source Language",
            choices=[
                "eng_Latn",
                "hin_Deva",
                "kan_Knda",
                "tam_Taml"
            ],
            value="hin_Deva"
        )
        tgt_lang = gr.Dropdown(
            label="Target Language",
            choices=[
                "eng_Latn",
                "hin_Deva",
                "kan_Knda",
                "tam_Taml"
            ],
            value="eng_Latn"
        )

    run_btn = gr.Button("Translate & Speak")

    original_text = gr.Textbox(label="Original Text")
    translated_text = gr.Textbox(label="Translated Text")
    output_audio = gr.Audio(label="Output Speech")

    run_btn.click(
        speech_to_speech,
        inputs=[
            input_audio,
            speaker_voice,
            src_lang,
            tgt_lang
        ],
        outputs=[
            original_text,
            translated_text,
            output_audio
        ]
    )

demo.launch()
