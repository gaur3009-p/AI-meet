# api/main.py
import os
import sys
import uuid
import time
import numpy as np
import soundfile as sf
import gradio as gr
from dataclasses import dataclass
from typing import Optional, Tuple
from queue import Queue
import threading

# ===============================
# FIX PROJECT ROOT
# ===============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===============================
# IMPORT SERVICES
# ===============================
from services.asr.whisper_streaming import LiveWhisperASR
from services.translation.nllb_translate import Translator
from services.tts.voice_cloning_tts import VoiceCloningTTS
from services.utils.stream_manager import StreamManager, Speaker

# ===============================
# INIT MODELS (SINGLETON)
# ===============================
asr = LiveWhisperASR()
translator = Translator()
tts = VoiceCloningTTS()

LANG_MAP = {
    "Hindi": {"code": "hin_Deva", "whisper": "hi"},
    "English": {"code": "eng_Latn", "whisper": "en"},
    "Spanish": {"code": "spa_Latn", "whisper": "es"},
    "French": {"code": "fra_Latn", "whisper": "fr"},
    "German": {"code": "deu_Latn", "whisper": "de"},
    "Chinese": {"code": "zho_Hans", "whisper": "zh"},
    "Japanese": {"code": "jpn_Jpan", "whisper": "ja"},
    "Korean": {"code": "kor_Hang", "whisper": "ko"},
}

# ===============================
# GLOBAL STATE
# ===============================
stream_manager_A = StreamManager(speaker_id="A")
stream_manager_B = StreamManager(speaker_id="B")

# ===============================
# PROCESSING PIPELINE
# ===============================
def process_speaker_stream(
    audio: Optional[Tuple],
    speaker_lang: str,
    listener_lang: str,
    speaker_id: str,
    voice_reference: Optional[str] = None
):
    """
    Process audio stream for one speaker.
    
    Args:
        audio: (sample_rate, numpy_array) from microphone
        speaker_lang: Language speaker is using
        listener_lang: Language to translate to
        speaker_id: "A" or "B"
        voice_reference: Path to reference audio for voice cloning
    """
    manager = stream_manager_A if speaker_id == "A" else stream_manager_B
    
    if audio is None:
        return manager.get_display_state()
    
    sr, chunk = audio
    
    # Convert to float32 and normalize
    if chunk.dtype == np.int16:
        chunk = chunk.astype(np.float32) / 32768.0
    
    # Add chunk to manager
    result = manager.add_audio_chunk(chunk, sr)
    
    # If utterance detected, process it
    if result and result["type"] == "utterance_complete":
        audio_path = result["audio_path"]
        
        # 1. Transcribe
        src_text = asr.transcribe(
            audio_path,
            LANG_MAP[speaker_lang]["whisper"]
        )
        
        if src_text.strip():
            # 2. Translate
            translated = translator.translate(
                src_text,
                LANG_MAP[speaker_lang]["code"],
                LANG_MAP[listener_lang]["code"]
            )
            
            # 3. Clone voice and synthesize
            if voice_reference and os.path.exists(voice_reference):
                audio_output = tts.speak_with_cloning(
                    translated,
                    voice_reference,
                    LANG_MAP[listener_lang]["whisper"]
                )
            else:
                # Fallback to default voice
                audio_output = tts.speak(
                    translated,
                    LANG_MAP[listener_lang]["whisper"]
                )
            
            # Update manager state
            manager.add_translation(src_text, translated)
            
            return manager.get_display_state() + (audio_output,)
    
    # Return current state without new audio
    return manager.get_display_state() + (None,)

# ===============================
# VOICE REFERENCE HANDLER
# ===============================
voice_refs = {"A": None, "B": None}

def save_voice_reference(audio, speaker_id):
    """Save voice reference for cloning."""
    if audio is None:
        return f"No audio provided for Speaker {speaker_id}"
    
    sr, data = audio
    
    # Save reference
    ref_path = f"/tmp/voice_ref_{speaker_id}.wav"
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    sf.write(ref_path, data, sr)
    
    voice_refs[speaker_id] = ref_path
    return f"✓ Voice reference saved for Speaker {speaker_id}"

# ===============================
# GRADIO INTERFACE
# ===============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌍 Real-Time Bidirectional Voice Translation
        **Features:**
        - ✨ Live streaming translation
        - 🎭 Voice cloning for natural output
        - 🔄 Bidirectional conversation
        - 🌐 Multi-language support
        """
    )
    
    with gr.Tabs():
        # ============ SPEAKER A ============
        with gr.Tab("👤 Speaker A"):
            gr.Markdown("### Person A - Setup & Stream")
            
            with gr.Row():
                with gr.Column():
                    lang_a = gr.Dropdown(
                        choices=list(LANG_MAP.keys()),
                        value="Hindi",
                        label="Your Language"
                    )
                    target_lang_a = gr.Dropdown(
                        choices=list(LANG_MAP.keys()),
                        value="English",
                        label="Translate to (for Person B)"
                    )
                
                with gr.Column():
                    voice_ref_a = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Record Voice Sample (5-10 sec for cloning)"
                    )
                    voice_status_a = gr.Textbox(
                        label="Voice Cloning Status",
                        value="No reference recorded"
                    )
                    save_voice_a = gr.Button("💾 Save Voice Reference")
            
            gr.Markdown("---")
            
            mic_a = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="🎙️ Start Speaking"
            )
            
            with gr.Row():
                live_caption_a = gr.Textbox(
                    label="📝 Live Captions (Your Speech)",
                    lines=2
                )
                translation_a = gr.Textbox(
                    label="🌐 Translation History",
                    lines=5
                )
            
            audio_output_a = gr.Audio(
                label="🔊 Translated Output (for Person B)",
                autoplay=True
            )
        
        # ============ SPEAKER B ============
        with gr.Tab("👤 Speaker B"):
            gr.Markdown("### Person B - Setup & Stream")
            
            with gr.Row():
                with gr.Column():
                    lang_b = gr.Dropdown(
                        choices=list(LANG_MAP.keys()),
                        value="English",
                        label="Your Language"
                    )
                    target_lang_b = gr.Dropdown(
                        choices=list(LANG_MAP.keys()),
                        value="Hindi",
                        label="Translate to (for Person A)"
                    )
                
                with gr.Column():
                    voice_ref_b = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Record Voice Sample (5-10 sec for cloning)"
                    )
                    voice_status_b = gr.Textbox(
                        label="Voice Cloning Status",
                        value="No reference recorded"
                    )
                    save_voice_b = gr.Button("💾 Save Voice Reference")
            
            gr.Markdown("---")
            
            mic_b = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="🎙️ Start Speaking"
            )
            
            with gr.Row():
                live_caption_b = gr.Textbox(
                    label="📝 Live Captions (Your Speech)",
                    lines=2
                )
                translation_b = gr.Textbox(
                    label="🌐 Translation History",
                    lines=5
                )
            
            audio_output_b = gr.Audio(
                label="🔊 Translated Output (for Person A)",
                autoplay=True
            )
        
        # ============ CONVERSATION VIEW ============
        with gr.Tab("💬 Conversation"):
            gr.Markdown("### Full Conversation Timeline")
            conversation_view = gr.Textbox(
                label="Complete Dialogue",
                lines=15,
                interactive=False
            )
            refresh_btn = gr.Button("🔄 Refresh Conversation")
    
    # ===============================
    # EVENT HANDLERS
    # ===============================
    
    # Voice reference saving
    save_voice_a.click(
        lambda x: save_voice_reference(x, "A"),
        inputs=[voice_ref_a],
        outputs=[voice_status_a]
    )
    
    save_voice_b.click(
        lambda x: save_voice_reference(x, "B"),
        inputs=[voice_ref_b],
        outputs=[voice_status_b]
    )
    
    # Speaker A streaming
    mic_a.stream(
        lambda audio, sl, tl: process_speaker_stream(
            audio, sl, tl, "A", voice_refs.get("A")
        ),
        inputs=[mic_a, lang_a, target_lang_a],
        outputs=[live_caption_a, translation_a, audio_output_a]
    )
    
    # Speaker B streaming
    mic_b.stream(
        lambda audio, sl, tl: process_speaker_stream(
            audio, sl, tl, "B", voice_refs.get("B")
        ),
        inputs=[mic_b, lang_b, target_lang_b],
        outputs=[live_caption_b, translation_b, audio_output_b]
    )
    
    # Conversation refresh
    def get_full_conversation():
        conv_a = stream_manager_A.get_conversation_history()
        conv_b = stream_manager_B.get_conversation_history()
        
        # Merge and sort by timestamp
        all_msgs = conv_a + conv_b
        all_msgs.sort(key=lambda x: x["timestamp"])
        
        output = []
        for msg in all_msgs:
            speaker = "👤 Person A" if msg["speaker"] == "A" else "👤 Person B"
            output.append(f"{speaker} ({msg['timestamp']})")
            output.append(f"  Original: {msg['original']}")
            output.append(f"  Translation: {msg['translation']}")
            output.append("")
        
        return "\n".join(output)
    
    refresh_btn.click(
        get_full_conversation,
        outputs=[conversation_view]
    )

if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_port=7860
    )
