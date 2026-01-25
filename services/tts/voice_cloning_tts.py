# services/tts/voice_cloning_tts.py
import torch
import torchaudio
import soundfile as sf
import tempfile
import os
import numpy as np
from typing import Optional
from transformers import (
    VitsModel,
    VitsTokenizer,
    AutoProcessor,
    AutoModelForTextToWaveform
)

class VoiceCloningTTS:
    """
    Multi-strategy TTS with voice cloning capabilities.
    Falls back gracefully based on available models.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Initializing on {self.device}")
        
        # Strategy 1: Try XTTS (best quality, voice cloning)
        self.xtts_available = self._init_xtts()
        
        # Strategy 2: Fallback to YourTTS (good quality, multi-lingual)
        if not self.xtts_available:
            self.yourtts_available = self._init_yourtts()
        
        # Strategy 3: Fallback to VITS (fast, multi-lingual)
        if not self.xtts_available and not hasattr(self, 'yourtts_available'):
            self._init_vits()
    
    def _init_xtts(self) -> bool:
        """Initialize Coqui XTTS v2 for high-quality voice cloning."""
        try:
            from TTS.api import TTS
            self.xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print("[TTS] ✓ XTTS v2 loaded (best quality)")
            return True
        except Exception as e:
            print(f"[TTS] XTTS not available: {e}")
            return False
    
    def _init_yourtts(self) -> bool:
        """Initialize YourTTS for multi-lingual synthesis."""
        try:
            from TTS.api import TTS
            self.yourtts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(self.device)
            print("[TTS] ✓ YourTTS loaded")
            return True
        except Exception as e:
            print(f"[TTS] YourTTS not available: {e}")
            return False
    
    def _init_vits(self):
        """Initialize VITS as fallback."""
        try:
            # Multi-lingual VITS model
            self.vits_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
            self.vits_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(self.device)
            print("[TTS] ✓ VITS loaded (fallback)")
        except Exception as e:
            print(f"[TTS] Warning: All TTS models failed: {e}")
            self.vits_model = None
    
    def speak_with_cloning(
        self,
        text: str,
        reference_audio: str,
        language: str = "en"
    ) -> Optional[str]:
        """
        Generate speech with voice cloning.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio (5-10 sec recommended)
            language: Target language code
        
        Returns:
            Path to generated audio file
        """
        if self.xtts_available:
            return self._xtts_clone(text, reference_audio, language)
        elif hasattr(self, 'yourtts_available') and self.yourtts_available:
            return self._yourtts_clone(text, reference_audio, language)
        else:
            print("[TTS] Voice cloning not available, using default voice")
            return self.speak(text, language)
    
    def _xtts_clone(self, text: str, ref_audio: str, language: str) -> str:
        """XTTS voice cloning implementation."""
        try:
            output_path = tempfile.mktemp(suffix=".wav")
            
            # XTTS supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu
            lang_map = {
                "en": "en", "es": "es", "fr": "fr", "de": "de",
                "zh": "zh-cn", "ja": "ja", "ko": "ko", "hi": "en"  # Hindi uses English model
            }
            
            xtts_lang = lang_map.get(language, "en")
            
            self.xtts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=ref_audio,
                language=xtts_lang
            )
            
            return output_path
        except Exception as e:
            print(f"[TTS] XTTS cloning failed: {e}")
            return self.speak(text, language)
    
    def _yourtts_clone(self, text: str, ref_audio: str, language: str) -> str:
        """YourTTS voice cloning implementation."""
        try:
            output_path = tempfile.mktemp(suffix=".wav")
            
            self.yourtts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=ref_audio,
                language=language
            )
            
            return output_path
        except Exception as e:
            print(f"[TTS] YourTTS cloning failed: {e}")
            return self.speak(text, language)
    
    def speak(self, text: str, language: str = "en") -> Optional[str]:
        """
        Generate speech without voice cloning (default voice).
        
        Args:
            text: Text to synthesize
            language: Target language code
        
        Returns:
            Path to generated audio file
        """
        if not text.strip():
            return None
        
        try:
            # Use XTTS with default speaker
            if self.xtts_available:
                output_path = tempfile.mktemp(suffix=".wav")
                
                lang_map = {
                    "en": "en", "es": "es", "fr": "fr", "de": "de",
                    "zh": "zh-cn", "ja": "ja", "ko": "ko", "hi": "en"
                }
                xtts_lang = lang_map.get(language, "en")
                
                self.xtts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=xtts_lang
                )
                return output_path
            
            # Use YourTTS
            elif hasattr(self, 'yourtts_available') and self.yourtts_available:
                output_path = tempfile.mktemp(suffix=".wav")
                self.yourtts.tts_to_file(text=text, file_path=output_path)
                return output_path
            
            # Use VITS
            elif self.vits_model is not None:
                inputs = self.vits_tokenizer(text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.vits_model(**inputs)
                    waveform = outputs.waveform[0]
                
                output_path = tempfile.mktemp(suffix=".wav")
                sf.write(output_path, waveform.cpu().numpy(), 16000)
                return output_path
            
            else:
                print("[TTS] No TTS model available")
                return None
                
        except Exception as e:
            print(f"[TTS] Synthesis failed: {e}")
            return None
    
    def preprocess_reference_audio(self, audio_path: str, target_length: int = 6) -> str:
        """
        Preprocess reference audio for optimal voice cloning.
        
        Args:
            audio_path: Input audio path
            target_length: Target duration in seconds
        
        Returns:
            Path to preprocessed audio
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 22050 Hz (optimal for XTTS)
            if sr != 22050:
                resampler = torchaudio.transforms.Resample(sr, 22050)
                waveform = resampler(waveform)
                sr = 22050
            
            # Trim or pad to target length
            target_samples = target_length * sr
            current_samples = waveform.shape[1]
            
            if current_samples > target_samples:
                # Trim from center
                start = (current_samples - target_samples) // 2
                waveform = waveform[:, start:start + target_samples]
            elif current_samples < target_samples:
                # Pad with silence
                padding = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Normalize
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Save preprocessed audio
            output_path = tempfile.mktemp(suffix=".wav")
            torchaudio.save(output_path, waveform, sr)
            
            return output_path
            
        except Exception as e:
            print(f"[TTS] Audio preprocessing failed: {e}")
            return audio_path  # Return original on failure
