# services/utils/stream_manager.py
import numpy as np
import time
import soundfile as sf
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

@dataclass
class Speaker:
    """Represents a speaker in the conversation."""
    id: str
    language: str
    voice_reference: Optional[str] = None

@dataclass
class TranslationSegment:
    """A single translation segment."""
    timestamp: str
    speaker_id: str
    original_text: str
    translated_text: str
    audio_path: Optional[str] = None

class StreamManager:
    """
    Manages audio streaming, buffering, and state for one speaker.
    Implements intelligent utterance detection and phrase committing.
    """
    
    def __init__(
        self,
        speaker_id: str,
        silence_threshold: float = 0.02,
        silence_duration: float = 0.8,
        min_utterance_length: float = 0.5,
        max_buffer_duration: float = 30.0
    ):
        self.speaker_id = speaker_id
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_utterance_length = min_utterance_length
        self.max_buffer_duration = max_buffer_duration
        
        # Audio buffering
        self.audio_chunks: List[np.ndarray] = []
        self.sample_rate: int = 16000
        self.last_voice_time: float = time.time()
        self.buffer_start_time: float = time.time()
        
        # Translation history
        self.translations: List[TranslationSegment] = []
        
        # Live caption state
        self.current_caption: str = ""
        self.last_committed_text: str = ""
        
        # Statistics
        self.total_utterances: int = 0
        self.total_audio_duration: float = 0.0
    
    def add_audio_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int
    ) -> Optional[Dict]:
        """
        Add audio chunk and detect utterance completion.
        
        Returns:
            Dict with utterance info if complete, None otherwise
        """
        self.sample_rate = sample_rate
        
        # Calculate energy
        energy = np.sqrt(np.mean(chunk ** 2))
        
        current_time = time.time()
        is_speech = energy > self.silence_threshold
        
        if is_speech:
            self.last_voice_time = current_time
            self.audio_chunks.append(chunk)
            
            # Update live caption placeholder
            self.current_caption = "[Speaking...]"
            
            # Prevent infinite buffer growth
            buffer_duration = len(self.audio_chunks) * len(chunk) / sample_rate
            if buffer_duration > self.max_buffer_duration:
                # Force commit if buffer too long
                return self._commit_utterance()
        
        else:
            # Check for silence-based utterance end
            silence_time = current_time - self.last_voice_time
            
            if silence_time >= self.silence_duration and self.audio_chunks:
                # Calculate utterance duration
                total_samples = sum(len(c) for c in self.audio_chunks)
                duration = total_samples / sample_rate
                
                if duration >= self.min_utterance_length:
                    return self._commit_utterance()
                else:
                    # Too short, discard
                    self.audio_chunks = []
                    self.current_caption = ""
        
        return None
    
    def _commit_utterance(self) -> Dict:
        """Save current buffer as complete utterance."""
        if not self.audio_chunks:
            return None
        
        # Concatenate all chunks
        full_audio = np.concatenate(self.audio_chunks)
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, full_audio, self.sample_rate)
        
        # Update statistics
        duration = len(full_audio) / self.sample_rate
        self.total_audio_duration += duration
        self.total_utterances += 1
        
        # Reset buffer
        self.audio_chunks = []
        self.current_caption = ""
        self.buffer_start_time = time.time()
        
        return {
            "type": "utterance_complete",
            "audio_path": temp_path,
            "duration": duration,
            "utterance_id": self.total_utterances
        }
    
    def add_translation(
        self,
        original_text: str,
        translated_text: str,
        audio_path: Optional[str] = None
    ):
        """Add a completed translation to history."""
        segment = TranslationSegment(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            speaker_id=self.speaker_id,
            original_text=original_text,
            translated_text=translated_text,
            audio_path=audio_path
        )
        
        self.translations.append(segment)
        self.last_committed_text = original_text
    
    def get_display_state(self) -> tuple:
        """
        Get current state for UI display.
        
        Returns:
            (live_caption, translation_history)
        """
        # Live caption
        caption = self.current_caption if self.audio_chunks else self.last_committed_text
        
        # Translation history
        history_lines = []
        for seg in self.translations[-10:]:  # Last 10 segments
            history_lines.append(f"[{seg.timestamp}] {seg.original_text}")
            history_lines.append(f"  → {seg.translated_text}")
            history_lines.append("")
        
        history = "\n".join(history_lines) if history_lines else "No translations yet"
        
        return (caption, history)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get all translations as structured data."""
        return [
            {
                "timestamp": seg.timestamp,
                "speaker": seg.speaker_id,
                "original": seg.original_text,
                "translation": seg.translated_text
            }
            for seg in self.translations
        ]
    
    def reset(self):
        """Clear all state."""
        self.audio_chunks = []
        self.translations = []
        self.current_caption = ""
        self.last_committed_text = ""
        self.total_utterances = 0
        self.total_audio_duration = 0.0
    
    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        return {
            "speaker_id": self.speaker_id,
            "total_utterances": self.total_utterances,
            "total_duration_seconds": round(self.total_audio_duration, 2),
            "total_translations": len(self.translations),
            "buffer_status": "active" if self.audio_chunks else "idle"
        }
