# tests/test_pipeline.py
import unittest
import numpy as np
import soundfile as sf
import tempfile
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from services.asr.whisper_streaming import LiveWhisperASR
from services.translation.nllb_translate import Translator
from services.tts.voice_cloning_tts import VoiceCloningTTS
from services.utils.stream_manager import StreamManager

class TestASR(unittest.TestCase):
    """Test Automatic Speech Recognition."""
    
    @classmethod
    def setUpClass(cls):
        cls.asr = LiveWhisperASR()
    
    def test_english_transcription(self):
        """Test English speech recognition."""
        # Generate synthetic audio
        audio = self._generate_test_audio(duration=2.0)
        audio_path = self._save_temp_audio(audio)
        
        result = self.asr.transcribe(audio_path, "en")
        
        self.assertIsInstance(result, str)
        print(f"✓ ASR English: {result}")
    
    def test_hindi_transcription(self):
        """Test Hindi speech recognition."""
        audio = self._generate_test_audio(duration=2.0)
        audio_path = self._save_temp_audio(audio)
        
        result = self.asr.transcribe(audio_path, "hi")
        
        self.assertIsInstance(result, str)
        print(f"✓ ASR Hindi: {result}")
    
    @staticmethod
    def _generate_test_audio(duration=2.0, sr=16000):
        """Generate synthetic audio signal."""
        t = np.linspace(0, duration, int(duration * sr))
        # Mix of frequencies to simulate speech
        signal = (
            np.sin(2 * np.pi * 200 * t) * 0.3 +
            np.sin(2 * np.pi * 400 * t) * 0.2 +
            np.random.randn(len(t)) * 0.05
        )
        return signal
    
    @staticmethod
    def _save_temp_audio(audio, sr=16000):
        """Save audio to temporary file."""
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, sr)
        return temp_path

class TestTranslation(unittest.TestCase):
    """Test translation service."""
    
    @classmethod
    def setUpClass(cls):
        cls.translator = Translator()
    
    def test_hindi_to_english(self):
        """Test Hindi to English translation."""
        hindi_text = "नमस्ते, आप कैसे हैं?"
        
        result = self.translator.translate(
            hindi_text,
            "hin_Deva",
            "eng_Latn"
        )
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        print(f"✓ Translation HI→EN: {hindi_text} → {result}")
    
    def test_english_to_hindi(self):
        """Test English to Hindi translation."""
        english_text = "Hello, how are you?"
        
        result = self.translator.translate(
            english_text,
            "eng_Latn",
            "hin_Deva"
        )
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        print(f"✓ Translation EN→HI: {english_text} → {result}")
    
    def test_multi_sentence(self):
        """Test translation of longer text."""
        text = "This is a test. We are testing the translation system. It should work well."
        
        result = self.translator.translate(
            text,
            "eng_Latn",
            "spa_Latn"
        )
        
        self.assertIsInstance(result, str)
        print(f"✓ Multi-sentence EN→ES: {result}")

class TestTTS(unittest.TestCase):
    """Test Text-to-Speech with voice cloning."""
    
    @classmethod
    def setUpClass(cls):
        cls.tts = VoiceCloningTTS()
    
    def test_basic_synthesis(self):
        """Test basic TTS without voice cloning."""
        text = "This is a test of the text to speech system."
        
        result = self.tts.speak(text, "en")
        
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        print(f"✓ TTS Basic: Generated {result}")
    
    def test_voice_cloning(self):
        """Test voice cloning functionality."""
        # Generate reference audio
        ref_audio = TestASR._generate_test_audio(duration=5.0)
        ref_path = TestASR._save_temp_audio(ref_audio)
        
        text = "This is a test with voice cloning."
        
        result = self.tts.speak_with_cloning(text, ref_path, "en")
        
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        print(f"✓ TTS Voice Clone: Generated {result}")
    
    def test_multilingual(self):
        """Test TTS in multiple languages."""
        tests = [
            ("Hello world", "en"),
            ("Hola mundo", "es"),
            ("Bonjour le monde", "fr"),
        ]
        
        for text, lang in tests:
            result = self.tts.speak(text, lang)
            self.assertIsNotNone(result)
            print(f"✓ TTS {lang.upper()}: Generated {result}")

class TestStreamManager(unittest.TestCase):
    """Test stream management and buffering."""
    
    def setUp(self):
        self.manager = StreamManager("A")
    
    def test_audio_buffering(self):
        """Test audio chunk buffering."""
        # Add speech chunks
        for _ in range(10):
            chunk = TestASR._generate_test_audio(duration=0.1)
            result = self.manager.add_audio_chunk(chunk, 16000)
        
        self.assertIsNone(result)  # No utterance yet
        print("✓ Stream buffering works")
    
    def test_utterance_detection(self):
        """Test silence-based utterance detection."""
        # Add speech
        for _ in range(20):
            chunk = TestASR._generate_test_audio(duration=0.1)
            self.manager.add_audio_chunk(chunk, 16000)
        
        # Add silence
        import time
        time.sleep(1.0)
        silence = np.zeros(1600)
        result = self.manager.add_audio_chunk(silence, 16000)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "utterance_complete")
        print(f"✓ Utterance detection: {result['duration']:.2f}s")
    
    def test_translation_history(self):
        """Test translation tracking."""
        self.manager.add_translation(
            "Hello",
            "Hola",
            "/tmp/test.wav"
        )
        
        history = self.manager.get_conversation_history()
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["original"], "Hello")
        self.assertEqual(history[0]["translation"], "Hola")
        print("✓ Translation history tracking works")

class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    @classmethod
    def setUpClass(cls):
        cls.asr = LiveWhisperASR()
        cls.translator = Translator()
        cls.tts = VoiceCloningTTS()
        cls.manager = StreamManager("test")
    
    def test_full_pipeline(self):
        """Test complete translation pipeline."""
        # 1. Generate test audio
        audio = TestASR._generate_test_audio(duration=3.0)
        audio_path = TestASR._save_temp_audio(audio)
        
        # 2. Transcribe
        transcript = self.asr.transcribe(audio_path, "en")
        self.assertIsInstance(transcript, str)
        print(f"✓ Transcribed: {transcript}")
        
        # 3. Translate
        if transcript.strip():
            translation = self.translator.translate(
                transcript,
                "eng_Latn",
                "hin_Deva"
            )
            self.assertIsInstance(translation, str)
            print(f"✓ Translated: {translation}")
            
            # 4. Synthesize
            tts_output = self.tts.speak(translation, "hi")
            self.assertIsNotNone(tts_output)
            print(f"✓ Synthesized: {tts_output}")
            
            # 5. Track in manager
            self.manager.add_translation(transcript, translation, tts_output)
            history = self.manager.get_conversation_history()
            self.assertEqual(len(history), 1)
            print("✓ Full pipeline successful")

class TestPerformance(unittest.TestCase):
    """Performance and benchmarking tests."""
    
    def test_asr_latency(self):
        """Measure ASR processing time."""
        import time
        
        asr = LiveWhisperASR()
        audio = TestASR._generate_test_audio(duration=5.0)
        audio_path = TestASR._save_temp_audio(audio)
        
        start = time.time()
        result = asr.transcribe(audio_path, "en")
        latency = time.time() - start
        
        print(f"✓ ASR Latency: {latency:.2f}s for 5s audio")
        self.assertLess(latency, 10.0)  # Should process in under 10s
    
    def test_translation_latency(self):
        """Measure translation processing time."""
        import time
        
        translator = Translator()
        text = "This is a longer sentence to test translation performance. " * 5
        
        start = time.time()
        result = translator.translate(text, "eng_Latn", "hin_Deva")
        latency = time.time() - start
        
        print(f"✓ Translation Latency: {latency:.2f}s")
        self.assertLess(latency, 5.0)
    
    def test_memory_usage(self):
        """Check memory consumption."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load models
        asr = LiveWhisperASR()
        translator = Translator()
        tts = VoiceCloningTTS()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        print(f"✓ Memory Usage: {memory_used:.0f} MB")
        self.assertLess(memory_used, 16000)  # Should use less than 16GB

def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("🧪 Running Voice Translation System Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestASR))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslation))
    suite.addTests(loader.loadTestsFromTestCase(TestTTS))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamManager))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
