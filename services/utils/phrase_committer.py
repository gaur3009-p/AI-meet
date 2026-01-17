class PhraseCommitter:
    def __init__(self, min_words=8):
        self.last_committed_text = ""
        self.min_words = min_words

    def extract_phrase(self, full_text: str):
        if not full_text.startswith(self.last_committed_text):
            # Whisper re-write → reset safely
            self.last_committed_text = full_text
            return None

        delta = full_text[len(self.last_committed_text):].strip()
        words = delta.split()

        if len(words) >= self.min_words:
            phrase = " ".join(words)
            self.last_committed_text = full_text
            return phrase

        return None
