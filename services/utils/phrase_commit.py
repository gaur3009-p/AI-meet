class PhraseCommitter:
    def __init__(self, min_words=6):
        self.last_committed = ""

    def process(self, live_text: str):
        if not live_text.startswith(self.last_committed):
            # Whisper rewrote earlier words → reset safely
            self.last_committed = live_text
            return None

        delta = live_text[len(self.last_committed):].strip()
        words = delta.split()

        if len(words) >= 6:
            phrase = " ".join(words)
            self.last_committed = live_text
            return phrase

        return None
