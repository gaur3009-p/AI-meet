from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Translator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        out = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_length=200
        )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)
