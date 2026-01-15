from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Translator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # Set source language
        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Convert target language token to ID (NEW API)
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=256
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
