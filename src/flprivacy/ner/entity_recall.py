"""Clinical named-entity recall via MedGemma.

Extracts clinical entities (diagnoses, procedures, anatomy, medications) from
reference and reconstructed text with a fixed prompt, then computes per-tokenizer
recall = |recovered ∩ reference| / |reference|. A pre-specified mechanistic
variant restricts the reference set to entities the GPT-2 tokenizer fragments into
>= 2 subwords (where a single-token domain-tokenizer advantage should be largest).
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass

_PROMPT = (
    "You are a clinical NER system. From the clinical text below, extract every "
    "clinical entity: diagnoses, procedures, anatomical structures, and "
    "medications. Return ONLY a comma-separated list of the exact surface forms, "
    "no explanations.\n\nTEXT:\n{}"
)
_STOP = ("here", "the follow", "clinical entit", "entit", "text", "none", "n/a",
         "sure", "okay")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).lower().strip())


def parse_entities(model_output: str) -> set[str]:
    """Parse a model completion into a normalised set of entity surface forms."""
    out: set[str] = set()
    for part in re.split(r"[,\n;]+", model_output):
        part = _norm(re.sub(r"^[\-\*\d\.\)\s]+", "", part))
        if 2 <= len(part) <= 40 and not part.startswith(_STOP):
            out.add(part)
    return out


@dataclass
class RecallResult:
    reference_entities: int
    recovered: int
    recall_pct: float


class MedGemmaEntityRecall:
    """Batched MedGemma entity extraction + recall scoring."""

    def __init__(self, model_name: str = "google/medgemma-4b-it",
                 device: str = "cuda", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._proc = None
        self._model = None

    def _load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._proc = AutoProcessor.from_pretrained(self.model_name)
        self._proc.tokenizer.padding_side = "left"
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, dtype=torch.bfloat16).to(self.device).eval()

    def extract(self, texts: list[str]) -> list[set[str]]:
        """Extract entity sets for a list of texts (batched generation)."""
        import torch
        if self._model is None:
            self._load()
        results: list[set[str]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            prompts = [self._proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "text",
                  "text": _PROMPT.format(t[:1000])}]}],
                add_generation_prompt=True, tokenize=False) for t in chunk]
            enc = self._proc.tokenizer(prompts, return_tensors="pt",
                                       padding=True).to(self.device)
            with torch.inference_mode():
                out = self._model.generate(**enc, max_new_tokens=96, do_sample=False)
            gen = self._proc.tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            results.extend(parse_entities(g) for g in gen)
        return results

    @staticmethod
    def recall(reference: set[str], reconstructed: set[str]) -> RecallResult:
        rec = reference & reconstructed
        return RecallResult(len(reference), len(rec),
                            round(100 * len(rec) / max(1, len(reference)), 1))
