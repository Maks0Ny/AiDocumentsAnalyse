from dataclasses import dataclass, field
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class NeuralPrediction:
    text: str
    predicted_label: str
    confidence: float


class NeuralRiskModel:
    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_one(self, text: str) -> NeuralPrediction:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())
        predicted_label = self.model.config.id2label[pred_id]

        return NeuralPrediction(
            text=text,
            predicted_label=predicted_label,
            confidence=confidence,
        )

    def predict_many(self, texts: List[str]) -> List[NeuralPrediction]:
        return [self.predict_one(text) for text in texts]