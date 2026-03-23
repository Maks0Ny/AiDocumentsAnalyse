from dataclasses import dataclass, field
from typing import List

from neural_risk_inference import NeuralRiskModel
from classifier import RiskItem


@dataclass
class HybridRiskResult:
    rule_based_risks: List[RiskItem] = field(default_factory=list)
    neural_risks: List[RiskItem] = field(default_factory=list)
    final_risks: List[RiskItem] = field(default_factory=list)


class HybridRiskAnalyzer:
    def __init__(self, neural_model: NeuralRiskModel, confidence_threshold: float = 0.65):
        self.neural_model = neural_model
        self.confidence_threshold = confidence_threshold

    def analyze_fragments(self, fragments) -> List[RiskItem]:
        neural_risks = []

        for fragment in fragments:
            prediction = self.neural_model.predict_one(fragment.text)

            if prediction.predicted_label != "no_risk" and prediction.confidence >= self.confidence_threshold:
                neural_risks.append(
                    RiskItem(
                        category="Нейросетевой анализ",
                        risk_level=self._risk_level_from_confidence(prediction.confidence),
                        title=f"Нейросеть обнаружила риск: {prediction.predicted_label}",
                        description=(
                            f"Модель классифицировала фрагмент как '{prediction.predicted_label}' "
                            f"с уверенностью {prediction.confidence:.2f}."
                        ),
                        fragment_text=fragment.text,
                        recommendation=self._recommendation_from_label(prediction.predicted_label),
                    )
                )

        return neural_risks

    def merge_with_rule_based(self, rule_based_risks: List[RiskItem], neural_risks: List[RiskItem]) -> HybridRiskResult:
        final_risks = []
        seen = set()

        for risk in rule_based_risks + neural_risks:
            key = (
                risk.title.strip().lower(),
                risk.fragment_text.strip().lower(),
            )
            if key not in seen:
                seen.add(key)
                final_risks.append(risk)

        return HybridRiskResult(
            rule_based_risks=rule_based_risks,
            neural_risks=neural_risks,
            final_risks=final_risks,
        )

    def _risk_level_from_confidence(self, confidence: float) -> str:
        if confidence >= 0.90:
            return "Высокий"
        if confidence >= 0.75:
            return "Средний"
        return "Низкий"

    def _recommendation_from_label(self, label: str) -> str:
        mapping = {
            "risk_uncertain_time": "Уточнить срок исполнения обязательства и указать конкретную дату или период.",
            "risk_weak_liability": "Конкретизировать меры ответственности сторон: штраф, неустойка, убытки.",
            "risk_unclear_payment": "Уточнить порядок, сроки и форму оплаты.",
            "risk_dispute_missing_details": "Добавить претензионный порядок, срок ответа и подсудность.",
        }
        return mapping.get(label, "Проверить данный фрагмент и уточнить его юридическую формулировку.")