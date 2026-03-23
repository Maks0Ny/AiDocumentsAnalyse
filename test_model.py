from neural_risk_inference import NeuralRiskModel

model = NeuralRiskModel("trained_model")

tests = [
    "Услуги оказываются в разумный срок.",
    "Стороны несут ответственность в установленном порядке.",
    "Оплата производится по согласованию сторон.",
    "Споры разрешаются в судебном порядке.",
    "Оплата производится в течение 5 банковских дней после подписания акта."
]

for text in tests:
    pred = model.predict_one(text)
    print("-" * 80)
    print("Текст:", text)
    print("Метка:", pred.predicted_label)
    print("Уверенность:", round(pred.confidence, 4))