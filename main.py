from preprocessor import ContractPreprocessor
from analyzer import ContractTextAnalyzer
from classifier import ContractRiskClassifier
from neural_risk_inference import NeuralRiskModel
from hybrid_classifier import HybridRiskAnalyzer
from report_generator import ContractReportGenerator, build_report_metadata

# 1. Предобработка
processor = ContractPreprocessor()
preprocess_result = processor.preprocess("")

# 2. Анализ текста
analyzer = ContractTextAnalyzer(use_spacy=False)
analysis_result = analyzer.analyze(
    cleaned_text=preprocess_result.cleaned_text,
    normalized_text=preprocess_result.normalized_text,
)

# 3. Правиловая классификация
rule_classifier = ContractRiskClassifier()
rule_based_result = rule_classifier.classify(analysis_result)

# 4. Нейросетевой анализ
neural_model = NeuralRiskModel(model_dir="trained_model")
hybrid_analyzer = HybridRiskAnalyzer(
    neural_model=neural_model,
    confidence_threshold=0.90
)

neural_risks = hybrid_analyzer.analyze_fragments(analysis_result.fragments)
hybrid_result = hybrid_analyzer.merge_with_rule_based(
    rule_based_risks=rule_based_result.risks,
    neural_risks=neural_risks,
)

# 5. Формирование отчёта
report_generator = ContractReportGenerator()
metadata = build_report_metadata(
    contract_name="Договор оказания услуг",
    source_path="dogovor.docx",
)

summary = {
    "total_risks": len(hybrid_result.final_risks),
    "high_risk": sum(1 for r in hybrid_result.final_risks if r.risk_level.lower() == "высокий"),
    "medium_risk": sum(1 for r in hybrid_result.final_risks if r.risk_level.lower() == "средний"),
    "low_risk": sum(1 for r in hybrid_result.final_risks if r.risk_level.lower() == "низкий"),
    "categories_count": len(set(r.category for r in hybrid_result.final_risks)),
}

report_generator.generate_docx_report(
    output_path="final_contract_report.docx",
    metadata=metadata,
    final_risks=hybrid_result.final_risks,
    summary=summary,
)

report_generator.generate_txt_report(
    output_path="final_contract_report.txt",
    metadata=metadata,
    final_risks=hybrid_result.final_risks,
    summary=summary,
)

print("Отчёт успешно сформирован.")