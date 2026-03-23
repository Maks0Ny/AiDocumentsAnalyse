import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import spacy
except ImportError:
    spacy = None

try:
    import pymorphy2
except Exception:
    pymorphy2 = None


@dataclass
class AnalyzedFragment:
    """Отдельный фрагмент текста, выделенный для анализа."""
    text: str
    fragment_type: str
    matched_keywords: List[str] = field(default_factory=list)
    risk_markers: List[str] = field(default_factory=list)
    comment: str = ""


@dataclass
class ContractAnalysisResult:
    """Итог анализа текста договора на 2 этапе."""
    source_text: str
    sentences: List[str]
    fragments: List[AnalyzedFragment]
    missing_blocks: List[str]
    suspicious_phrases_found: List[Dict[str, str]]
    summary: Dict[str, str]


class ContractTextAnalyzer:
    """
    Анализ договора:
    1. Разбиение на предложения
    2. Выделение юридически значимых фрагментов
    3. Проверка наличия ключевых блоков
    4. Поиск подозрительных / размытых формулировок
    """

    REQUIRED_BLOCKS = {
        "subject": {
            "name": "Предмет договора",
            "keywords": [
                "предмет договора",
                "предмет",
                "обязуется передать",
                "обязуется выполнить",
                "обязуется оказать",
                "товар",
                "услуги",
                "работы",
            ],
        },
        "term": {
            "name": "Срок исполнения / срок действия",
            "keywords": [
                "срок",
                "срок действия",
                "дата",
                "в течение",
                "до",
                "не позднее",
            ],
        },
        "price": {
            "name": "Цена / порядок оплаты",
            "keywords": [
                "цена",
                "стоимость",
                "оплата",
                "оплачивает",
                "порядок расчетов",
                "вознаграждение",
                "арендная плата",
            ],
        },
        "liability": {
            "name": "Ответственность сторон",
            "keywords": [
                "ответственность",
                "неустойка",
                "штраф",
                "пени",
                "убытки",
                "возмещает",
            ],
        },
        "disputes": {
            "name": "Порядок разрешения споров",
            "keywords": [
                "споры",
                "разрешение споров",
                "в судебном порядке",
                "арбитражный суд",
                "претензионный порядок",
            ],
        },
        "termination": {
            "name": "Изменение / расторжение договора",
            "keywords": [
                "расторжение",
                "изменение договора",
                "односторонний отказ",
                "досрочное прекращение",
            ],
        },
    }

    # Формулировки, которые часто создают риск правовой неопределённости
    SUSPICIOUS_PATTERNS = [
        {
            "pattern": r"\bв разумный срок\b",
            "label": "Неопределённый срок",
            "comment": "Формулировка не содержит конкретного срока исполнения.",
        },
        {
            "pattern": r"\bпо согласованию сторон\b",
            "label": "Размытый порядок согласования",
            "comment": "Следует уточнить механизм, сроки и форму согласования.",
        },
        {
            "pattern": r"\bв установленном порядке\b",
            "label": "Неопределённый порядок действий",
            "comment": "Не указано, какой именно порядок применяется.",
        },
        {
            "pattern": r"\bпри необходимости\b",
            "label": "Оценочная формулировка",
            "comment": "Следует уточнить критерии наступления необходимости.",
        },
        {
            "pattern": r"\bвозможно\b",
            "label": "Неоднозначность обязательства",
            "comment": "Фраза создаёт неопределённость в объёме обязательств.",
        },
        {
            "pattern": r"\bпо возможности\b",
            "label": "Необязательная формулировка",
            "comment": "Следует заменить на чёткое обязательство или условие.",
        },
        {
            "pattern": r"\bв кратчайшие сроки\b",
            "label": "Неконкретный срок",
            "comment": "Нет точного временного интервала исполнения.",
        },
        {
            "pattern": r"\bнадлежащим образом\b",
            "label": "Оценочная формулировка",
            "comment": "Желательно уточнить критерии надлежащего исполнения.",
        },
        {
            "pattern": r"\bи иные\b",
            "label": "Открытый перечень",
            "comment": "Следует конкретизировать перечень условий или документов.",
        },
        {
            "pattern": r"\bдругие условия\b",
            "label": "Неконкретное условие",
            "comment": "Фраза требует детализации.",
        },
    ]

    FRAGMENT_PATTERNS = {
        "subject": [
            r"предмет договора",
            r"обязуется передать",
            r"обязуется выполнить",
            r"обязуется оказать",
        ],
        "price": [
            r"цена договора",
            r"стоимость",
            r"оплата",
            r"порядок расчетов",
            r"вознаграждение",
        ],
        "term": [
            r"срок",
            r"в течение",
            r"до \d{1,2}\.\d{1,2}\.\d{2,4}",
            r"не позднее",
        ],
        "liability": [
            r"ответственность",
            r"неустойка",
            r"штраф",
            r"пени",
            r"убытки",
        ],
        "disputes": [
            r"споры",
            r"претензионный порядок",
            r"арбитражный суд",
            r"судебном порядке",
        ],
        "termination": [
            r"расторжение",
            r"односторонний отказ",
            r"изменение договора",
        ],
    }

    def __init__(self, use_spacy: bool = False):
        self.use_spacy = use_spacy and spacy is not None
        self.nlp = None
        self.morph = None

        if self.use_spacy:
            try:
                self.nlp = spacy.load("ru_core_news_sm")
            except Exception:
                self.nlp = None
                self.use_spacy = False

    # =========================
    # 1. Разбиение на предложения
    # =========================
    def split_sentences(self, text: str) -> List[str]:
        """
        Разбивает текст на предложения.
        Если установлен spaCy — использует его.
        Иначе — регулярные выражения.
        """
        if not text.strip():
            return []

        if self.use_spacy and self.nlp is not None:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return sentences

        # Базовое разбиение без NLP
        raw_sentences = re.split(r'(?<=[.!?;])\s+|\n+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    # =========================
    # 2. Поиск ключевых блоков
    # =========================
    def find_missing_blocks(self, normalized_text: str) -> List[str]:
        """
        Проверяет, какие обязательные смысловые блоки, вероятно, отсутствуют.
        """
        missing = []

        for block_key, block_info in self.REQUIRED_BLOCKS.items():
            keywords = block_info["keywords"]
            found = any(keyword in normalized_text for keyword in keywords)
            if not found:
                missing.append(block_info["name"])

        return missing

    # =========================
    # 3. Поиск подозрительных формулировок
    # =========================
    def find_suspicious_phrases(self, sentences: List[str]) -> List[Dict[str, str]]:
        """
        Ищет формулировки, создающие правовую неопределённость.
        """
        results = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern_info in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern_info["pattern"], sentence_lower):
                    results.append({
                        "sentence": sentence,
                        "label": pattern_info["label"],
                        "comment": pattern_info["comment"],
                    })

        return results

    # =========================
    # 4. Выделение юридических фрагментов
    # =========================
    def extract_legal_fragments(self, sentences: List[str]) -> List[AnalyzedFragment]:
        """
        Выделяет предложения / фрагменты, содержащие юридически значимые конструкции.
        """
        fragments: List[AnalyzedFragment] = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            detected_types = []
            matched_keywords = []
            risk_markers = []

            for fragment_type, patterns in self.FRAGMENT_PATTERNS.items():
                local_matches = []
                for pattern in patterns:
                    if re.search(pattern, sentence_lower):
                        local_matches.append(pattern)

                if local_matches:
                    detected_types.append(fragment_type)
                    matched_keywords.extend(local_matches)

            for suspicious in self.SUSPICIOUS_PATTERNS:
                if re.search(suspicious["pattern"], sentence_lower):
                    risk_markers.append(suspicious["label"])

            if detected_types or risk_markers:
                fragment_type = ", ".join(sorted(set(detected_types))) if detected_types else "general_risk"
                comment = self._build_fragment_comment(detected_types, risk_markers)

                fragments.append(
                    AnalyzedFragment(
                        text=sentence,
                        fragment_type=fragment_type,
                        matched_keywords=sorted(set(matched_keywords)),
                        risk_markers=sorted(set(risk_markers)),
                        comment=comment,
                    )
                )

        return fragments

    def _build_fragment_comment(self, detected_types: List[str], risk_markers: List[str]) -> str:
        """
        Формирует пояснение к найденному фрагменту.
        """
        comments = []

        if detected_types:
            comments.append(
                f"Фрагмент относится к юридическому блоку: {', '.join(sorted(set(detected_types)))}."
            )

        if risk_markers:
            comments.append(
                f"Обнаружены потенциально рискованные формулировки: {', '.join(sorted(set(risk_markers)))}."
            )

        if not comments:
            comments.append("Фрагмент выделен для дополнительного анализа.")

        return " ".join(comments)

    # =========================
    # 5. Морфологическая нормализация (необязательно)
    # =========================
    def lemmatize_text(self, text: str) -> str:
        """
        Приводит слова к нормальной форме.
        Используется только если установлен pymorphy2.
        """
        if self.morph is None:
            return text

        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        normalized_tokens = []

        for token in tokens:
            if re.match(r"\w+", token, flags=re.UNICODE):
                parsed = self.morph.parse(token)[0]
                normalized_tokens.append(parsed.normal_form)
            else:
                normalized_tokens.append(token)

        return " ".join(normalized_tokens)

    # =========================
    # 6. Общий запуск анализа
    # =========================
    def analyze(self, cleaned_text: str, normalized_text: Optional[str] = None) -> ContractAnalysisResult:
        """
        Полный анализ текста договора на 2 этапе.
        """
        if normalized_text is None:
            normalized_text = cleaned_text.lower().replace("ё", "е")

        sentences = self.split_sentences(cleaned_text)
        missing_blocks = self.find_missing_blocks(normalized_text)
        suspicious_phrases_found = self.find_suspicious_phrases(sentences)
        fragments = self.extract_legal_fragments(sentences)

        summary = {
            "total_sentences": str(len(sentences)),
            "fragments_selected": str(len(fragments)),
            "missing_blocks_count": str(len(missing_blocks)),
            "suspicious_phrases_count": str(len(suspicious_phrases_found)),
        }

        return ContractAnalysisResult(
            source_text=cleaned_text,
            sentences=sentences,
            fragments=fragments,
            missing_blocks=missing_blocks,
            suspicious_phrases_found=suspicious_phrases_found,
            summary=summary,
        )


# =========================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# =========================

def print_analysis_report(result: ContractAnalysisResult) -> None:
    """
    Красивый консольный отчёт по 2 этапу анализа.
    """
    print("=" * 90)
    print("ОТЧЁТ ПО АНАЛИЗУ ТЕКСТА ДОГОВОРА")
    print("=" * 90)

    print("\nСводка:")
    for key, value in result.summary.items():
        print(f"- {key}: {value}")

    print("\nОтсутствующие ключевые блоки:")
    if result.missing_blocks:
        for block in result.missing_blocks:
            print(f"- {block}")
    else:
        print("Все основные блоки предварительно обнаружены.")

    print("\nПодозрительные формулировки:")
    if result.suspicious_phrases_found:
        for idx, item in enumerate(result.suspicious_phrases_found, start=1):
            print(f"{idx}. [{item['label']}]")
            print(f"   Фрагмент: {item['sentence']}")
            print(f"   Комментарий: {item['comment']}")
    else:
        print("Подозрительные формулировки не найдены.")

    print("\nЮридически значимые фрагменты:")
    if result.fragments:
        for idx, fragment in enumerate(result.fragments[:20], start=1):
            print(f"{idx}. Тип: {fragment.fragment_type}")
            print(f"   Текст: {fragment.text}")
            if fragment.matched_keywords:
                print(f"   Совпадения: {', '.join(fragment.matched_keywords)}")
            if fragment.risk_markers:
                print(f"   Маркеры риска: {', '.join(fragment.risk_markers)}")
            print(f"   Комментарий: {fragment.comment}")
            print()
        if len(result.fragments) > 20:
            print(f"... и ещё {len(result.fragments) - 20} фрагментов")
    else:
        print("Юридически значимые фрагменты не выделены.")


# =========================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =========================

if __name__ == "__main__":
    sample_text = """
    ДОГОВОР ОКАЗАНИЯ УСЛУГ

    1. Предмет договора
    Исполнитель обязуется оказать услуги Заказчику, а Заказчик обязуется оплатить эти услуги.

    2. Стоимость услуг и порядок расчетов
    Оплата производится по согласованию сторон.
    Стоимость услуг составляет 50 000 рублей.

    3. Срок оказания услуг
    Услуги оказываются в разумный срок.

    4. Ответственность сторон
    Стороны несут ответственность в установленном порядке.

    5. Разрешение споров
    Споры разрешаются в судебном порядке.
    """

    analyzer = ContractTextAnalyzer(use_spacy=False)
    analysis_result = analyzer.analyze(
        cleaned_text=sample_text,
        normalized_text=sample_text.lower().replace("ё", "е")
    )

    print_analysis_report(analysis_result)