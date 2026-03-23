from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =========================
# МОДЕЛИ ДАННЫХ
# =========================

@dataclass
class RiskItem:
    """Одна выявленная проблема договора."""
    category: str
    risk_level: str
    title: str
    description: str
    fragment_text: str
    recommendation: str


@dataclass
class RiskClassificationResult:
    """Итог классификации рисков."""
    risks: List[RiskItem] = field(default_factory=list)
    grouped_risks: Dict[str, List[RiskItem]] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)


# =========================
# КЛАССИФИКАТОР РИСКОВ
# =========================

class ContractRiskClassifier:
    """
    Классифицирует обнаруженные проблемы договора:
    1. отсутствие существенных условий
    2. неопределённые формулировки
    3. отсутствие ответственности сторон
    4. отсутствие порядка разрешения споров
    5. отсутствие условий изменения/расторжения
    6. структурные и технические недостатки
    """

    MISSING_BLOCK_CATEGORY_MAP = {
        "Предмет договора": {
            "category": "Отсутствие существенных условий",
            "risk_level": "Высокий",
            "title": "Не найден предмет договора",
            "description": (
                "В тексте договора не обнаружен явный блок, описывающий предмет договора. "
                "Это может привести к признанию договора незаключённым."
            ),
            "recommendation": (
                "Добавить раздел, в котором чётко определить предмет договора: "
                "что именно передаётся, выполняется или оказывается."
            ),
        },
        "Срок исполнения / срок действия": {
            "category": "Неполнота условий договора",
            "risk_level": "Средний",
            "title": "Не найден срок исполнения или срок действия",
            "description": (
                "В договоре не выявлены точные условия, регулирующие срок исполнения обязательств "
                "или срок действия договора."
            ),
            "recommendation": (
                "Указать конкретные даты, периоды или сроки исполнения обязательств."
            ),
        },
        "Цена / порядок оплаты": {
            "category": "Неполнота условий договора",
            "risk_level": "Высокий",
            "title": "Не найдены условия об оплате",
            "description": (
                "В тексте не обнаружен блок, описывающий цену договора, стоимость услуг "
                "или порядок расчётов."
            ),
            "recommendation": (
                "Добавить условия о размере оплаты, сроках и порядке расчётов."
            ),
        },
        "Ответственность сторон": {
            "category": "Недостаточность условий ответственности",
            "risk_level": "Средний",
            "title": "Не найден раздел об ответственности сторон",
            "description": (
                "В договоре отсутствует явно выраженный блок, регулирующий ответственность сторон "
                "за нарушение обязательств."
            ),
            "recommendation": (
                "Добавить положения о штрафах, неустойке, возмещении убытков и иных мерах ответственности."
            ),
        },
        "Порядок разрешения споров": {
            "category": "Отсутствие порядка разрешения споров",
            "risk_level": "Средний",
            "title": "Не найден порядок разрешения споров",
            "description": (
                "В договоре отсутствует раздел, определяющий претензионный порядок "
                "или судебную подсудность споров."
            ),
            "recommendation": (
                "Добавить условия о досудебном урегулировании, подсудности и порядке рассмотрения споров."
            ),
        },
        "Изменение / расторжение договора": {
            "category": "Отсутствие условий изменения и расторжения",
            "risk_level": "Низкий",
            "title": "Не найдены условия изменения или расторжения договора",
            "description": (
                "В тексте отсутствуют чёткие правила изменения и досрочного прекращения договора."
            ),
            "recommendation": (
                "Добавить раздел о порядке изменения, расторжения и одностороннего отказа от договора."
            ),
        },
    }

    SUSPICIOUS_PHRASE_CATEGORY_MAP = {
        "Неопределённый срок": {
            "category": "Неопределённость формулировок",
            "risk_level": "Средний",
            "title": "Обнаружен неопределённый срок",
            "recommendation": "Заменить оценочное выражение на точный срок или конкретный временной период.",
        },
        "Размытый порядок согласования": {
            "category": "Неопределённость формулировок",
            "risk_level": "Средний",
            "title": "Обнаружен размытый порядок согласования",
            "recommendation": "Уточнить, кто, в какие сроки и в какой форме осуществляет согласование.",
        },
        "Неопределённый порядок действий": {
            "category": "Неопределённость формулировок",
            "risk_level": "Средний",
            "title": "Обнаружен неопределённый порядок действий",
            "recommendation": "Указать конкретный механизм действий, ссылки на документ или процедуру.",
        },
        "Оценочная формулировка": {
            "category": "Неопределённость формулировок",
            "risk_level": "Низкий",
            "title": "Обнаружена оценочная формулировка",
            "recommendation": "Конкретизировать критерии применения данного условия.",
        },
        "Неоднозначность обязательства": {
            "category": "Неопределённость формулировок",
            "risk_level": "Средний",
            "title": "Обнаружена неоднозначность обязательства",
            "recommendation": "Сформулировать обязательство в императивной и однозначной форме.",
        },
        "Необязательная формулировка": {
            "category": "Неопределённость формулировок",
            "risk_level": "Средний",
            "title": "Обнаружена необязательная формулировка",
            "recommendation": "Указать чёткую обязанность, условие или право стороны.",
        },
        "Неконкретный срок": {
            "category": "Неопределённость формулировок",
            "risk_level": "Средний",
            "title": "Обнаружен неконкретный срок",
            "recommendation": "Установить точный срок исполнения обязательства.",
        },
        "Открытый перечень": {
            "category": "Структурная неопределённость",
            "risk_level": "Низкий",
            "title": "Обнаружен открытый перечень условий",
            "recommendation": "По возможности сделать перечень закрытым или определить его границы.",
        },
        "Неконкретное условие": {
            "category": "Неопределённость формулировок",
            "risk_level": "Низкий",
            "title": "Обнаружено неконкретное условие",
            "recommendation": "Уточнить содержание соответствующего условия.",
        },
    }

    def classify(self, analysis_result) -> RiskClassificationResult:
        """
        Принимает результат 2 этапа и возвращает структурированную классификацию рисков.
        """
        risks: List[RiskItem] = []

        # 1. Риски по отсутствующим блокам
        risks.extend(self._classify_missing_blocks(analysis_result.missing_blocks))

        # 2. Риски по подозрительным формулировкам
        risks.extend(self._classify_suspicious_phrases(analysis_result.suspicious_phrases_found))

        # 3. Дополнительные риски по содержанию фрагментов
        risks.extend(self._classify_fragments(analysis_result.fragments))

        # 4. Удаляем дубли
        risks = self._deduplicate_risks(risks)

        # 5. Группировка и сводка
        grouped_risks = self._group_risks(risks)
        summary = self._build_summary(risks, grouped_risks)

        return RiskClassificationResult(
            risks=risks,
            grouped_risks=grouped_risks,
            summary=summary,
        )

    def _classify_missing_blocks(self, missing_blocks: List[str]) -> List[RiskItem]:
        risks = []

        for block_name in missing_blocks:
            if block_name in self.MISSING_BLOCK_CATEGORY_MAP:
                data = self.MISSING_BLOCK_CATEGORY_MAP[block_name]
                risks.append(
                    RiskItem(
                        category=data["category"],
                        risk_level=data["risk_level"],
                        title=data["title"],
                        description=data["description"],
                        fragment_text=f"[Отсутствующий блок: {block_name}]",
                        recommendation=data["recommendation"],
                    )
                )

        return risks

    def _classify_suspicious_phrases(self, suspicious_phrases_found: List[Dict[str, str]]) -> List[RiskItem]:
        risks = []

        for item in suspicious_phrases_found:
            label = item["label"]
            sentence = item["sentence"]
            comment = item["comment"]

            if label in self.SUSPICIOUS_PHRASE_CATEGORY_MAP:
                data = self.SUSPICIOUS_PHRASE_CATEGORY_MAP[label]
                risks.append(
                    RiskItem(
                        category=data["category"],
                        risk_level=data["risk_level"],
                        title=data["title"],
                        description=comment,
                        fragment_text=sentence,
                        recommendation=data["recommendation"],
                    )
                )
            else:
                risks.append(
                    RiskItem(
                        category="Иные риски формулировок",
                        risk_level="Низкий",
                        title="Обнаружена потенциально рискованная формулировка",
                        description=comment,
                        fragment_text=sentence,
                        recommendation="Уточнить данную формулировку и сделать её более определённой.",
                    )
                )

        return risks

    def _classify_fragments(self, fragments: List) -> List[RiskItem]:
        """
        Дополнительная логика классификации по содержанию фрагментов.
        """
        risks = []

        for fragment in fragments:
            text_lower = fragment.text.lower()

            # 1. Блок ответственности есть, но слабый
            if "liability" in fragment.fragment_type:
                if any(phrase in text_lower for phrase in ["в установленном порядке", "в соответствии с законодательством"]):
                    risks.append(
                        RiskItem(
                            category="Недостаточность условий ответственности",
                            risk_level="Средний",
                            title="Ответственность сторон сформулирована слишком общо",
                            description=(
                                "Раздел об ответственности присутствует, однако условия ответственности "
                                "не конкретизированы и отсылают к общим нормам."
                            ),
                            fragment_text=fragment.text,
                            recommendation=(
                                "Указать конкретные последствия нарушения: неустойку, штраф, порядок возмещения убытков."
                            ),
                        )
                    )

            # 2. Оплата есть, но способ расчёта неясен
            if "price" in fragment.fragment_type:
                if "по согласованию сторон" in text_lower:
                    risks.append(
                        RiskItem(
                            category="Неполнота условий договора",
                            risk_level="Средний",
                            title="Порядок оплаты сформулирован неопределённо",
                            description=(
                                "Условие об оплате содержит ссылку на последующее согласование, "
                                "но не закрепляет конкретный порядок расчётов."
                            ),
                            fragment_text=fragment.text,
                            recommendation=(
                                "Указать точные сроки, форму оплаты, реквизиты и порядок подтверждения расчётов."
                            ),
                        )
                    )

            # 3. Срок есть, но он неопределённый
            if "term" in fragment.fragment_type:
                if any(phrase in text_lower for phrase in ["в разумный срок", "в кратчайшие сроки"]):
                    risks.append(
                        RiskItem(
                            category="Неопределённость формулировок",
                            risk_level="Средний",
                            title="Срок исполнения обязательства не конкретизирован",
                            description=(
                                "Вместо точного срока используется оценочная формулировка, "
                                "что может вызывать споры при исполнении договора."
                            ),
                            fragment_text=fragment.text,
                            recommendation=(
                                "Установить конкретный срок исполнения обязательств в днях, месяцах или датах."
                            ),
                        )
                    )

            # 4. Споры упомянуты, но без детализации
            if "disputes" in fragment.fragment_type:
                if "в судебном порядке" in text_lower and "претензион" not in text_lower:
                    risks.append(
                        RiskItem(
                            category="Отсутствие детального порядка разрешения споров",
                            risk_level="Низкий",
                            title="Порядок разрешения споров указан слишком кратко",
                            description=(
                                "Договор упоминает судебное разрешение споров, "
                                "но не содержит условий о претензионном порядке или подсудности."
                            ),
                            fragment_text=fragment.text,
                            recommendation=(
                                "Добавить претензионный порядок, срок ответа на претензию и указание подсудности."
                            ),
                        )
                    )

            # 5. Общая структурная слабость формулировок
            if fragment.risk_markers:
                if "Открытый перечень" in fragment.risk_markers:
                    risks.append(
                        RiskItem(
                            category="Структурная неопределённость",
                            risk_level="Низкий",
                            title="Используется открытый перечень",
                            description=(
                                "Открытые перечни могут привести к произвольному толкованию "
                                "объёма обязательств или состава документов."
                            ),
                            fragment_text=fragment.text,
                            recommendation=(
                                "Ограничить перечень или добавить критерии включения иных элементов."
                            ),
                        )
                    )

        return risks

    def _deduplicate_risks(self, risks: List[RiskItem]) -> List[RiskItem]:
        unique = []
        seen = set()

        for risk in risks:
            key = (
                risk.category.strip().lower(),
                risk.risk_level.strip().lower(),
                risk.title.strip().lower(),
                risk.fragment_text.strip().lower(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(risk)

        return unique

    def _group_risks(self, risks: List[RiskItem]) -> Dict[str, List[RiskItem]]:
        grouped: Dict[str, List[RiskItem]] = {}

        for risk in risks:
            grouped.setdefault(risk.category, []).append(risk)

        return grouped

    def _build_summary(self, risks: List[RiskItem], grouped_risks: Dict[str, List[RiskItem]]) -> Dict[str, int]:
        summary = {
            "total_risks": len(risks),
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "categories_count": len(grouped_risks),
        }

        for risk in risks:
            level = risk.risk_level.lower()
            if level == "высокий":
                summary["high_risk"] += 1
            elif level == "средний":
                summary["medium_risk"] += 1
            elif level == "низкий":
                summary["low_risk"] += 1

        return summary


# =========================
# СЛУЖЕБНЫЙ ВЫВОД
# =========================

def print_risk_classification_report(result: RiskClassificationResult) -> None:
    print("=" * 100)
    print("ОТЧЁТ ПО КЛАССИФИКАЦИИ ЮРИДИЧЕСКИХ РИСКОВ")
    print("=" * 100)

    print("\nСводка:")
    for key, value in result.summary.items():
        print(f"- {key}: {value}")

    print("\nРиски по категориям:")
    if not result.grouped_risks:
        print("Риски не обнаружены.")
        return

    for category, risks in result.grouped_risks.items():
        print(f"\n[{category}]")
        for idx, risk in enumerate(risks, start=1):
            print(f"{idx}. {risk.title} ({risk.risk_level} риск)")
            print(f"   Описание: {risk.description}")
            print(f"   Фрагмент: {risk.fragment_text}")
            print(f"   Рекомендация: {risk.recommendation}")
            print()


# =========================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =========================

if __name__ == "__main__":
    # Здесь ожидается объект analysis_result из 2 этапа.
    # Ниже — простой пример-заглушка.
    from dataclasses import dataclass, field
    from typing import List, Dict

    @dataclass
    class DummyFragment:
        text: str
        fragment_type: str
        matched_keywords: List[str] = field(default_factory=list)
        risk_markers: List[str] = field(default_factory=list)
        comment: str = ""

    @dataclass
    class DummyAnalysisResult:
        missing_blocks: List[str]
        suspicious_phrases_found: List[Dict[str, str]]
        fragments: List[DummyFragment]

    analysis_result = DummyAnalysisResult(
        missing_blocks=[
            "Ответственность сторон",
            "Изменение / расторжение договора",
        ],
        suspicious_phrases_found=[
            {
                "sentence": "Услуги оказываются в разумный срок.",
                "label": "Неопределённый срок",
                "comment": "Формулировка не содержит конкретного срока исполнения.",
            },
            {
                "sentence": "Оплата производится по согласованию сторон.",
                "label": "Размытый порядок согласования",
                "comment": "Следует уточнить механизм, сроки и форму согласования.",
            },
        ],
        fragments=[
            DummyFragment(
                text="Стороны несут ответственность в установленном порядке.",
                fragment_type="liability",
                risk_markers=["Неопределённый порядок действий"],
            ),
            DummyFragment(
                text="Споры разрешаются в судебном порядке.",
                fragment_type="disputes",
            ),
        ],
    )

    classifier = ContractRiskClassifier()
    classification_result = classifier.classify(analysis_result)
    print_risk_classification_report(classification_result)