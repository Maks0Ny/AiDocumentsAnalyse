import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

@dataclass
class ContractSection:
    """Описывает найденный раздел договора."""
    title: str
    content: str


@dataclass
class ContractPreprocessResult:
    """Результат предварительной обработки договора."""
    source_path: str
    raw_text: str
    cleaned_text: str
    normalized_text: str
    sections: List[ContractSection] = field(default_factory=list)
    detected_blocks: Dict[str, str] = field(default_factory=dict)


class ContractLoaderError(Exception):
    pass


class ContractPreprocessor:
    """
    Класс для:
    1. загрузки договора из TXT / DOCX / PDF
    2. очистки текста
    3. нормализации
    4. первичного структурного анализа
    """


    # Шаблоны разделов
    SECTION_PATTERNS = {
        "subject": [
            r"предмет договора",
            r"предмет",
        ],
        "rights_obligations": [
            r"права и обязанности сторон",
            r"обязанности сторон",
            r"права сторон",
        ],
        "price": [
            r"цена договора",
            r"стоимость услуг",
            r"стоимость работ",
            r"цена",
            r"порядок расчетов",
            r"порядок оплаты",
            r"оплата",
        ],
        "term": [
            r"срок действия договора",
            r"сроки выполнения",
            r"срок оказания услуг",
            r"срок аренды",
            r"срок",
        ],
        "liability": [
            r"ответственность сторон",
            r"ответственность",
            r"санкции",
            r"неустойка",
        ],
        "termination": [
            r"порядок расторжения",
            r"изменение и расторжение договора",
            r"расторжение договора",
            r"изменение договора",
        ],
        "disputes": [
            r"порядок разрешения споров",
            r"разрешение споров",
            r"урегулирование споров",
            r"споры",
        ],
        "final": [
            r"заключительные положения",
            r"прочие условия",
        ],
        "requisites": [
            r"реквизиты сторон",
            r"адреса и реквизиты сторон",
            r"юридические адреса и реквизиты",
        ],
    }

    def load_document(self, path: str) -> str:

        if not os.path.exists(path):
            raise ContractLoaderError(f"Файл не найден: {path}")

        extension = os.path.splitext(path)[1].lower()

        if extension == ".txt":
            return self._load_txt(path)
        elif extension == ".docx":
            return self._load_docx(path)
        elif extension == ".pdf":
            return self._load_pdf(path)
        else:
            raise ContractLoaderError(
                f"Неподдерживаемый формат файла: {extension}. "
                "Допустимы: .txt, .docx, .pdf"
            )

    def _load_txt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="cp1251") as file:
                return file.read()
        except Exception as e:
            raise ContractLoaderError(f"Ошибка чтения TXT: {e}") from e

    def _load_docx(self, path: str) -> str:
        if Document is None:
            raise ContractLoaderError(
            )
        try:
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            raise ContractLoaderError(f"Ошибка чтения DOCX: {e}") from e

    def _load_pdf(self, path: str) -> str:
        if pdfplumber is None:
            raise ContractLoaderError(
            )
        try:
            pages_text = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
            return "\n".join(pages_text)
        except Exception as e:
            raise ContractLoaderError(f"Ошибка чтения PDF: {e}") from e

    def clean_text(self, text: str) -> str:

        if not text:
            return ""

        # Убираем неразрывные пробелы и табуляцию
        text = text.replace("\xa0", " ").replace("\t", " ")

        # Убираем лишние пробелы внутри строк
        text = re.sub(r"[ ]{2,}", " ", text)

        # Убираем лишние переводы строк
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Убираем пробелы вокруг переводов строк
        text = re.sub(r" *\n *", "\n", text)

        # Удаляем мусорные символы, но сохраняем базовую пунктуацию
        text = re.sub(r"[^\w\s.,;:!?()\"«»№/\-—\n%]", "", text, flags=re.UNICODE)

        return text.strip()

    def normalize_text(self, text: str) -> str:

        if not text:
            return ""

        normalized = text.lower()
        normalized = normalized.replace("ё", "е")
        normalized = normalized.replace("–", "-").replace("—", "-")
        normalized = normalized.replace("«", "\"").replace("»", "\"")

        # Снова убираем лишние пробелы
        normalized = re.sub(r"[ ]{2,}", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized.strip()

    def split_into_sections(self, text: str) -> List[ContractSection]:

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sections: List[ContractSection] = []

        current_title = "Введение"
        current_content: List[str] = []

        section_title_pattern = re.compile(
            r"^(\d+(\.\d+)?\.?\s+)?[А-ЯA-Z][А-ЯA-Zа-яa-zёЁ0-9\s\-\"()]{3,}$"
        )

        for line in lines:
            is_possible_title = len(line) < 120 and section_title_pattern.match(line)

            if is_possible_title:
                lowered = line.lower()
                if any(
                    keyword in lowered
                    for keyword_list in self.SECTION_PATTERNS.values()
                    for keyword in keyword_list
                ):
                    if current_content:
                        sections.append(
                            ContractSection(
                                title=current_title,
                                content="\n".join(current_content).strip()
                            )
                        )
                    current_title = line
                    current_content = []
                    continue

            current_content.append(line)

        if current_content:
            sections.append(
                ContractSection(
                    title=current_title,
                    content="\n".join(current_content).strip()
                )
            )

        return sections

    def detect_key_blocks(self, sections: List[ContractSection], normalized_text: str) -> Dict[str, str]:

        detected_blocks: Dict[str, str] = {}

        for section in sections:
            title_lower = section.title.lower()
            for block_name, patterns in self.SECTION_PATTERNS.items():
                if any(re.search(pattern, title_lower) for pattern in patterns):
                    detected_blocks[block_name] = section.content

        for block_name, patterns in self.SECTION_PATTERNS.items():
            if block_name not in detected_blocks:
                if any(re.search(pattern, normalized_text) for pattern in patterns):
                    detected_blocks[block_name] = "[Блок найден в тексте, но не выделен как отдельный раздел]"

        return detected_blocks

    def preprocess(self, path: str) -> ContractPreprocessResult:

        raw_text = self.load_document(path)
        cleaned_text = self.clean_text(raw_text)
        normalized_text = self.normalize_text(cleaned_text)
        sections = self.split_into_sections(cleaned_text)
        detected_blocks = self.detect_key_blocks(sections, normalized_text)

        return ContractPreprocessResult(
            source_path=path,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            sections=sections,
            detected_blocks=detected_blocks,
        )


def print_preprocess_report(result: ContractPreprocessResult) -> None:

    print("=" * 80)
    print("ОТЧЁТ ПО ПРЕДВАРИТЕЛЬНОЙ ОБРАБОТКЕ ДОГОВОРА")
    print("=" * 80)
    print(f"Файл: {result.source_path}\n")

    print("Найденные разделы:")
    if result.sections:
        for i, section in enumerate(result.sections, start=1):
            print(f"{i}. {section.title}")
    else:
        print("Разделы не выделены.")

    print("\nОбнаруженные ключевые блоки:")
    if result.detected_blocks:
        for block_name in result.detected_blocks:
            print(f"- {block_name}")
    else:
        print("Ключевые блоки не обнаружены.")

    print("\nФрагмент очищенного текста:")
    preview = result.cleaned_text[:1000]
    print(preview + ("..." if len(result.cleaned_text) > 1000 else ""))


if __name__ == "__main__":

    file_path = "example_contract.docx"

    processor = ContractPreprocessor()

    try:
        result = processor.preprocess(file_path)
        print_preprocess_report(result)
    except ContractLoaderError as e:
        print(f"Ошибка: {e}")