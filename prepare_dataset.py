import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: str, items: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(
    items: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(seed)
    items = items[:]
    random.shuffle(items)

    n = len(items)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]

    return train_items, val_items, test_items


if __name__ == "__main__":
    source_path = "contracts_dataset.jsonl"
    output_dir = Path("dataset_split")
    output_dir.mkdir(exist_ok=True)

    data = load_jsonl(source_path)
    train_data, val_data, test_data = split_dataset(data)

    save_jsonl(output_dir / "train.jsonl", train_data)
    save_jsonl(output_dir / "val.jsonl", val_data)
    save_jsonl(output_dir / "test.jsonl", test_data)

    print(f"Всего записей: {len(data)}")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")