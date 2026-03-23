import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW


MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
OUTPUT_DIR = "trained_model"

LABEL2ID = {
    "no_risk": 0,
    "risk_uncertain_time": 1,
    "risk_weak_liability": 2,
    "risk_unclear_payment": 3,
    "risk_dispute_missing_details": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_jsonl(path: str) -> List[Dict]:
    print(f"[INFO] Загружается: {path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"[INFO] Записей: {len(items)}")
    return items


class ContractDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, max_len: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL2ID[label], dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)

    train_path = "dataset_split/train.jsonl"
    val_path = "dataset_split/val.jsonl"

    if not Path(train_path).exists():
        raise FileNotFoundError(f"Не найден {train_path}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Не найден {val_path}")

    train_items = load_jsonl(train_path)
    val_items = load_jsonl(val_path)

    print("[INFO] Загружаю токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("[INFO] Загружаю модель...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    print("[INFO] Создаю датасеты...")
    train_dataset = ContractDataset(train_items, tokenizer, MAX_LEN)
    val_dataset = ContractDataset(val_items, tokenizer, MAX_LEN)

    print("[INFO] Создаю DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=LR)

    print("[INFO] Начинаю обучение...")
    print(f"[INFO] Батчей в train: {len(train_loader)}")
    print(f"[INFO] Батчей в val: {len(val_loader)}")

    for epoch in range(EPOCHS):
        print(f"\n[INFO] Эпоха {epoch + 1}/{EPOCHS}")
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if step % 20 == 0 or step == 1:
                print(f"[TRAIN] epoch={epoch + 1} step={step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"[RESULT] Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[INFO] Модель сохранена в {OUTPUT_DIR}")


if __name__ == "__main__":
    train()

