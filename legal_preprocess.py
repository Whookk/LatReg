import os
import re
import random
import hashlib
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from langdetect import detect


# =======================
# CONFIG
# =======================

SEED = 42
TARGET_TOKENS = 35_000_000

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

MIN_TEXT_LEN = 200

MODEL_NAME = "EleutherAI/pythia-160m"
CACHE_DIR = "./hf_cache"
OUTPUT_DIR = "./continual_datasets"

LEGAL_CONFIG = {
    "name": "eurlex",
    "config": None,
    "text_column": "text",
}

# =======================
# UTILS
# =======================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_text(example, text_column):
    text = example[text_column]

    if text is None:
        return {"text": ""}

    text = re.sub(r"\s+", " ", text).strip()
    return {"text": text}


# ---------- FILTERS ----------

def length_filter(example):
    return len(example["text"]) > MIN_TEXT_LEN


def language_filter(example):
    try:
        return detect(example["text"]) == "en"
    except Exception:
        return False


def quality_filter(example):
    text = example["text"]

    if text.count("http") > 3:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False

    return True


# ---------- DEDUP ----------

def add_hash(example):
    h = hashlib.md5(example["text"].encode("utf-8")).hexdigest()
    return {"hash": h}


def deduplicate(dataset):
    seen = set()
    indices = []

    for i, ex in enumerate(tqdm(dataset, desc="Deduplicating")):
        if ex["hash"] not in seen:
            seen.add(ex["hash"])
            indices.append(i)

    return dataset.select(indices).remove_columns(["hash"])


# ---------- TOKENIZATION ----------

def tokenize_fn(batch, tokenizer):
    tokens = tokenizer(
        batch["text"],
        truncation=False,
        add_special_tokens=False
    )
    return {"input_ids": tokens["input_ids"]}


def truncate_to_token_budget(dataset, target_tokens):
    total_tokens = 0
    selected_indices = []

    for idx, ex in enumerate(tqdm(dataset, desc="Accumulating tokens")):
        n_tokens = len(ex["input_ids"])

        if total_tokens + n_tokens > target_tokens:
            break

        total_tokens += n_tokens
        selected_indices.append(idx)

    truncated = dataset.select(selected_indices)
    return truncated, total_tokens


def split_dataset(dataset):
    n = len(dataset)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return DatasetDict({
        "train": dataset.select(range(0, train_end)),
        "validation": dataset.select(range(train_end, val_end)),
        "test": dataset.select(range(val_end, n)),
    })


# =======================
# MAIN PIPELINE
# =======================

def main():
    print(f"PyTorch version: {torch.__version__}")

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 50)
    print("Processing domain: Legal (EurLex)")
    print("=" * 50)

    ds = load_dataset(
        LEGAL_CONFIG["name"],
        LEGAL_CONFIG["config"],
        split="train",
        cache_dir=CACHE_DIR
    )

    ds = ds.shuffle(seed=SEED)

    # ---- CLEAN ----
    ds = ds.map(
        clean_text,
        fn_kwargs={"text_column": LEGAL_CONFIG["text_column"]},
        remove_columns=ds.column_names,
        num_proc=4
    )

    # ---- REMOVE EMPTY ----
    ds = ds.filter(lambda x: len(x["text"]) > 0)

    # ---- DEDUP ----
    ds = ds.map(add_hash, num_proc=4)
    ds = deduplicate(ds)

    # ---- FILTERS ----
    ds = ds.filter(length_filter, num_proc=4)
    ds = ds.filter(language_filter, num_proc=4)
    ds = ds.filter(quality_filter, num_proc=4)

    print(f"Documents after cleaning: {len(ds):,}")

    # ---- TOKENIZE ----
    tokenized = ds.map(
        tokenize_fn,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        num_proc=4,
        desc="Tokenizing"
    )

    # ---- TOKEN BUDGET ----
    tokenized, total_tokens = truncate_to_token_budget(
        tokenized,
        TARGET_TOKENS
    )

    print(f"Tokens retained: {total_tokens:,}")

    # ---- SPLIT ----
    split_ds = split_dataset(tokenized)

    save_path = os.path.join(OUTPUT_DIR, "Legal")
    split_ds.save_to_disk(save_path)

    stats = {
        "Domain": "Legal",
        "Documents": len(tokenized),
        "Train examples": len(split_ds["train"]),
        "Val examples": len(split_ds["validation"]),
        "Test examples": len(split_ds["test"]),
        "Tokens": total_tokens
    }

    stats_df = pd.DataFrame([stats])
    print("\nDataset statistics:")
    print(stats_df)

    stats_path = os.path.join(OUTPUT_DIR, "legal_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved stats to {stats_path}")


# =======================
# ENTRY POINT
# =======================

if __name__ == "__main__":
    main()
