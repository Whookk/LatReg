import os
import json
import math
import random
import numpy as np
from datetime import datetime
import time
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer

# ============================================================
# CONFIG
# ============================================================


SEEDS = [42, 123, 456, 789, 101]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "gpt2"
VOCAB_SIZE = 50257
MAX_SEQ_LEN = 512

N_LAYERS = 6
HIDDEN_SIZE = 512
NUM_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
FFN_DIM = 2048

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LR = 2e-4
EPOCHS = 1

LAMBDA_MESU = 2e-2
MESU_LAYERS = [2, 3, 4]

DATA_DIR = "./continual_datasets"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ["Wikipedia", "News", "Legal"]

RUN_NAME = f"fox_mesu_5seeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_JSON = os.path.join(RESULTS_DIR, f"{RUN_NAME}.json")


# ============================================================
# SEED
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ROTARY EMBEDDINGS
# ============================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k):
        T = q.size(-2)
        t = torch.arange(T, device=q.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat([-x2, x1], dim=-1)

        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


# ============================================================
# МОДИФІКОВАНА УВАГА
# ============================================================
class FoXAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(HIDDEN_SIZE, 3 * HIDDEN_SIZE)
        self.out = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.rotary = RotaryEmbedding(HEAD_DIM)
        self.forget_gate = nn.Linear(HIDDEN_SIZE, NUM_HEADS)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        q, k = self.rotary(q, k)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        causal = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(causal == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        gate = torch.sigmoid(self.forget_gate(x))
        gate = gate.transpose(1, 2).unsqueeze(-1)
        attn = attn * gate

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# ============================================================
# ТРАНСФОРМЕРНИЙ БЛОК
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(HIDDEN_SIZE)
        self.attn = FoXAttention()
        self.ln2 = nn.LayerNorm(HIDDEN_SIZE)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, HIDDEN_SIZE)
        )

    def forward(self, x, return_latent=False):
        h = x + self.attn(self.ln1(x))
        z = self.ln2(h)
        out = h + self.ffn(z)
        if return_latent:
            return out, z
        return out


# ============================================================
# МОДЕЛЬ
# ============================================================
class PythiaLiteFoX(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, labels=None, return_latents=False):
        x = self.embed(input_ids)
        latents = []

        for i, block in enumerate(self.blocks):
            if return_latents:
                x, z = block(x, return_latent=True)
                latents.append(z)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits, "latents": latents}



def build_loader(domain, split, tokenizer):
    ds = load_from_disk(os.path.join(DATA_DIR, domain))[split]

    def collate(batch):
        ids = [torch.tensor(x["input_ids"][:MAX_SEQ_LEN]) for x in batch]
        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": ids, "labels": labels}

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)


@torch.no_grad()
def estimate_latent_stats(model, loader):
    model.eval()
    mus = {l: 0.0 for l in MESU_LAYERS}
    vars_ = {l: 0.0 for l in MESU_LAYERS}
    n = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch, return_latents=True)
        for l in MESU_LAYERS:
            z = out["latents"][l].reshape(-1, HIDDEN_SIZE)
            mus[l] += z.mean(0)
            vars_[l] += z.var(0, unbiased=False)
        n += 1
    for l in MESU_LAYERS:
        mus[l] /= n
        vars_[l] = torch.clamp(vars_[l] / n, min=1e-6)
    return {"mu": mus, "var": vars_}


def mesu_kl_batch(latents, prev_stats):
    kl = 0.0
    for l in MESU_LAYERS:
        z = latents[l].reshape(-1, HIDDEN_SIZE)
        mu = z.mean(0)
        var = z.var(0, unbiased=False) + 1e-6
        mu0 = prev_stats["mu"][l]
        var0 = prev_stats["var"][l]
        kl += 0.5 * torch.sum(torch.log(var0 / var) + (var + (mu - mu0) ** 2) / var0 - 1)
    return kl


def train_domain(domain, model, loader, optimizer, prev_stats=None):
    model.train()
    optimizer.zero_grad()
    running = 0.0
    step = 0

    pbar = tqdm(loader, desc=f"Training {domain}")
    for batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch, return_latents=prev_stats is not None)
        base_loss = out["loss"]

        if prev_stats is not None:
            kl = mesu_kl_batch(out["latents"], prev_stats)
            loss = base_loss + LAMBDA_MESU * kl
        else:
            loss = base_loss

        (loss / GRAD_ACCUM_STEPS).backward()
        running += loss.item()
        step += 1

        if step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        pbar.set_postfix(loss=f"{running / step:.4f}")

    return running / step


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        losses.append(model(**batch)["loss"].item())
    m = float(np.mean(losses))
    return {"loss": m, "ppl": float(math.exp(m))}


# ============================================================
# СТАТИСТИЧНІ МЕТРИКИ
# ============================================================
def compute_stats(values):

    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0

    if n > 1:
        sem = std / np.sqrt(n)
        # Критерій Стьюдента
        ci_95 = stats.t.ppf(0.975, n - 1) * sem
    else:
        ci_95 = 0.0

    return {
        "mean": float(mean),
        "std": float(std),
        "ci_95": float(ci_95),
        "values": [float(v) for v in values]
    }


# ============================================================
# ОДИН ЗАПУСК
# ============================================================
def run_single_seed(seed, tokenizer, loaders_train, loaders_val):
    set_seed(seed)

    # Випадкова ініціалізація ваг моделі для кожного окремого запуску
    model = PythiaLiteFoX().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 0. Init Evaluation
    eval_init = {d: evaluate(model, loaders_val[d]) for d in DOMAINS}

    # 1. Wikipedia
    train_domain("Wikipedia", model, loaders_train["Wikipedia"], optimizer)
    wiki_stats = estimate_latent_stats(model, loaders_val["Wikipedia"])
    eval_wiki = {d: evaluate(model, loaders_val[d]) for d in DOMAINS}

    # 2. News
    train_domain("News", model, loaders_train["News"], optimizer, prev_stats=wiki_stats)
    news_stats = estimate_latent_stats(model, loaders_val["News"])
    eval_news = {d: evaluate(model, loaders_val[d]) for d in DOMAINS}

    # 3. Legal
    train_domain("Legal", model, loaders_train["Legal"], optimizer, prev_stats=news_stats)
    eval_legal = {d: evaluate(model, loaders_val[d]) for d in DOMAINS}


    forgetting = {
        "wiki_after_news": eval_news["Wikipedia"]["ppl"] - eval_wiki["Wikipedia"]["ppl"],
        "wiki_after_legal": eval_legal["Wikipedia"]["ppl"] - eval_wiki["Wikipedia"]["ppl"],
        "news_after_legal": eval_legal["News"]["ppl"] - eval_news["News"]["ppl"],
    }

    return {
        "seed": seed,
        "eval_init": eval_init,
        "eval_wiki": eval_wiki,
        "eval_news": eval_news,
        "eval_legal": eval_legal,
        "forgetting": forgetting
    }


# ============================================================
# Основний запуск
# ============================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    loaders_train = {d: build_loader(d, "train", tokenizer) for d in DOMAINS}
    loaders_val = {d: build_loader(d, "validation", tokenizer) for d in DOMAINS}

    all_results = []
    start_global_time = time.time()


    for seed_idx, seed in enumerate(SEEDS, start=1):
        print(f"\n{'=' * 60}\n🚀 ЗАПУСК {seed_idx}/{len(SEEDS)} | RANDOM SEED: {seed}\n{'=' * 60}")
        run_stats = run_single_seed(seed, tokenizer, loaders_train, loaders_val)
        all_results.append(run_stats)

    # ----------------------------------------------------
    # ЗБІР МАСИВІВ
    # ----------------------------------------------------
    final_metrics = {
        "final_ppl_legal": [r["eval_legal"]["Legal"]["ppl"] for r in all_results],
        "final_ppl_news": [r["eval_legal"]["News"]["ppl"] for r in all_results],
        "final_ppl_wiki": [r["eval_legal"]["Wikipedia"]["ppl"] for r in all_results],
        "forgetting_wiki_total": [r["forgetting"]["wiki_after_legal"] for r in all_results],
        "forgetting_news_total": [r["forgetting"]["news_after_legal"] for r in all_results],
    }


    summary_stats = {metric_name: compute_stats(values) for metric_name, values in final_metrics.items()}

    final_report = {
        "seeds_used": SEEDS,
        "summary_statistics": summary_stats,
        "raw_runs": all_results,
        "total_experiment_time_hrs": (time.time() - start_global_time) / 3600.0
    }


    with open(RESULTS_JSON, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n✔ Усі 5 запусків успішно завершено")
    print("\n=== ФІНАЛЬНА ЗВЕДЕНА СТАТИСТИКА (N=5) ===")
    for k, v in summary_stats.items():
        print(f"{k}: {v['mean']:.2f} ± {v['std']:.2f} (95% довірчий інтервал: ±{v['ci_95']:.2f})")


if __name__ == "__main__":
    main()