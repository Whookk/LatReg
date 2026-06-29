# LatReg: Continual Learning for Language Models using Latent Distribution Regularization

> Official implementation of the LatReg continual learning framework for autoregressive language models.

## Overview

Forgetting remains one of the major challenges in continual learning of Language Models (LMs). This repository contains the implementation of a continual learning framework that combines:

- **Forgetting Attention** – a modified self-attention mechanism with learnable forgetting gates;
- **LatReg (Latent Distribution Regularization)** – a latent distribution regularization method that preserves previously learned knowledge without storing previous datasets.

The project includes:

- automatic dataset preprocessing;
- continual learning pipeline;
- multi-seed evaluation;
- statistical analysis of experimental results.

---

## Method

The continual learning process follows three sequential domains:

```
Wikipedia → News → Legal
```

During training:

1. The model is first trained on **Wikipedia**.
2. Latent feature statistics (mean and variance) are estimated.
3. Training continues on **News**, while **LatReg** constrains latent representations.
4. Statistics are updated.
5. Training proceeds on the **Legal** domain.

The regularization objective minimizes the KL divergence between latent feature distributions of consecutive domains.

---

## Repository Structure

```
.
├── LatReg_5_rand_seeds.py      # Main continual learning experiments
├── dataset_preprocess.py               # Wikipedia & News preprocessing
├── legal_preprocess.py                 # EurLex preprocessing
├── requirements.txt                #Requirements of the project
└── README.md
```

---

## Features

- Continual language model training
- Forgetting Attention implementation
- LatReg latent regularization
- Rotary positional embeddings
- Multi-domain continual learning
- Multi-seed evaluation (5 random seeds)
- Statistical significance reporting
- Automatic preprocessing pipeline
- HuggingFace Datasets integration

---

# Installation

Clone the repository

```bash
git clone https://github.com/Whookk/LatReg.git

cd LatReg
```

Create environment

```bash
python -m venv .venv

source .venv/bin/activate      # Linux/Mac

.venv\Scripts\activate         # Windows
```

Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch transformers datasets pandas numpy tqdm scipy langdetect
```

---

# Dataset Preparation

The project automatically downloads datasets from HuggingFace.

## Wikipedia + News

```bash
python dataset_preprocess.py
```

The script performs

- dataset downloading
- cleaning
- language filtering
- quality filtering
- tokenization
- train/validation/test split
- token budget truncation (35M tokens)

Generated datasets are saved to

```
continual_datasets/
```

---

## Legal Dataset

```bash
python legal_preprocess.py
```

Additional preprocessing includes

- duplicate removal
- MD5 hashing
- English language filtering
- quality filtering
- token budget truncation

The Legal domain is based on the **EurLex** dataset.

---

# Running Experiments

Run the continual learning experiments:

```bash
python LatReg_5_rand_seeds.py
```

The script automatically

- initializes the model
- trains sequentially on all domains
- estimates latent statistics
- applies **LatReg** regularization
- evaluates after every task
- repeats experiments using **5 different random seeds**
- computes confidence intervals
- saves results as JSON

---

# Model Architecture

The implementation includes

- Rotary Positional Embeddings
- Forgetting Attention
- Transformer blocks
- GPT-style decoder architecture
- LayerNorm
- GELU feed-forward networks

Configuration:

| Parameter | Value |
|-----------|------:|
| Layers | 6 |
| Hidden Size | 512 |
| Attention Heads | 8 |
| FFN Dimension | 2048 |
| Context Length | 512 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |

---

# LatReg Regularization

After each domain, latent statistics are estimated:

- mean
- variance

During continual learning, the following KL divergence is minimized

```
KL(N(μ,σ²) || N(μ₀,σ₀²))
```

This encourages latent representations of previous domains to remain stable while allowing adaptation to new tasks.

---

# Evaluation Metrics

The implementation reports

- Cross Entropy Loss
- Perplexity (PPL)
- Forgetting
- Mean
- Standard Deviation
- 95% Confidence Interval

Forgetting is measured as the increase in perplexity after learning subsequent domains.

---

# Output

Results are automatically saved into

```
results/
```

Example output

```
results/
└── latreg_5seeds_YYYYMMDD_HHMMSS.json
```

The JSON file contains

- raw results for every seed
- summary statistics
- confidence intervals
- total experiment time

---

# Datasets

The experiments use

| Domain | Dataset |
|---------|----------|
| Wikipedia | Wikimedia English Wikipedia |
| News | CC-News |
| Legal | EurLex |

Each dataset is automatically downloaded through the HuggingFace Datasets library.

---

# Reproducibility

Default configuration

- 5 random seeds
- deterministic initialization
- fixed train/validation/test split
- automatic tokenizer loading
- fixed token budget
- reproducible preprocessing pipeline

Random seeds used

```
42
123
456
789
101
```

---

# Citation

If you use this implementation in your research, please cite the accompanying paper.

```bibtex
@article{
  title={ADAPTIVE KNOWLEDGE REGULARIZATION FOR CONTINUAL LEARNING IN TRANSFORMER ARCHITECTURES},
  author={Arsentii Kriukov, Yurii Pushkarenko},
  year={2026}
}
```

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This project builds upon the following open-source libraries:

- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- NumPy
- SciPy
- Pandas

---

## Contact

For questions, suggestions, or collaboration, please open an issue or submit a pull request.
