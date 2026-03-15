# DANP LLM Conciseness

DANP (Decorrelation-adapted Node Perturbation) for finetuning Qwen 0.5B on a conciseness task. Uses activation noise + loss difference for gradient estimation (no backpropagation).

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python danp_llm_conciseness.py --n_train 32 --n_eval 50 --epochs 5 --batch_size 2
```

**Hyperparameters**: eta=1e-3, sigma=1e-3, alpha=1e-4 (no scaling). Dataset: XSum (article → one-sentence summary). GPU recommended.
