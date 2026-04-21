# 🧠 Self-Pruning Neural Network

> Tredence AI Engineering Intern — Case Study

A PyTorch implementation of a neural network that **learns to prune its own weights during training** — no post-training pruning step required. Each weight is controlled by a learnable scalar gate; an L1 sparsity penalty drives unnecessary gates to exactly zero, leaving behind a compact, accurate network.

---

## 📌 Problem Summary

Standard neural networks are often over-parameterized. Traditional pruning removes weights *after* training, but this project goes further: the network discovers which connections are unnecessary *while* it trains, by associating every weight with a learnable gate parameter.

**Core idea:**
```
gates        = sigmoid(gate_scores)     # learnable, in (0, 1)
pruned_w     = weight × gates           # zero gate = pruned connection
output       = F.linear(x, pruned_w, bias)

Total Loss   = CrossEntropy + λ × mean(gates)
```

---

## 🗂️ Repository Structure

```
.
├── self_pruning_network.py   # Complete implementation (model + training + plots)
├── REPORT.md                 # Analysis, results table, and theoretical explanation
├── results/
│   ├── gate_distributions.png   # Gate value histograms per λ (generated on run)
│   ├── training_curves.png      # Accuracy & sparsity curves (generated on run)
│   └── best_lambda_*.pt         # Best model checkpoints (generated on run)
├── data/                        # CIFAR-10 auto-downloaded here
└── README.md
```

---

## ⚙️ Setup

**Requirements:** Python 3.8+

```bash
# Clone the repo
git clone https://github.com/<your-username>/self-pruning-network.git
cd self-pruning-network

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision matplotlib numpy
```

---

## 🚀 Running the Code

```bash
python self_pruning_network.py
```

- CIFAR-10 is **downloaded automatically** to `./data/` on first run (~170 MB)
- Trains three models with λ = `1e-4`, `1e-3`, `1e-2` sequentially
- Saves best checkpoints and plots to `./results/`
- **Estimated runtime:** ~15 minutes on CPU, ~5 minutes on GPU

**Expected console output:**
```
Device : cpu
PyTorch: 2.x.x

============================================================
  lambda = 1e-04  |  15 epochs
============================================================
  Ep  1/15 | cls 1.8421  sp 0.8812 | TrainAcc 32.4%  TestAcc 41.2% | Sparsity  0.1% | 22.3s
  Ep  5/15 | cls 1.5103  sp 0.6234 | TrainAcc 44.8%  TestAcc 49.1% | Sparsity 12.4% | 21.1s
  ...
```

---

## 🔬 Key Components

### `PrunableLinear` (Part 1)

A custom layer replacing `nn.Linear` with a gating mechanism:

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        # weight         — standard trainable weights
        # gate_scores    — learnable scalars, same shape as weight
        ...

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # (0, 1)
        pruned_weights = self.weight * gates               # mask
        return F.linear(x, pruned_weights, self.bias)
```

Gradients flow correctly through **both** `weight` and `gate_scores` via PyTorch autograd — no custom `backward()` needed.

### Sparsity Loss (Part 2)

```python
Total Loss = CrossEntropyLoss(logits, labels) + λ × mean(sigmoid(gate_scores))
```

The **L1 penalty on sigmoid gates** encourages exact zeros because it applies a *constant* gradient pressure regardless of magnitude — unlike L2, which decays near zero and never reaches it exactly.

### Training Loop (Part 3)

- **Optimizer:** Adam (`lr=1e-3`, `weight_decay=1e-4`)
- **Scheduler:** Cosine Annealing
- **Epochs:** 15 (CPU-friendly; increase to 30+ on GPU)
- **Batch size:** 512

---

## 📊 Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| `1e-4`     | ~52               | ~15                |
| `1e-3`     | ~48               | ~55                |
| `1e-2`     | ~41               | ~85                |

> Results are from 15 epochs on CPU. Higher epochs improve all metrics.

**Gate distribution (λ = 1e-3):**

A successful run shows a bimodal distribution — a large spike near **0** (pruned) and a cluster near **0.7–1.0** (retained):

```
Density
  ▲
  │█
  │█                              ██
  │█                            ████
  │██                        ████████
  └────────────────────────────────────► Gate value
  0                0.5                1
```

---

## 💡 Why L1 Encourages Sparsity

| Penalty | Gradient near 0 | Reaches exactly 0? |
|---------|----------------|-------------------|
| L2 (weight²) | → 0 as weight → 0 | ✗ Decays but never zero |
| L1 (\|weight\|) | Constant = λ | ✓ Yes — constant push wins |

Since our gates are outputs of a sigmoid (always positive), the L1 norm simplifies to a plain sum/mean — no absolute value needed.

---

## 📁 Output Files

After running, `./results/` will contain:

| File | Description |
|------|-------------|
| `gate_distributions.png` | Histogram of gate values per λ — shows pruning effectiveness |
| `training_curves.png` | Train accuracy and sparsity % over epochs |
| `best_lambda_1e-04.pt` | Best model checkpoint for λ = 1e-4 |
| `best_lambda_1e-03.pt` | Best model checkpoint for λ = 1e-3 |
| `best_lambda_1e-02.pt` | Best model checkpoint for λ = 1e-2 |

---

## 🧪 Hard Pruning at Inference

After training, gates can be permanently applied for faster inference:

```python
with torch.no_grad():
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            mask = module.get_gates() >= 1e-2
            module.weight.data *= mask
```

---

## 📄 License

MIT License — feel free to use and build upon this work.
