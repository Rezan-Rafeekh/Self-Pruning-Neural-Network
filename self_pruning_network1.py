"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
Tredence AI Engineering Intern – Case Study Submission

Fixes vs v1:
  - Sparsity loss is now MEAN (not sum) of gates → λ stays in a sane range
  - Smaller network (fewer params) → much faster on CPU
  - Fewer epochs (15) with progress printed every epoch
  - pin_memory disabled on CPU to suppress the warning
  - num_workers=0 on Windows (avoids multiprocessing pickle issues)
  - Larger batch size (512) → fewer steps per epoch

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import platform

# ─────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear augmented with learnable sigmoid gates.

    Forward pass:
        gates        = sigmoid(gate_scores)      -- squash to (0, 1)
        pruned_w     = weight * gates            -- element-wise mask
        out          = F.linear(x, pruned_w, bias)

    Gradients flow through both `weight` and `gate_scores` automatically
    because sigmoid and element-wise multiply are fully differentiable.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Gate scores – same shape as weight.
        # Init at 2.0 → sigmoid(2) ≈ 0.88 so training starts with open gates.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates          = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """Mean of gate values – keeps magnitude in [0,1] regardless of layer size."""
        return torch.sigmoid(self.gate_scores).mean()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────
# Network Definition  (compact for CPU speed)
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Compact feed-forward net for CIFAR-10 (32x32x3 -> 10 classes).
    Uses PrunableLinear throughout so every weight participates in gating.
    """

    def __init__(self, hidden_dims=(512, 256)):
        super().__init__()
        input_dim = 3 * 32 * 32   # 3072 after flattening

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(PrunableLinear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))
            prev = h

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = PrunableLinear(prev, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        return self.classifier(x)

    def sparsity_loss(self) -> torch.Tensor:
        """
        Average sparsity loss across all PrunableLinear layers.
        Using MEAN (not sum) keeps the value in [0, 1] regardless of
        network size, so lambda can be set intuitively.
        """
        losses = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                losses.append(module.sparsity_loss())
        return torch.stack(losses).mean()

    def get_all_gates(self) -> torch.Tensor:
        """Concatenate gate values from every PrunableLinear (for plotting)."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().cpu().flatten())
        return torch.cat(all_gates)

    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage of gates below threshold (effectively pruned)."""
        gates = self.get_all_gates()
        return (gates < threshold).float().mean().item() * 100.0


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 512):
    """CIFAR-10 loaders with standard augmentation. CPU-safe settings."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                              download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                              download=True, transform=test_transform)

    # num_workers=0 is safest on Windows; pin_memory only when CUDA is available
    use_cuda    = torch.cuda.is_available()
    num_workers = 0 if platform.system() == "Windows" else 2

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_cuda)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_cuda)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, lambda_sparsity, device):
    model.train()
    total_loss = total_cls = total_sp = correct = n = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits   = model(images)
        cls_loss = criterion(logits, labels)
        sp_loss  = model.sparsity_loss()          # value in [0, 1]

        # Total Loss = CrossEntropy + lambda * mean(gates)
        loss = cls_loss + lambda_sparsity * sp_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs          = labels.size(0)
        total_loss += loss.item()    * bs
        total_cls  += cls_loss.item()* bs
        total_sp   += sp_loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += bs

    return {
        "loss":     total_loss / n,
        "cls_loss": total_cls  / n,
        "sp_loss":  total_sp   / n,
        "acc":      correct / n * 100,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits      = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += labels.size(0)
    return total_loss / n, correct / n * 100


def run_experiment(lambda_val, epochs, device, train_loader, test_loader, results_dir):
    print(f"\n{'='*60}")
    print(f"  lambda = {lambda_val:.0e}  |  {epochs} epochs")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history  = {"train_acc": [], "sparsity": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0          = time.time()
        stats       = train_one_epoch(model, train_loader, optimizer,
                                      criterion, lambda_val, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        sp = model.compute_sparsity()
        history["train_acc"].append(stats["acc"])
        history["sparsity"].append(sp)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(results_dir, f"best_lambda_{lambda_val:.0e}.pt"))

        print(f"  Ep {epoch:2d}/{epochs} | "
              f"cls {stats['cls_loss']:.4f}  sp {stats['sp_loss']:.4f} | "
              f"TrainAcc {stats['acc']:.1f}%  TestAcc {test_acc:.1f}% | "
              f"Sparsity {sp:.1f}% | {time.time()-t0:.1f}s")

    # Reload best checkpoint
    model.load_state_dict(
        torch.load(os.path.join(results_dir, f"best_lambda_{lambda_val:.0e}.pt"),
                   map_location=device))
    _, final_acc = evaluate(model, test_loader, criterion, device)
    final_sp     = model.compute_sparsity()

    print(f"\n  Best checkpoint -> TestAcc={final_acc:.2f}%  Sparsity={final_sp:.1f}%")
    return {
        "lambda":   lambda_val,
        "test_acc": final_acc,
        "sparsity": final_sp,
        "gates":    model.get_all_gates().numpy(),
        "history":  history,
    }


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_gate_distribution(results, results_dir):
    n     = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, res, c in zip(axes, results, colors):
        gates      = res["gates"]
        pruned_pct = (gates < 1e-2).mean() * 100
        ax.hist(gates, bins=80, color=c, edgecolor="white", linewidth=0.4, density=True)
        ax.axvline(0.01, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Threshold\n{pruned_pct:.1f}% pruned")
        ax.set_title(f"lambda = {res['lambda']:.0e}\n"
                     f"Sparsity {res['sparsity']:.1f}%  |  Acc {res['test_acc']:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Gate value", fontsize=11)
        ax.set_ylabel("Density",    fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Distribution of Learned Gate Values per Lambda", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "gate_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


def plot_training_curves(results, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors    = ["#4C72B0", "#DD8452", "#55A868"]

    for res, c in zip(results, colors):
        label = f"lambda={res['lambda']:.0e}"
        axes[0].plot(res["history"]["train_acc"], label=label, color=c)
        axes[1].plot(res["history"]["sparsity"],  label=label, color=c)

    for ax, title, ylabel in zip(
            axes,
            ["Training Accuracy", "Sparsity During Training"],
            ["Accuracy (%)", "Sparsity (%)"]):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    RESULTS_DIR = "./results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")

    EPOCHS     = 15            # ~8-15 min on CPU total; raise to 30 on GPU
    BATCH_SIZE = 512           # larger batch = fewer steps per epoch
    LAMBDAS    = [1e-4, 1e-3, 1e-2]   # low / medium / high sparsity pressure

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    all_results = []
    for lam in LAMBDAS:
        res = run_experiment(lam, EPOCHS, device,
                             train_loader, test_loader, RESULTS_DIR)
        all_results.append(res)

    # ── Summary table ──────────────────────────────────────
    print("\n\n" + "="*52)
    print(f"{'Lambda':>10} | {'Test Acc (%)':>14} | {'Sparsity (%)':>14}")
    print("-"*52)
    for r in all_results:
        print(f"{r['lambda']:>10.0e} | {r['test_acc']:>14.2f} | {r['sparsity']:>14.1f}")
    print("="*52)

    plot_gate_distribution(all_results, RESULTS_DIR)
    plot_training_curves(all_results, RESULTS_DIR)
    print("\nAll done! Check ./results/")


if __name__ == "__main__":
    main()
