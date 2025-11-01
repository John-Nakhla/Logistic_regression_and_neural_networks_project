import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class FlexibleFeedForward(nn.Module):
    def __init__(self, layers: List[int]):
        super().__init__()
        assert len(layers) >= 3
        self.layers_sizes = layers
        
        # Build linear layers
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            l = nn.Linear(layers[i], layers[i+1])
            self.linears.append(l)

        # Initialization
        for i, linear in enumerate(self.linears):
            # If next layer is hidden with ReLU use Kaiming; else use Xavier
            # Here we assume all intermediate layers use ReLU
            if i < len(self.linears) - 1:
                init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            else:
                # final layer
                init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                init.zeros_(linear.bias)

    def forward(self, x):
        # Expect input tensors of shape [batch, C, H, W] or [batch, features]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # flatten
        out = x
        for i, linear in enumerate(self.linears):
            out = linear(out)
            if i < len(self.linears) - 1:
                out = F.relu(out)
        return out  # logits (no softmax) - suitable for CrossEntropyLoss

# -----------------------
# ---------- B2 ----------
# Training utilities (manual SGD, train/validate loops), supports batch processing
# -----------------------
def manual_sgd_step(model: nn.Module, lr: float):
    """
    Manually apply SGD update: param = param - lr * grad
    Works in-place and wrapped in torch.no_grad()
    """
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad.data

def train_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, lr: float, device: torch.device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    batch_losses = []
    batch_accuracies = []
    N = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # Zero gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Forward
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * X.size(0)

        # Backward
        loss.backward()

        # Manual update
        manual_sgd_step(model, lr)

        # Metrics
        _, preds = torch.max(logits.detach(), dim=1)
        correct = (preds == y).sum().item()
        total_correct += correct
        N += X.size(0)

        batch_losses.append(loss.item())
        batch_accuracies.append(correct / X.size(0))

    avg_loss = total_loss / N
    avg_acc = total_correct / N
    # compute batch-level statistics for plotting error bars
    batch_loss_mean = float(np.mean(batch_losses))
    batch_loss_std = float(np.std(batch_losses, ddof=1)) if len(batch_losses) > 1 else 0.0
    batch_acc_mean = float(np.mean(batch_accuracies))
    batch_acc_std = float(np.std(batch_accuracies, ddof=1)) if len(batch_accuracies) > 1 else 0.0

    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "batch_loss_mean": batch_loss_mean,
        "batch_loss_std": batch_loss_std,
        "batch_acc_mean": batch_acc_mean,
        "batch_acc_std": batch_acc_std
    }

def validate_epoch(model: nn.Module, dataloader: DataLoader, loss_fn, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    N = 0
    batch_losses = []
    batch_accuracies = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * X.size(0)
            _, preds = torch.max(logits, dim=1)
            correct = (preds == y).sum().item()
            total_correct += correct
            N += X.size(0)

            batch_losses.append(loss.item())
            batch_accuracies.append(correct / X.size(0))

    avg_loss = total_loss / N
    avg_acc = total_correct / N
    batch_loss_mean = float(np.mean(batch_losses))
    batch_loss_std = float(np.std(batch_losses, ddof=1)) if len(batch_losses) > 1 else 0.0
    batch_acc_mean = float(np.mean(batch_accuracies))
    batch_acc_std = float(np.std(batch_accuracies, ddof=1)) if len(batch_accuracies) > 1 else 0.0

    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "batch_loss_mean": batch_loss_mean,
        "batch_loss_std": batch_loss_std,
        "batch_acc_mean": batch_acc_mean,
        "batch_acc_std": batch_acc_std
    }

# -----------------------
# ---------- B3 ----------
# Plotting utilities (loss/accuracy curves with error bars and convergence analysis)
# -----------------------
def plot_metrics(epochs: List[int],
                 train_means: List[float],
                 train_stds: List[float],
                 val_means: List[float],
                 val_stds: List[float],
                 ylabel: str,
                 title: str,
                 filename: str = None):
    plt.figure(figsize=(8,5))
    # training line with error bars (std)
    plt.errorbar(epochs, train_means, yerr=train_stds, label='Train', marker='o', capsize=4)
    plt.errorbar(epochs, val_means, yerr=val_stds, label='Val', marker='o', capsize=4)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

def plot_convergence(epochs: List[int], train_losses: List[float], val_losses: List[float], filename: str = None):
    # Plot train - val gap as convergence indicator
    gap = np.array(train_losses) - np.array(val_losses)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.plot(epochs, gap, marker='x', linestyle='--', label='Train - Val Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence analysis: Loss & Gap')
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

# -----------------------
# ---------- Main: configuration and training run ----------
# -----------------------
def main():
    # Configuration (B2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = 28 * 28
    H1 = 512
    H2 = 128
    NUM_CLASSES = 10

    LR = 0.01
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    TRAIN_SIZE = 50000
    VAL_SIZE = 10000

    # Data transforms and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)

    train_ds, val_ds = random_split(full_train, [TRAIN_SIZE, VAL_SIZE])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model (B1) - flexible architecture
    model = FlexibleFeedForward([INPUT_SIZE, H1, H2, NUM_CLASSES]).to(device)
    loss_fn = nn.CrossEntropyLoss()  # cross-entropy (B2 requirement)

    # Trackers for plotting
    epochs_list = []
    train_losses = []
    val_losses = []
    train_loss_stds = []
    val_loss_stds = []
    train_accs = []
    val_accs = []
    train_acc_stds = []
    val_acc_stds = []

    print(f"Training on device: {device}")
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_stats = train_epoch(model, train_loader, loss_fn, LR, device)
        val_stats = validate_epoch(model, val_loader, loss_fn, device)

        # Logging
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {tr_stats['avg_loss']:.4f} | Train Acc: {tr_stats['avg_acc']*100:.2f}% | "
              f"Val Loss: {val_stats['avg_loss']:.4f} | Val Acc: {val_stats['avg_acc']*100:.2f}%")

        # Save stats
        epochs_list.append(epoch)
        train_losses.append(tr_stats['avg_loss'])
        val_losses.append(val_stats['avg_loss'])
        train_loss_stds.append(tr_stats['batch_loss_std'])
        val_loss_stds.append(val_stats['batch_loss_std'])
        train_accs.append(tr_stats['avg_acc'])
        val_accs.append(val_stats['avg_acc'])
        train_acc_stds.append(tr_stats['batch_acc_std'])
        val_acc_stds.append(val_stats['batch_acc_std'])

    # B3: Plots
    plot_metrics(epochs_list,
                 train_losses, train_loss_stds,
                 val_losses, val_loss_stds,
                 ylabel='Loss', title='Training & Validation Loss (with error bars)',
                 filename='loss_with_errorbars.png')

    plot_metrics(epochs_list,
                 train_accs, train_acc_stds,
                 val_accs, val_acc_stds,
                 ylabel='Accuracy', title='Training & Validation Accuracy (with error bars)',
                 filename='accuracy_with_errorbars.png')

    plot_convergence(epochs_list, train_losses, val_losses, filename='convergence.png')

if __name__ == "__main__":
    main()
