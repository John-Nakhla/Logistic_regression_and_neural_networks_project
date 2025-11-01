import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt

class ManualFeedForward(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            l = nn.Linear(layers[i], layers[i + 1])
            self.layers.append(l)

        for i, l in enumerate(self.layers):
            if i < len(self.layers) - 1:
                init.kaiming_uniform_(l.weight, nonlinearity='relu')
            else:
                init.xavier_uniform_(l.weight)
            init.zeros_(l.bias)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        self.activations = [x]
        self.z_values = []
        out = x
        for i, layer in enumerate(self.layers):
            z = out @ layer.weight.T + layer.bias
            self.z_values.append(z)
            if i < len(self.layers) - 1:
                out = torch.relu(z)
            else:
                out = z
            self.activations.append(out)
        return out

    def manual_backward(self, y_true, lr):
        batch_size = y_true.size(0)
        logits = self.activations[-1]

        exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
        probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        y_onehot = torch.zeros_like(probs)
        y_onehot[torch.arange(batch_size), y_true] = 1.0
        loss = -torch.sum(y_onehot * torch.log(probs + 1e-9)) / batch_size

        delta = (probs - y_onehot) / batch_size

        for i in reversed(range(len(self.layers))):
            a_prev = self.activations[i]
            z = self.z_values[i]
            dW = delta.T @ a_prev
            db = delta.sum(dim=0)
            self.layers[i].weight.grad = dW.clone()
            self.layers[i].bias.grad = db.clone()

            if i > 0:
                delta = delta @ self.layers[i].weight
                z_prev = self.z_values[i - 1]
                delta = delta * (z_prev > 0)

        with torch.no_grad():
            for layer in self.layers:
                layer.weight -= lr * layer.weight.grad
                layer.bias -= lr * layer.bias.grad

        return loss.item()


def train_manual(model, dataloader, lr, device):
    model.train()
    total_loss, total_correct, N = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model.forward(X)
        loss = model.manual_backward(y, lr)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_loss += loss * X.size(0)
        N += X.size(0)
    return total_loss / N, total_correct / N


def validate_manual(model, dataloader, device):
    model.eval()
    total_loss, total_correct, N = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model.forward(X)
            exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
            probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
            loss = -torch.log(probs[torch.arange(y.size(0)), y] + 1e-9).mean()
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * X.size(0)
            N += X.size(0)
    return total_loss / N, total_correct / N


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 784
HIDDEN1 = 512
HIDDEN2 = 128
OUTPUT = 10
LR = 0.01
BATCH_SIZE = 64
PATIENCE = 3
CHECKPOINT_PATH = "best_manual_model.pth"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_ds, val_ds = random_split(full_train_dataset, [50000, 10000])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = ManualFeedForward([INPUT_SIZE, HIDDEN1, HIDDEN2, OUTPUT]).to(device)
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
epoch = 0
best_epoch = 0
print(f"Training manually on {device}...\n")

while True:
    epoch += 1
    train_loss, train_acc = train_manual(model, train_loader, LR, device)
    val_loss, val_acc = validate_manual(model, val_loader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        torch.save(best_model_state, CHECKPOINT_PATH)
        print(f" Validation loss improved. Checkpoint saved to {CHECKPOINT_PATH}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE:
            print(f"\n Early stopping triggered after {epoch} epochs.")
            break

if best_model_state is not None:
    print(f"\n Restoring best model from epoch {best_epoch} with val loss {best_val_loss:.4f}.")
    model.load_state_dict(best_model_state)
    print("\n Restored best model from checkpoint.")

print("\nTraining complete.")
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
