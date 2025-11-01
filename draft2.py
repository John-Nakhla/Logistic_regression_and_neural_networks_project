import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================================
# 1. Utility Functions
# ==========================================================

def relu(x):
    return torch.clamp(x, min=0.0)

def relu_derivative(x):
    return (x > 0).float()

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=1, keepdim=True)

def cross_entropy_loss(y_hat, y_true):
    # y_true: tensor of class indices (not one-hot)
    batch_size = y_hat.size(0)
    y_onehot = F.one_hot(y_true, num_classes=y_hat.size(1)).float()
    log_probs = torch.log(y_hat + 1e-9)
    loss = -(y_onehot * log_probs).sum(dim=1).mean()
    return loss

# ==========================================================
# 2. FeedForward Network (No autograd)
# ==========================================================

class ManualFeedForwardNet:
    def __init__(self, input_size, hidden1, hidden2, output_size):
        # Xavier/He initialization
        self.W1 = torch.randn(input_size, hidden1) * (2 / input_size) ** 0.5
        self.b1 = torch.zeros(hidden1)
        self.W2 = torch.randn(hidden1, hidden2) * (2 / hidden1) ** 0.5
        self.b2 = torch.zeros(hidden2)
        self.W3 = torch.randn(hidden2, output_size) * (1 / hidden2) ** 0.5
        self.b3 = torch.zeros(output_size)

        # Move to same device later
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def to(self, device):
        self.W1 = self.W1.to(device)
        self.b1 = self.b1.to(device)
        self.W2 = self.W2.to(device)
        self.b2 = self.b2.to(device)
        self.W3 = self.W3.to(device)
        self.b3 = self.b3.to(device)
        self.device = device

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.y_hat = softmax(self.Z3)
        return self.y_hat

    def backward(self, y_true):
        # Manual gradient computation using chain rule
        batch_size = y_true.size(0)
        y_onehot = F.one_hot(y_true, num_classes=self.y_hat.size(1)).float()

        dZ3 = (self.y_hat - y_onehot) / batch_size
        dW3 = self.A2.T @ dZ3
        db3 = dZ3.sum(dim=0)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(dim=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(dim=0)

        return dW1, db1, dW2, db2, dW3, db3

    def step(self, grads, lr):
        dW1, db1, dW2, db2, dW3, db3 = grads
        with torch.no_grad():
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W3 -= lr * dW3
            self.b3 -= lr * db3

# ==========================================================
# 3. Training and Validation Functions
# ==========================================================

def train_epoch(model, dataloader, lr):
    total_loss, correct = 0.0, 0
    model.W1.requires_grad = False  # ensure autograd disabled
    for X, y in dataloader:
        X, y = X.to(model.device), y.to(model.device)
        X = X.view(-1, 784)

        # Forward
        y_hat = model.forward(X)

        # Loss
        loss = cross_entropy_loss(y_hat, y)
        total_loss += loss.item() * X.size(0)

        # Backward
        grads = model.backward(y)

        # Update
        model.step(grads, lr)

        # Accuracy
        preds = y_hat.argmax(1)
        correct += (preds == y).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    acc = correct / len(dataloader.dataset)
    return avg_loss, acc


def validate_epoch(model, dataloader):
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model.device), y.to(model.device)
            X = X.view(-1, 784)
            y_hat = model.forward(X)
            loss = cross_entropy_loss(y_hat, y)
            total_loss += loss.item() * X.size(0)
            preds = y_hat.argmax(1)
            correct += (preds == y).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    acc = correct / len(dataloader.dataset)
    return avg_loss, acc

# ==========================================================
# 4. Main Execution
# ==========================================================

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 784
HIDDEN1 = 512
HIDDEN2 = 128
OUTPUT = 10
LR = 0.01
BATCH_SIZE = 64
EPOCHS = 10

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_size, val_size = 50000, 10000
train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = ManualFeedForwardNet(INPUT_SIZE, HIDDEN1, HIDDEN2, OUTPUT)

train_losses, val_losses, train_accs, val_accs = [], [], [], []

print(f"Training manually on {device}...\n")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, LR)
    val_loss, val_acc = validate_epoch(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

print("\nTraining complete.")
