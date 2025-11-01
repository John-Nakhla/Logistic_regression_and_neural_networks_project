import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 784
HIDDEN1 = 512
HIDDEN2 = 128
OUTPUT = 10
LR = 0.01
BATCH_SIZE = 64
PATIENCE = 3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_ds, val_ds = random_split(full_train_dataset, [50000, 10000])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

class ManualFeedForwardNet:
    def __init__(self, input_size, h1, h2, output_size):
        self.W1 = torch.randn(input_size, h1) * (2. / input_size) ** 0.5
        self.b1 = torch.zeros(h1)
        self.W2 = torch.randn(h1, h2) * (2. / h1) ** 0.5
        self.b2 = torch.zeros(h2)
        self.W3 = torch.randn(h2, output_size) * (2. / h2) ** 0.5
        self.b3 = torch.zeros(output_size)

    def relu(self, x):
        return torch.maximum(x, torch.tensor(0.0))

    def softmax(self, x):
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        return exp_x / exp_x.sum(dim=1, keepdim=True)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = self.softmax(self.z3)
        return self.out

    def compute_loss(self, y_pred, y_true):
        y_onehot = torch.zeros_like(y_pred)
        y_onehot.scatter_(1, y_true.view(-1, 1), 1)
        loss = -torch.mean(torch.sum(y_onehot * torch.log(y_pred + 1e-9), dim=1))
        return loss

    def manual_backward(self, y_true, lr):
        y_onehot = torch.zeros_like(self.out)
        y_onehot.scatter_(1, y_true.view(-1, 1), 1)
        delta3 = (self.out - y_onehot) / y_true.size(0)
        dW3 = self.a2.T @ delta3
        db3 = delta3.sum(dim=0)
        delta2 = (delta3 @ self.W3.T) * (self.z2 > 0)
        dW2 = self.a1.T @ delta2
        db2 = delta2.sum(dim=0)
        delta1 = (delta2 @ self.W2.T) * (self.z1 > 0)
        dW1 = self.x.T @ delta1
        db1 = delta1.sum(dim=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def validate_epoch(model, loader):
    model.eval = True
    total_loss, total_acc, count = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1)
            preds = model.forward(x)
            total_loss += model.compute_loss(preds, y).item() * x.size(0)
            total_acc += accuracy(preds, y) * x.size(0)
            count += x.size(0)
    return total_loss / count, total_acc / count

def train_manual(model, train_loader, val_loader, lr, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    epoch = 0
    while True:
        model.eval = False
        train_loss, train_acc, total = 0, 0, 0
        for x, y in train_loader:
            x = x.view(x.size(0), -1)
            preds = model.forward(x)
            loss = model.compute_loss(preds, y)
            model.manual_backward(y, lr)
            train_loss += loss.item() * x.size(0)
            train_acc += accuracy(preds, y) * x.size(0)
            total += x.size(0)
        train_loss /= total
        train_acc /= total
        val_loss, val_acc = validate_epoch(model, val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model, "best_manual_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        epoch += 1

model = ManualFeedForwardNet(INPUT_SIZE, HIDDEN1, HIDDEN2, OUTPUT)
print(f"Training manually on {device}...\n")
train_manual(model, train_loader, val_loader, LR, PATIENCE)
