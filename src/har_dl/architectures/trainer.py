import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class HARDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epochs: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

    def _train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        self.scheduler.step()
        return total_loss / total, correct / total

    def evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                preds = logits.argmax(dim=1)
                total_loss += loss.item() * x.size(0)
                correct += (preds == y).sum().item()
                total += x.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        return (
            total_loss / total,
            correct / total,
            np.concatenate(all_preds),
            np.concatenate(all_targets),
        )

    def fit(self, train_loader: DataLoader, save_dir: Path) -> list[dict]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        best_state = None
        history = []

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_acc": round(train_acc, 6),
                    "lr": self.scheduler.get_last_lr()[0],
                }
            )

            if train_loss < best_loss:
                best_loss = train_loss
                best_state = copy.deepcopy(self.model.state_dict())

            if epoch % 10 == 0 or epoch == self.epochs:
                print(
                    f"    epoch {epoch:>3}/{self.epochs}"
                    f"  loss={train_loss:.4f}"
                    f"  acc={train_acc:.4f}"
                    f"  lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

        torch.save(best_state, save_dir / "best_model.pt")
        torch.save(self.model.state_dict(), save_dir / "last_model.pt")
        self.model.load_state_dict(best_state)
        return history