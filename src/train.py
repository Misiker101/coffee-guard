"""
Training script for CoffeeGuard.

Usage:
    python src/train.py --data-dir data/coffee-leaves --epochs 10 --batch-size 32

Logs params/metrics/model artifacts to a local MLflow tracking server.
Run `mlflow ui` in another terminal to view the dashboard at
http://127.0.0.1:5000
"""

import argparse
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model, CLASS_NAMES


def get_dataloaders(data_dir: str, batch_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/coffee-leaves")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/coffeeguard.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    # Phase 1: freeze backbone, train head only
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    mlflow.set_experiment("coffeeguard")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        best_acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )
            print(f"epoch {epoch+1}/{args.epochs} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save(model.state_dict(), args.out)

        mlflow.log_metric("best_val_acc", best_acc)

        # A real example input, so MLflow can trace the forward pass
        # (required for its default serialization mode as of newer
        # mlflow/torch versions).
        example_x, _ = next(iter(val_loader))
        example_input = example_x[:1].cpu().numpy()

        model.to("cpu")  # log a CPU copy so the artifact isn't tied to this GPU
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input,
        )
        model.to(device)

        print(f"Best val acc: {best_acc:.4f}. Saved to {args.out}")


if __name__ == "__main__":
    main()
