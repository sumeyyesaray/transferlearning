"""
CIFAR-10 üzerinde transfer learning karşılaştırması: ResNet-50 vs EfficientNet-B3 vs ConvNeXt-T

Kullanım:
    python train.py --model resnet50 --epochs 25
    python train.py --model efficientnet_b3 --epochs 25
    python train.py --model convnext_tiny --epochs 25

Her model için sonuçlar results/<model_adi>.json dosyasına yazılır,
en iyi checkpoint checkpoints/<model_adi>_best.pt olarak kaydedilir.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import wandb

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 10


def build_model(name: str) -> nn.Module:
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Bilinmeyen model: {name}")
    return model


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=16, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet50", "efficientnet_b3", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="backbone-comparison")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device} | Model: {args.model} | Epoch: {args.epochs}")

    Path(args.output_dir, "results").mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.model,
        mode=args.wandb_mode,
        config=vars(args),
    )

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))
    wandb.config.update({"num_params": sum(p.numel() for p in model.parameters())})

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time_sec": []}
    best_val_acc = 0.0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(epoch_time)

        print(f"[{args.model}] Epoch {epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({epoch_time:.1f}s)")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(args.output_dir, "checkpoints", f"{args.model}_best.pt"))

    total_time = time.time() - total_start

    result = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "total_train_time_sec": total_time,
        "num_params": sum(p.numel() for p in model.parameters()),
        "history": history,
    }

    result_path = Path(args.output_dir, "results", f"{args.model}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["final_val_acc"] = history["val_acc"][-1]
    wandb.summary["total_train_time_sec"] = total_time
    wandb.summary["num_params"] = result["num_params"]
    wandb.finish()

    print(f"\nTamamlandı. En iyi val_acc: {best_val_acc:.4f} | Sonuç dosyası: {result_path}")


if __name__ == "__main__":
    main()
