from __future__ import annotations
import csv, json, time
from pathlib import Path
import torch
from tqdm import tqdm
from .metrics import update_confusion_matrix, metrics_from_confusion_matrix

def run_epoch(model, loader, criterion, device, optimizer=None, num_classes=3, class_names=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_images += batch_size
            update_confusion_matrix(confmat, logits.detach(), masks, num_classes)

    metrics = metrics_from_confusion_matrix(confmat.cpu(), class_names)
    metrics["loss"] = total_loss / max(total_images, 1)
    return metrics

def fit_model(model, train_loader, val_loader, criterion, optimizer, device,
              epochs, patience, num_classes, class_names, checkpoint_path,
              history_path):
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer,
                                  num_classes, class_names)
        val_metrics = run_epoch(model, val_loader, criterion, device, None,
                                num_classes, class_names)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(f"Epoch {epoch}/{epochs} | train loss={train_metrics['loss']:.4f} "
              f"| val loss={val_metrics['loss']:.4f} "
              f"| val mIoU={val_metrics['mean_iou']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "best_val_loss": best_val_loss}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    elapsed = time.perf_counter() - start
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss,
            "training_seconds": elapsed, "history": history}
