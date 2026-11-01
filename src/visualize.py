from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
CLASS_COLORS = np.array([[220, 60, 60], [60, 150, 80], [240, 190, 40]], dtype=np.uint8)

def denormalize(image_tensor):
    arr = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * STD + MEAN).clip(0, 1)
    return arr

def colorize(mask):
    mask = np.asarray(mask, dtype=np.int64)
    return CLASS_COLORS[mask]

@torch.no_grad()
def save_prediction_grid(model, loader, device, output_path, max_items=6):
    model.eval()
    images, masks = next(iter(loader))
    images = images[:max_items].to(device)
    masks = masks[:max_items]
    preds = model(images).argmax(dim=1).cpu()

    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3.2*n), squeeze=False)
    for i in range(n):
        original = denormalize(images[i])
        gt = masks[i].numpy()
        pred = preds[i].numpy()
        error = (gt != pred).astype(np.uint8)
        axes[i, 0].imshow(original); axes[i, 0].set_title("Input")
        axes[i, 1].imshow(colorize(gt)); axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(colorize(pred)); axes[i, 2].set_title("Prediction")
        axes[i, 3].imshow(error, cmap="gray", vmin=0, vmax=1); axes[i, 3].set_title("Pixel error")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_training_curves(history_csv, output_path):
    import pandas as pd
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(df["epoch"], df["train_mean_iou"], label="Train")
    axes[1].plot(df["epoch"], df["val_mean_iou"], label="Validation")
    axes[1].set_title("Mean IoU"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
