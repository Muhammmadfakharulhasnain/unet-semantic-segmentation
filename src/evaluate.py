import argparse
from pathlib import Path
import torch, yaml, pandas as pd
from .data import build_data_bundle
from .model import UNet
from .losses import build_loss
from .engine import run_epoch
from .run_experiments import choose_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--loss", default="combined")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = choose_device(cfg["training"]["device"])
    data = build_data_bundle(
        root=cfg["data"]["root"], image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"],
        seed=cfg["project"]["seed"], augment_train=False,
        train_fraction=cfg["data"]["train_fraction"],
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"])
    model = UNet(cfg["model"]["in_channels"], cfg["model"]["num_classes"],
                 cfg["model"]["base_channels"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    criterion = build_loss(args.loss, cfg["model"]["num_classes"])
    metrics = run_epoch(model, data.test_loader, criterion, device, None,
                        cfg["model"]["num_classes"], data.class_names)
    print(pd.Series(metrics).to_string())

if __name__ == "__main__":
    main()
