from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from .data import build_data_bundle
from .model import UNet, count_parameters
from .losses import build_loss
from .engine import fit_model, run_epoch
from .visualize import save_prediction_grid, save_training_curves

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["project"]["output_dir"])
    results_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    checkpoints_dir = root / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(cfg["training"]["device"])
    print(f"Device: {device}")
    all_results = []

    for exp in cfg["experiments"]:
        name = exp["name"]
        seed = int(exp.get("seed", cfg["project"]["seed"]))
        set_seed(seed)
        print(f"\n===== Experiment: {name} =====")
        data = build_data_bundle(
            root=cfg["data"]["root"],
            image_size=cfg["data"]["image_size"],
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            seed=cfg["project"]["seed"],
            augment_train=exp["augmentation"],
            train_fraction=cfg["data"]["train_fraction"],
            val_fraction=cfg["data"]["val_fraction"],
            test_fraction=cfg["data"]["test_fraction"],
        )
        model = UNet(cfg["model"]["in_channels"], cfg["model"]["num_classes"],
                     cfg["model"]["base_channels"]).to(device)
        criterion = build_loss(exp["loss"], cfg["model"]["num_classes"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"]
        )
        checkpoint = checkpoints_dir / f"{name}.pt"
        history_path = results_dir / f"{name}_history.csv"

        fit_info = fit_model(
            model, data.train_loader, data.val_loader, criterion, optimizer, device,
            cfg["training"]["epochs"], cfg["training"]["patience"],
            cfg["model"]["num_classes"], data.class_names, checkpoint, history_path
        )

        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model_state_dict"])
        test_metrics = run_epoch(model, data.test_loader, criterion, device, None,
                                 cfg["model"]["num_classes"], data.class_names)
        row = {
            "experiment": name,
            "augmentation": exp["augmentation"],
            "loss": exp["loss"],
            "seed": seed,
            "best_epoch": fit_info["best_epoch"],
            "training_seconds": round(fit_info["training_seconds"], 2),
            "trainable_parameters": count_parameters(model),
        }
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        all_results.append(row)
        save_prediction_grid(model, data.test_loader, device,
                             figures_dir / f"{name}_predictions.png")
        save_training_curves(history_path, figures_dir / f"{name}_curves.png")
        print("Test metrics:", json.dumps(row, indent=2))

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(results_dir / "experiment_comparison.csv", index=False)
    (results_dir / "experiment_comparison.json").write_text(
        result_df.to_json(orient="records", indent=2)
    )
    print("\nSaved measured results to results/tables/experiment_comparison.csv")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()
