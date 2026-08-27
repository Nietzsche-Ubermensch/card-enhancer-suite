"""
ELAN training entrypoint:  elan-train --config configs/elan_x2.yml

Config keys (see configs/elan_x2.yml):
    model            elan | elan_light
    scale            upscale factor (power of two)
    gt_size          ground-truth crop size (LR crop = gt_size / scale)
    batch_size       per-step batch
    epochs           total epochs
    lr               Adam learning rate
    decays           epoch milestones where lr *= gamma
    gamma            decay factor
    log_every        iterations between training logs
    test_every       epochs between validation runs
    num_workers      dataloader workers
    threads          torch CPU threads
    gpu_ids          CUDA device ids (first is used); empty -> CPU
    dataroot         folder of ground-truth training images
    valid_dataroots  mapping of name -> folder of ground-truth validation images
    use_hflip        horizontal-flip augmentation
    use_rot          90-degree rotation augmentation
    log_path         experiment output root (checkpoints + logs)
    channels         (optional) model width   — default 60 (elan) / 36 (elan_light)
    n_blocks         (optional) ELAB blocks   — default 24 (elan) / 12 (elan_light)
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from .data import CardSRDataset, CardSRValidDataset
from .model import ELAN, ELANLight

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RGB PSNR for tensors in [0, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def build_model(cfg: dict) -> torch.nn.Module:
    variant = cfg.get("model", "elan")
    scale = int(cfg.get("scale", 2))
    extra = {
        k: int(cfg[k]) for k in ("num_heads", "n_groups") if k in cfg
    }
    if variant == "elan_light":
        channels = int(cfg.get("channels", 36))
        n_blocks = int(cfg.get("n_blocks", 12))
        return ELANLight(upscale=scale, channels=channels, n_blocks=n_blocks, **extra)
    channels = int(cfg.get("channels", 60))
    n_blocks = int(cfg.get("n_blocks", 24))
    return ELAN(upscale=scale, channels=channels, n_blocks=n_blocks, **extra)


@torch.no_grad()
def validate(model: torch.nn.Module, root: Path, scale: int, device: str) -> float:
    """Mean RGB PSNR over a folder of ground-truth images."""
    ds = CardSRValidDataset(root, scale=scale)
    scores = []
    for lr, gt, _ in ds:
        lr = lr.unsqueeze(0).to(device)
        sr = model(lr).squeeze(0).clamp(0, 1).cpu()
        scores.append(psnr(sr, gt))
    return sum(scores) / max(1, len(scores))


def train(cfg: dict, resume: Optional[Path] = None) -> Path:
    scale = int(cfg.get("scale", 2))
    device = (
        f"cuda:{cfg['gpu_ids'][0]}"
        if cfg.get("gpu_ids") and torch.cuda.is_available()
        else "cpu"
    )
    torch.set_num_threads(int(cfg.get("threads", 1)))

    experiment = Path(cfg.get("log_path", "experiments")) / str(
        cfg.get("experiment_name", f"elan_x{scale}")
    )
    experiment.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ELAN on {device} — {n_params/1e6:.2f}M parameters")

    train_ds = CardSRDataset(
        Path(cfg["dataroot"]),
        scale=scale,
        gt_size=int(cfg.get("gt_size", 256)),
        use_hflip=bool(cfg.get("use_hflip", True)),
        use_rot=bool(cfg.get("use_rot", True)),
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 2e-4)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(m) for m in cfg.get("decays", [200, 400, 600, 800])],
        gamma=float(cfg.get("gamma", 0.5)),
    )
    loss_fn = torch.nn.L1Loss()

    start_epoch, best_psnr = 1, 0.0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("best_psnr", 0.0))
        logger.info(f"Resumed from {resume} at epoch {start_epoch}")

    epochs = int(cfg.get("epochs", 1000))
    log_every = int(cfg.get("log_every", 100))
    test_every = int(cfg.get("test_every", 10))
    valid_roots: Dict[str, str] = dict(cfg.get("valid_dataroots", {}))

    last_path = experiment / "checkpoint_last.pt"
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        for i, (lr, gt) in enumerate(loader, start=1):
            lr, gt = lr.to(device), gt.to(device)
            sr = model(lr)
            loss = loss_fn(sr, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % log_every == 0:
                logger.info(
                    f"epoch {epoch}/{epochs} iter {i}/{len(loader)} "
                    f"l1={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
        scheduler.step()

        if epoch % test_every == 0 and valid_roots:
            model.eval()
            for name, root in valid_roots.items():
                score = validate(model, Path(root), scale, device)
                logger.info(f"epoch {epoch} valid[{name}] PSNR={score:.2f} dB")
                if score > best_psnr:
                    best_psnr = score
                    torch.save(
                        {"model_state_dict": model.state_dict(), "epoch": epoch,
                         "best_psnr": best_psnr},
                        experiment / "checkpoint_best.pt",
                    )

        torch.save(
            {"model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "epoch": epoch, "best_psnr": best_psnr},
            last_path,
        )
        logger.info(f"epoch {epoch} done in {time.time() - t0:.1f}s — checkpoint saved")

    logger.success(f"Training finished. Best PSNR: {best_psnr:.2f} dB → {experiment}")
    return last_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ELAN for card super-resolution")
    parser.add_argument("--config", type=Path, required=True, help="YAML training config")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("experiment_name", Path(args.config).stem)

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
