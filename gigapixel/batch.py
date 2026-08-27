"""
Bulk enhancement and restoration for trading-card images.
Supports both Topaz Gigapixel AI (Windows GUI automation)
and ELAN PyTorch model (cross-platform inference).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Literal, Optional, Protocol

from loguru import logger
from tqdm import tqdm

from . import Gigapixel, Mode, Scale
from .exceptions import GigapixelException


class Upscaler(Protocol):
    """Protocol for any upscaling backend."""

    def process(self, input_path: Path, output_path: Path) -> None:
        ...


class GigapixelUpscaler:
    """Wraps Gigapixel so it matches the Upscaler protocol."""

    def __init__(
        self,
        exe_path: Path,
        scale: Optional[Scale] = None,
        mode: Optional[Mode] = None,
        timeout: int = 900,
    ):
        self.gp = Gigapixel(exe_path, processing_timeout=timeout)
        self.scale = scale
        self.mode = mode

    def process(self, input_path: Path, output_path: Path) -> None:
        # Gigapixel saves in-place or next to source; we accept the limitation
        self.gp.process(input_path, scale=self.scale, mode=self.mode)


class ELANUpscaler:
    """PyTorch ELAN backend for headless / cross-platform upscaling."""

    def __init__(
        self,
        checkpoint: Path,
        variant: Literal["elan", "elan_light"] = "elan",
        scale: int = 2,
        device: str = "cuda",
        channels: Optional[int] = None,
        n_blocks: Optional[int] = None,
        num_heads: Optional[int] = None,
    ):
        import torch
        from elan.model import ELAN, ELANLight

        self.device = device if torch.cuda.is_available() else "cpu"
        # Architecture must match the checkpoint; fall back to paper defaults.
        arch = {}
        if channels is not None:
            arch["channels"] = channels
        if n_blocks is not None:
            arch["n_blocks"] = n_blocks
        if num_heads is not None:
            arch["num_heads"] = num_heads
        if variant == "elan_light":
            model = ELANLight(upscale=scale, **arch)
        else:
            model = ELAN(upscale=scale, **arch)

        ckpt = torch.load(checkpoint, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.scale = scale

    def process(self, input_path: Path, output_path: Path) -> None:
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF

        img = Image.open(input_path).convert("RGB")
        x = TF.to_tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sr = self.model(x)

        sr = sr.squeeze(0).clamp(0, 1)
        sr_img = TF.to_pil_image(sr.cpu())
        sr_img.save(output_path)


def process_single(
    upscaler: Upscaler,
    photo_path: Path,
    output_dir: Path,
    output_log: Optional[Path] = None,
    suffix: str = "_enhanced",
) -> dict:
    """Process one image and return a status dict."""
    output_path = output_dir / f"{photo_path.stem}{suffix}{photo_path.suffix}"
    result = {
        "input": str(photo_path),
        "output": str(output_path),
        "success": False,
        "error": None,
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        upscaler.process(photo_path, output_path)
        result["success"] = True
        logger.success(f"Processed: {photo_path.name} -> {output_path.name}")
    except GigapixelException as exc:
        result["error"] = str(exc)
        logger.error(f"Failed {photo_path.name}: {exc}")
    except Exception as exc:
        result["error"] = f"Unexpected: {exc}"
        logger.exception(f"Crash on {photo_path.name}")

    if output_log:
        output_log.parent.mkdir(parents=True, exist_ok=True)
        with open(output_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    return result


def process_directory(
    upscaler: Upscaler,
    directory: Path,
    output_dir: Path,
    pattern: str = "*.jpg",
    output_log: Optional[Path] = None,
    suffix: str = "_enhanced",
) -> List[dict]:
    """Bulk-process every image matching *pattern* inside *directory*."""
    files = sorted(directory.glob(pattern))
    if not files:
        logger.warning(f"No files matched '{pattern}' in {directory}")
        return []

    logger.info(f"Bulk processing {len(files)} images from {directory}")
    results: List[dict] = []
    for photo in tqdm(files, desc="Enhancing", ncols=80):
        results.append(process_single(upscaler, photo, output_dir, output_log, suffix))
    return results


def load_completed_inputs(log_path: Path) -> set:
    """Read a JSONL log and return the set of already-processed input paths."""
    completed: set = set()
    if not log_path.exists():
        return completed
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("success"):
                    completed.add(entry["input"])
            except json.JSONDecodeError:
                continue
    return completed


def process_directory_resume(
    upscaler: Upscaler,
    directory: Path,
    output_dir: Path,
    pattern: str = "*.jpg",
    output_log: Optional[Path] = None,
    suffix: str = "_enhanced",
) -> List[dict]:
    """Bulk-process with resume support (skips already-successful items in log)."""
    files = sorted(directory.glob(pattern))
    if not files:
        logger.warning(f"No files matched '{pattern}' in {directory}")
        return []

    completed = load_completed_inputs(output_log) if output_log else set()
    pending = [f for f in files if str(f) not in completed]

    if completed:
        logger.info(f"Resuming: {len(completed)}/{len(files)} already done, {len(pending)} pending")

    results: List[dict] = []
    for photo in tqdm(pending, desc="Enhancing", ncols=80):
        results.append(process_single(upscaler, photo, output_dir, output_log, suffix))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk enhance trading-card images"
    )
    parser.add_argument(
        "--backend",
        choices=["gigapixel", "elan"],
        required=True,
        help="Upscaling backend to use",
    )
    # Gigapixel args
    parser.add_argument(
        "--exe",
        help=(
            "Path to Topaz Gigapixel AI.exe "
            '(e.g. "C:\\Program Files\\...\\Topaz Gigapixel AI.exe")'
        ),
    )
    parser.add_argument(
        "--scale",
        choices=[s.name for s in Scale],
        default="X2",
        help="Gigapixel upscaling factor",
    )
    parser.add_argument(
        "--mode",
        choices=[m.name for m in Mode],
        default="STANDARD",
        help="Gigapixel processing mode",
    )
    # ELAN args
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to ELAN .pt checkpoint",
    )
    parser.add_argument(
        "--variant",
        choices=["elan", "elan_light"],
        default="elan",
        help="ELAN model variant",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for ELAN inference",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="ELAN model width (must match checkpoint; default: paper size)",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="ELAN block count (must match checkpoint; default: paper size)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help="ELAN attention heads (must match checkpoint; default: paper size)",
    )
    # Common args
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing card images",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for enhanced images",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern for input images",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("enhancement_log.jsonl"),
        help="JSON Lines log path",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-image processing timeout (Gigapixel only)",
    )
    parser.add_argument(
        "--suffix",
        default="_enhanced",
        help="Suffix appended to output filenames",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-successful items found in the log",
    )
    args = parser.parse_args()

    # Build backend
    if args.backend == "gigapixel":
        if not args.exe:
            parser.error("--exe is required when using Gigapixel backend")
        upscaler: Upscaler = GigapixelUpscaler(
            exe_path=Path(args.exe),
            scale=Scale[args.scale],
            mode=Mode[args.mode],
            timeout=args.timeout,
        )
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required when using ELAN backend")
        upscaler = ELANUpscaler(
            checkpoint=args.checkpoint,
            variant=args.variant,
            scale=int(args.scale[-1]) if args.scale.startswith("X") else 2,
            device=args.device,
            channels=args.channels,
            n_blocks=args.blocks,
            num_heads=args.heads,
        )

    processor = process_directory_resume if args.resume else process_directory
    results = processor(
        upscaler,
        args.input,
        args.output,
        pattern=args.pattern,
        output_log=args.log,
        suffix=args.suffix,
    )

    success = sum(1 for r in results if r.get("success"))
    total = len(results)
    logger.info(f"Done: {success}/{total} succeeded in this run")
    sys.exit(0 if success == total else 1)


if __name__ == "__main__":
    main()
