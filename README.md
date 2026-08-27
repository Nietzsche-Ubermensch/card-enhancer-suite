# Card Enhancer Suite

Bulk enhancement and restoration pipeline for trading-card images.

## Components

| Package | Purpose |
|---------|---------|
| `gigapixel` | Windows automation for Topaz Gigapixel AI (upscaling) |
| `elan` | PyTorch super-resolution training (alternative upscaler) |
| `card_enhancer` | Quality scoring, metadata extraction, CLI orchestration |

## Installation

```bash
pip install -e .
```

With the ELAN PyTorch backend:

```bash
pip install -e ".[torch]"
```

On Windows, add the Gigapixel automation dependencies:

```bash
pip install -e ".[torch,gigapixel]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Bulk Enhancement (Gigapixel)

```bash
gigapixel-batch \
  --backend gigapixel \
  --exe "C:\Program Files\Topaz Labs LLC\Topaz Gigapixel AI\Topaz Gigapixel AI.exe" \
  --input "C:\cards\to_process" \
  --output "C:\cards\enhanced" \
  --scale X2 \
  --mode STANDARD \
  --pattern "*.jpg" \
  --log enhancement_log.jsonl
```

## Bulk Enhancement (ELAN)

```bash
gigapixel-batch \
  --backend elan \
  --checkpoint checkpoints/elan_x2.pt \
  --input ./cards \
  --output ./enhanced \
  --resume
```

`--resume` skips every input already marked successful in the JSONL log.
If the checkpoint was trained with a non-default architecture, pass
`--channels` / `--blocks` / `--heads` so the model matches it.

## ELAN Training

```bash
elan-train --config configs/elan_x2.yml
```

Ground truth images go in `dataroot`; low-resolution inputs are synthesized
with bicubic downsampling. Validation PSNR is reported on each folder in
`valid_dataroots`, and checkpoints land in `log_path/<config-name>/`.

## Quality Scoring

```python
from card_enhancer import quality_score_from_metadata

metadata = {
    "subjectName": "Alex Windsor",
    "cardNumber": "3",
    "manufacturer": "Upper Deck",
    "year": 2026,
    "stats": {"height": "5'5\"", "from": "Norwich, England"},
}
score = quality_score_from_metadata(metadata)  # → 2.0–2.5 range
```

Or from the CLI:

```bash
card-enhance alex_windsor.jpg
```

## Repository Layout

```
gigapixel/     # Topaz Gigapixel AI GUI automation (pywinauto, Windows)
elan/          # ELAN super-resolution: model, datasets, trainer
card_enhancer/ # truth-reward quality scoring + CLI
configs/       # YAML training configs
tests/         # pytest suite
docs/          # project landing page
```

## Tooling

- `tox -e mypy` — type checking
- `tox -e flake8` — linting
- `pytest` — tests

## License

MIT — see [LICENSE](LICENSE).
