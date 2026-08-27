import json
from pathlib import Path

from gigapixel.batch import load_completed_inputs, process_directory, process_directory_resume


class DummyUpscaler:
    """Writes the output file immediately, like a real backend would."""

    def process(self, input_path: Path, output_path: Path) -> None:
        output_path.write_bytes(b"enhanced")


def _make_images(directory: Path, names) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"fake-jpg")


def test_load_completed_inputs(tmp_path):
    log = tmp_path / "log.jsonl"
    entries = [
        {"input": "a.jpg", "success": True},
        {"input": "b.jpg", "success": False, "error": "boom"},
        {"input": "c.jpg", "success": True},
    ]
    log.write_text("\n".join(json.dumps(e) for e in entries))
    assert load_completed_inputs(log) == {"a.jpg", "c.jpg"}
    assert load_completed_inputs(tmp_path / "missing.jsonl") == set()


def test_process_directory(tmp_path):
    src = tmp_path / "cards"
    out = tmp_path / "enhanced"
    _make_images(src, ["a.jpg", "b.jpg", "c.jpg"])
    log = tmp_path / "log.jsonl"

    results = process_directory(DummyUpscaler(), src, out, output_log=log)
    assert len(results) == 3
    assert all(r["success"] for r in results)
    assert (out / "a_enhanced.jpg").exists()
    assert len(log.read_text().strip().splitlines()) == 3


def test_process_directory_resume_skips_completed(tmp_path):
    src = tmp_path / "cards"
    out = tmp_path / "enhanced"
    _make_images(src, ["a.jpg", "b.jpg", "c.jpg"])
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"input": str(src / "a.jpg"), "success": True}) + "\n")

    results = process_directory_resume(DummyUpscaler(), src, out, output_log=log)
    assert len(results) == 2
    assert {Path(r["input"]).name for r in results} == {"b.jpg", "c.jpg"}


def test_process_directory_empty(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    assert process_directory(DummyUpscaler(), src, tmp_path / "out") == []
