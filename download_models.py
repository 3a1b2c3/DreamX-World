"""Download Wan2.2-TI2V-5B and DreamX-World-5B-Cam checkpoints from Hugging Face.

Usage:
    python download_models.py                # downloads both to ./Wan2.2-TI2V-5B and ./DreamX-World-5B-Cam
    python download_models.py --only wan     # download only Wan base model
    python download_models.py --only dreamx  # download only DreamX adapter
    python download_models.py --root D:/ckpts # change destination root
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
DREAMX_REPO = "GD-ML/DreamX-World-5B-Cam"


def download(repo_id: str, dest: Path) -> None:
    print(f"[download] {repo_id} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[done] {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--only", choices=["wan", "dreamx"], default=None)
    args = parser.parse_args()

    targets = []
    if args.only in (None, "wan"):
        targets.append((WAN_REPO, args.root / "Wan2.2-TI2V-5B"))
    if args.only in (None, "dreamx"):
        targets.append((DREAMX_REPO, args.root / "DreamX-World-5B-Cam"))

    for repo, dest in targets:
        download(repo, dest)

    print("\nSummary:")
    for repo, dest in targets:
        print(f"  {repo} -> {dest}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    sys.exit(main())
