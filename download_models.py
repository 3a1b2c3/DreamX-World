"""Download Wan2.2-TI2V-5B and DreamX-World-5B-Cam into the HuggingFace cache.

Models are stored in the shared HF hub cache (~/.cache/huggingface/hub) instead of
being copied into the project folder, so they're deduplicated and reusable across
projects. The resolved snapshot paths are printed at the end.

Usage:
    python download_models.py                 # both -> HF cache
    python download_models.py --only wan      # only Wan base
    python download_models.py --only dreamx   # only DreamX model
    python download_models.py --cache_dir D:/hf  # override HF cache location
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download


WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
DREAMX_REPO = "GD-ML/DreamX-World-5B-Cam"


def download(repo_id: str, cache_dir: str | None) -> str:
    print(f"[download] {repo_id} -> HF cache")
    path = snapshot_download(repo_id=repo_id, resume_download=True, cache_dir=cache_dir)
    print(f"[done] {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", default=None, help="Override HF cache dir (default ~/.cache/huggingface/hub)")
    parser.add_argument("--only", choices=["wan", "dreamx"], default=None)
    args = parser.parse_args()

    repos = []
    if args.only in (None, "wan"):
        repos.append(WAN_REPO)
    if args.only in (None, "dreamx"):
        repos.append(DREAMX_REPO)

    paths = {repo: download(repo, args.cache_dir) for repo in repos}

    print("\nSummary (HF cache snapshot paths):")
    for repo, path in paths.items():
        print(f"  {repo} -> {path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    sys.exit(main())
