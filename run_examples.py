"""Single-GPU runner for DreamX-World-5B-Cam (Windows/Linux).

Iterates over every entry in configs/dreamx/eval.json by handing the JSON
file straight to inference_dreamx5b.py (which already loops internally) and
writes mp4s to ./outputs/.

Usage:
    python run_examples.py
    python run_examples.py --steps 30 --seed 7
    python run_examples.py --memory_mode model_cpu_offload   # if 32GB OOMs
    python run_examples.py --indices 0 3 5                   # subset of examples
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent

WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
DREAMX_REPO = "GD-ML/DreamX-World-5B-Cam"


def resolve_model(path: Path | None, repo_id: str) -> Path:
    """Use the given path if it exists locally, else resolve from the HF cache
    (downloads if missing)."""
    if path is not None and Path(path).exists():
        return Path(path)
    return Path(snapshot_download(repo_id=repo_id, resume_download=True))


def build_cmd(args: argparse.Namespace, input_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(HERE / "inference_dreamx5b.py"),
        "--config_path",         str(HERE / "configs" / "wan2.2" / "wan_ti2v_5b.yaml"),
        "--model_name",          str(args.model_name),
        "--transformer_path",    str(args.transformer_path),
        "--input_dir",           str(input_json),
        "--output_dir",          str(args.output_dir),
        "--cam_method",          args.cam_method,
        "--add_control_adapter",
        "--sample_size",         str(args.height), str(args.width),
        "--video_length",        str(args.video_length),
        "--fps",                 str(args.fps),
        "--guidance_scale",      str(args.guidance_scale),
        "--num_inference_steps", str(args.steps),
        "--seed",                str(args.seed),
        "--weight_dtype",        args.weight_dtype,
        "--ulysses_degree",      "1",
        "--ring_degree",         "1",
    ]
    if args.memory_mode:
        cmd += ["--GPU_memory_mode", args.memory_mode]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name",       type=Path, default=None,
                        help="Wan base path; default resolves from the HF cache")
    parser.add_argument("--transformer_path", type=Path, default=None,
                        help="DreamX model path; default resolves from the HF cache")
    parser.add_argument("--input_json",       type=Path, default=HERE / "configs" / "dreamx" / "eval.json")
    parser.add_argument("--output_dir",       type=Path, default=HERE / "outputs")
    parser.add_argument("--indices",          type=int, nargs="+", default=None,
                        help="Run only these eval.json indices (0-based)")
    parser.add_argument("--height",           type=int, default=704)
    parser.add_argument("--width",            type=int, default=1280)
    parser.add_argument("--video_length",     type=int, default=121)
    parser.add_argument("--fps",              type=int, default=24)
    parser.add_argument("--guidance_scale",   type=float, default=3.0)
    parser.add_argument("--steps",            type=int, default=50)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--weight_dtype",     default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cam_method",       default="prope", choices=["prope", "plucker"])
    parser.add_argument("--memory_mode",      default=None,
                        choices=[None, "model_full_load", "model_full_load_and_qfloat8",
                                 "model_cpu_offload", "model_cpu_offload_and_qfloat8",
                                 "sequential_cpu_offload"])
    args = parser.parse_args()

    # Resolve model paths from the HF cache (downloads if missing).
    args.model_name = resolve_model(args.model_name, WAN_REPO)
    args.transformer_path = resolve_model(args.transformer_path, DREAMX_REPO)
    print(f"[run_examples] model_name       = {args.model_name}")
    print(f"[run_examples] transformer_path = {args.transformer_path}")

    if not args.input_json.exists():
        print(f"[run_examples] missing eval json: {args.input_json}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.indices is not None:
        with open(args.input_json, encoding="utf-8") as f:
            items = json.load(f)
        subset = [items[i] for i in args.indices]
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False)
        json.dump(subset, tmp)
        tmp.close()
        input_json = Path(tmp.name)
        print(f"[run_examples] subset {args.indices} -> {input_json} ({len(subset)} items)")
    else:
        input_json = args.input_json

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    cmd = build_cmd(args, input_json)
    print("[run_examples] cmd:", " ".join(cmd))
    return subprocess.call(cmd, env=env, cwd=str(HERE))


if __name__ == "__main__":
    sys.exit(main())
