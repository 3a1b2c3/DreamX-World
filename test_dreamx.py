"""Quick single-sample DreamX test from the CLI.

Builds a one-entry eval.json on the fly, runs inference_dreamx5b.py, and
prints the path of the produced mp4.

Usage:
    test_dreamx.bat IMAGE PROMPT --actions w,wj --speeds 4,6
    python test_dreamx.py IMAGE PROMPT --actions w,wj

Action vocab: w/s/a/d (translate), j/k (tilt), h/l (pan). Combine letters
in one segment, e.g. "wj" = forward + tilt down. --speeds (1-8) is a
parallel comma-list; defaults to 4 for every segment.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFER = HERE / "inference_dreamx5b.py"
CONFIG = HERE / "configs" / "wan2.2" / "wan_ti2v_5b.yaml"
WAN = HERE / "Wan2.2-TI2V-5B"
TRANSFORMER = HERE / "DreamX-World-5B-Cam"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="First-frame image (PNG/JPG)")
    ap.add_argument("prompt", help="Caption / prompt string")
    ap.add_argument("--actions", default="w", help="Comma-separated action segments, e.g. 'w,wj,dl'")
    ap.add_argument("--speeds",  default=None, help="Comma-separated speeds (1-8) parallel to --actions; default 4")
    ap.add_argument("--output_dir",     type=Path,  default=HERE / "outputs")
    ap.add_argument("--steps",          type=int,   default=50)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--height",         type=int,   default=704)
    ap.add_argument("--width",          type=int,   default=1280)
    ap.add_argument("--video_length",   type=int,   default=121)
    ap.add_argument("--fps",            type=int,   default=24)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--weight_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--cam_method",   default="prope",    choices=["prope", "plucker"])
    args = ap.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image.resolve()}", file=sys.stderr)
        return 2
    for p in (INFER, CONFIG, WAN, TRANSFORMER):
        if not p.exists():
            print(f"ERROR: missing required path: {p}", file=sys.stderr)
            return 2

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    if not actions:
        print("ERROR: --actions cannot be empty", file=sys.stderr)
        return 2
    if args.speeds:
        speeds = [int(s.strip()) for s in args.speeds.split(",") if s.strip()]
        if len(speeds) != len(actions):
            print(f"ERROR: --speeds ({len(speeds)}) must match --actions ({len(actions)})", file=sys.stderr)
            return 2
    else:
        speeds = [4] * len(actions)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "image_path": str(args.image.resolve()).replace("\\", "/"),
        "caption": args.prompt,
        "action_seq": actions,
        "action_speed_list": speeds,
    }
    print(f"Image:      {entry['image_path']}")
    print(f"Caption:    {args.prompt}")
    print(f"Actions:    {actions}  speeds: {speeds}")
    print(f"Output dir: {args.output_dir.resolve()}")

    fd, tmp_name = tempfile.mkstemp(suffix=".eval.json", text=True)
    eval_json = Path(tmp_name)
    with open(fd, "w", encoding="utf-8") as f:
        json.dump([entry], f, ensure_ascii=False, indent=2)
    print(f"Eval JSON:  {eval_json}")

    cmd = [
        sys.executable, str(INFER),
        "--config_path",         str(CONFIG),
        "--model_name",          str(WAN),
        "--transformer_path",    str(TRANSFORMER),
        "--input_dir",           str(eval_json),
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
    print()

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(HERE))
    elapsed = time.perf_counter() - t0
    print(f"\nrc={rc}  elapsed={elapsed:.1f}s")

    try:
        eval_json.unlink()
    except OSError:
        pass

    if rc == 0:
        action_name = "_".join(actions)
        produced = args.output_dir / f"{args.image.stem}_{action_name}.mp4"
        if produced.exists():
            print(f"Output:     {produced.resolve()}")
        else:
            print(f"WARNING: expected {produced} not found; check {args.output_dir.resolve()} for the actual filename")
    return rc


if __name__ == "__main__":
    sys.exit(main())
