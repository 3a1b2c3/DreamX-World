"""Drive DreamX-World from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample:
  1. Extracts the first frame from video.mp4 -> a temp PNG.
  2. Builds a single-entry eval.json for DreamX-World pointing at that frame.
  3. Calls inference_dreamx5b.py for that one entry, writing the mp4 into a
     scratch dir.
  4. Renames the produced mp4 to:
        test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4

Skip-if-exists: samples whose output mp4 already exists are skipped.

MIND action.json fields (ws/ad/ud/lr timeseries) are NOT translated to DreamX's
action_seq / action_speed_list — this driver uses a single placeholder action
("w" at speed 6) so the pipeline runs end-to-end. Replace `derive_actions()`
with a real translator if you want pose-faithful evaluation.

After running, score with MIND:
    python C:\\workspace\\world\\MIND\\src\\process.py \\
        --gt_root  C:\\workspace\\world\\MIND-Data \\
        --test_root C:\\workspace\\world\\MIND-tests\\dreamx-world \\
        --metrics lcm,visual,dino,action

Usage:
    python drive_mind.py --gt-root C:\\workspace\\world\\MIND-Data \\
                         --test-root C:\\workspace\\world\\MIND-tests\\dreamx-world
    python drive_mind.py --limit 3 --perspective 1st_data --test-type action_space_test
    python drive_mind.py --dry-run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import av
from PIL import Image

DREAMX_REPO = Path(__file__).resolve().parent
DREAMX_INFERENCE = DREAMX_REPO / "inference_dreamx5b.py"
DREAMX_CONFIG = DREAMX_REPO / "configs" / "wan2.2" / "wan_ti2v_5b.yaml"
DREAMX_MODEL_NAME = DREAMX_REPO / "Wan2.2-TI2V-5B"
DREAMX_TRANSFORMER = DREAMX_REPO / "DreamX-World-5B-Cam"
DREAMX_VENV_PY = DREAMX_REPO / ".venv" / "Scripts" / "python.exe"

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")

# Same DreamX defaults as run_examples.py
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 1280
DEFAULT_VIDEO_LENGTH = 121
DEFAULT_FPS = 24
DEFAULT_GUIDANCE = 3.0
DEFAULT_STEPS = 50
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DTYPE = "bfloat16"
DEFAULT_CAM_METHOD = "prope"


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_image()
            img.save(out_path, "PNG")
            return
    raise RuntimeError(f"No frames decoded from {video_path}")


def derive_actions(action_path: Path, perspective: str) -> tuple[list[str], list[int]]:
    """Map MIND action.json -> DreamX action_seq / action_speed_list.

    Placeholder: returns a single forward action. Replace with a real translator
    if you need pose-faithful drive (would need to bucket the ws/ad/ud/lr
    timeseries into DreamX's discrete action codes per ~2-3 second segment).
    """
    return ["w"], [6]


def derive_caption(action_path: Path, perspective: str) -> str:
    """MIND samples don't ship a textual caption — use a stock prompt."""
    if perspective == "1st_data":
        return "First-person view exploring a 3D virtual environment in photorealistic style."
    return "Third-person view of a character exploring a 3D virtual environment in photorealistic style."


def gather_samples(gt_root: Path) -> list[dict]:
    samples: list[dict] = []
    for perspective in PERSPECTIVES:
        for test_type in TEST_TYPES:
            type_dir = gt_root / perspective / "test" / test_type
            if not type_dir.is_dir():
                continue
            for sample_dir in sorted(type_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                video = sample_dir / "video.mp4"
                action = sample_dir / "action.json"
                if not (video.exists() and action.exists()):
                    continue
                samples.append({
                    "perspective": perspective,
                    "test_type": test_type,
                    "gt_name": sample_dir.name,
                    "video": video,
                    "action": action,
                })
    return samples


def output_path(test_root: Path, model_name: str, sample: dict) -> Path:
    return (
        test_root / model_name / sample["perspective"] / sample["test_type"]
        / sample["gt_name"] / "video.mp4"
    )


def predict_dreamx_filename(image_stem: str, action_seq: list[str]) -> str:
    """Mirror DreamX's filename convention: <image_stem>_<action1>_<action2>...mp4."""
    return f"{image_stem}_{'_'.join(action_seq)}.mp4"


def run_one(sample: dict, test_root: Path, model_name: str, work_dir: Path,
            args: argparse.Namespace) -> int:
    out = output_path(test_root, model_name, sample)
    if out.exists() and not args.force:
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out} (exists)")
        return 0

    frame_png = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.png"
    extract_first_frame(sample["video"], frame_png)

    action_seq, action_speed_list = derive_actions(sample["action"], sample["perspective"])
    caption = derive_caption(sample["action"], sample["perspective"])

    # DreamX's inference_dreamx5b.py reads a JSON list of entries via --input_dir
    # (misnamed: it's a JSON path). Write a single-entry list pointing at our PNG.
    entry = {
        "image_path": str(frame_png),
        "caption": caption,
        "action_seq": action_seq,
        "action_speed_list": action_speed_list,
    }

    eval_json = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.eval.json"
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump([entry], f, indent=2)

    # DreamX writes <stem>_<actions>.mp4 into --output_dir. Use a scratch dir
    # per-sample so we can move the unique result into the MIND layout.
    scratch_out = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.out"
    scratch_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.venv_py),
        str(DREAMX_INFERENCE),
        "--config_path", str(DREAMX_CONFIG),
        "--model_name", str(DREAMX_MODEL_NAME),
        "--transformer_path", str(DREAMX_TRANSFORMER),
        "--input_dir", str(eval_json),
        "--output_dir", str(scratch_out),
        "--cam_method", args.cam_method,
        "--add_control_adapter",
        "--sample_size", str(args.height), str(args.width),
        "--video_length", str(args.video_length),
        "--fps", str(args.fps),
        "--guidance_scale", str(args.guidance_scale),
        "--num_inference_steps", str(args.steps),
        "--seed", str(args.seed),
        "--weight_dtype", args.weight_dtype,
        "--ulysses_degree", "1",
        "--ring_degree", "1",
    ]
    if args.memory_mode:
        cmd += ["--GPU_memory_mode", args.memory_mode]

    print(f"\n=== {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} ===")
    print(f"caption: {caption[:90]}{'...' if len(caption) > 90 else ''}")
    print(f"actions: {action_seq} @ speeds {action_speed_list}")
    print(f"out:     {out}")

    if args.dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return 0

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = f"{DREAMX_REPO};{env.get('PYTHONPATH', '')}"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(DREAMX_REPO), env=env)
    elapsed = time.perf_counter() - t0
    print(f"  rc={rc}  elapsed={elapsed:.1f}s")

    if rc != 0:
        return rc

    # Find the produced mp4 in scratch_out and move into the MIND layout
    predicted = scratch_out / predict_dreamx_filename(frame_png.stem, action_seq)
    if predicted.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(predicted), str(out))
        print(f"  -> {out}")
    else:
        # Fallback: pick any mp4 that landed
        mp4s = sorted(scratch_out.glob("*.mp4"))
        if not mp4s:
            print(f"  WARNING: no mp4 produced in {scratch_out}", file=sys.stderr)
            return 3
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mp4s[0]), str(out))
        print(f"  -> {out}  (fallback from {mp4s[0].name})")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-root", type=Path, default=Path(r"C:\workspace\world\MIND-Data"))
    p.add_argument("--test-root", type=Path, default=Path(r"C:\workspace\world\MIND-tests"),
                   help="Parent dir; outputs land at <test-root>/<model-name>/<perspective>/...")
    p.add_argument("--model-name", default="dreamx-world", help="Subfolder name under test-root")
    p.add_argument("--work-dir", type=Path, default=None, help="Temp dir (default: <test-root>/.frames)")
    p.add_argument("--only", nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    p.add_argument("--perspective", choices=PERSPECTIVES)
    p.add_argument("--test-type", choices=TEST_TYPES)
    p.add_argument("--limit", type=int, help="Only run first N matched samples")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--venv-py", type=Path, default=DREAMX_VENV_PY)
    # DreamX inference knobs
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--video-length", type=int, default=DEFAULT_VIDEO_LENGTH)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--weight-dtype", default=DEFAULT_WEIGHT_DTYPE, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cam-method", default=DEFAULT_CAM_METHOD, choices=["prope", "plucker"])
    p.add_argument("--memory-mode", default=None,
                   choices=[None, "model_full_load", "model_full_load_and_qfloat8",
                            "model_cpu_offload", "model_cpu_offload_and_qfloat8",
                            "sequential_cpu_offload"])
    args = p.parse_args()

    for path in (DREAMX_INFERENCE, DREAMX_CONFIG, DREAMX_MODEL_NAME, DREAMX_TRANSFORMER):
        if not path.exists():
            print(f"FATAL: missing {path}", file=sys.stderr)
            print("Run `python download_models.py` first.", file=sys.stderr)
            return 2
    if not args.venv_py.exists():
        print(f"FATAL: python.exe not found at {args.venv_py}", file=sys.stderr)
        return 2

    work_dir = args.work_dir or (args.test_root / ".frames")
    work_dir.mkdir(parents=True, exist_ok=True)

    samples = gather_samples(args.gt_root)
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    if args.only:
        samples = [s for s in samples if any(sub.lower() in s["gt_name"].lower() for sub in args.only)]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s):")
    for s in samples:
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}")
    print()

    failures: list[str] = []
    for s in samples:
        rc = run_one(s, args.test_root, args.model_name, work_dir, args)
        if rc != 0:
            failures.append(f"{s['perspective']}/{s['test_type']}/{s['gt_name']}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for name in failures:
            print(f"  {name}")
        return 1
    print(f"Done. {len(samples)} sample(s) produced.")
    print(f"Next: score with C:\\workspace\\world\\MIND\\src\\process.py --gt_root {args.gt_root} --test_root {args.test_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
