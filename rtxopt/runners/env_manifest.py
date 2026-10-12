"""Record the environment an experiment ran in (driver, CUDA, llama.cpp commit, GPU state)."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

GPU_FIELDS = (
    "name",
    "driver_version",
    "persistence_mode",
    "ecc.mode.current",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "memory.total",
)


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    """Return stripped stdout, or None if the tool is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def parse_gpu_query(csv_text: str, fields: tuple[str, ...] = GPU_FIELDS) -> list[dict[str, str]]:
    """Parse `nvidia-smi --format=csv,noheader,nounits` output into one dict per GPU."""
    gpus = []
    for line in csv_text.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(fields):
            raise ValueError(f"expected {len(fields)} columns, got {len(values)}: {line!r}")
        gpus.append(dict(zip(fields, values, strict=True)))
    return gpus


def parse_nvcc_version(text: str) -> str | None:
    for line in text.splitlines():
        if "release" in line:
            return line.split("release", 1)[1].split(",")[0].strip()
    return None


def git_info(repo: Path) -> dict[str, str | bool | None]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    return {
        "path": str(repo),
        "commit": head,
        "describe": _run(["git", "describe", "--tags", "--always"], cwd=repo),
        "dirty": bool(status) if status is not None else None,
    }


def collect(llama_cpp: Path | None = None, project: Path | None = None) -> dict:
    smi = _run(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"]
    )
    nvcc = _run(["nvcc", "--version"])
    manifest: dict = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "gpus": parse_gpu_query(smi) if smi else [],
        "cuda_toolkit": parse_nvcc_version(nvcc) if nvcc else None,
        "ncu": (_run(["ncu", "--version"]) or "").splitlines()[-1:] or None,
        "nsys": _run(["nsys", "--version"]),
        "project": git_info(project) if project else None,
        "llama_cpp": git_info(llama_cpp) if llama_cpp else None,
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-cpp", type=Path, help="path to the llama.cpp checkout")
    ap.add_argument("--project", type=Path, default=Path.cwd())
    ap.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    args = ap.parse_args(argv)
    text = json.dumps(collect(args.llama_cpp, args.project), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
