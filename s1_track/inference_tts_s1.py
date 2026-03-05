#!/usr/bin/env python3
"""Compatibility wrapper. Use codecs/s1_dac/scripts/inference_tts.py."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "codecs" / "s1_dac" / "scripts" / "inference_tts.py"
    print(
        "[DEPRECATED] s1_track/inference_tts_s1.py -> codecs/s1_dac/scripts/inference_tts.py",
        file=sys.stderr,
        flush=True,
    )
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
