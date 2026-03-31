#!/usr/bin/env python3
"""Compatibility wrapper. Use codecs/s1_dac/scripts/pretokenize_fleurs.py."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "codecs" / "s1_dac" / "scripts" / "pretokenize_fleurs.py"
    print(
        "[DEPRECATED] s1_track/scripts/pretokenize_fleurs_s1.py -> codecs/s1_dac/scripts/pretokenize_fleurs.py",
        file=sys.stderr,
        flush=True,
    )
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
