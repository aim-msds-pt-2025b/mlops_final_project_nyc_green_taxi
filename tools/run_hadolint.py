import os
import subprocess
import sys
from pathlib import Path


def main():
    files = [f for f in sys.argv[1:] if Path(f).is_file()]
    if not files:
        return 0
    # Ensure docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print("hadolint skipped: docker not available", file=sys.stderr)
        return 0
    image = os.environ.get("HADOLINT_IMAGE", "ghcr.io/hadolint/hadolint:latest")
    # Run hadolint for each file so filenames are correct in output
    # Mount repo root as /workspace
    repo = Path.cwd()
    failures = 0
    for f in files:
        rel = Path(f).resolve().relative_to(repo.resolve())
        rel_posix = rel.as_posix()
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{repo}:/workspace",
            "-w",
            "/workspace",
            image,
            "hadolint",
            rel_posix,
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failures = 1
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
