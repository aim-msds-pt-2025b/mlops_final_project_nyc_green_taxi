from pathlib import Path

import requests

from src.config import load_config


def download_file(url: str, out_path: Path, chunk: int = 1 << 20):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)


def main():
    cfg = load_config()
    url = cfg.data["url"]
    raw_dir = Path(cfg.paths["raw_dir"])
    out = raw_dir / url.split("/")[-1]
    if out.exists():
        print(f"[data] Exists: {out}")
        return str(out)
    print(f"[data] Downloading {url} -> {out}")
    download_file(url, out)
    print(f"[data] Saved: {out} ({out.stat().st_size/1e6:.2f} MB)")
    return str(out)


if __name__ == "__main__":
    main()
