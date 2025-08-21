import logging
from pathlib import Path

import requests

from src.config import load_config
from src.logging_utils import setup_logging

log = logging.getLogger(__name__)


def download_file(url: str, out_path: Path, chunk: int = 1 << 20):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug("starting download", extra={"url": url, "out": str(out_path)})
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
    log.info("download complete", extra={"out": str(out_path), "size": out_path.stat().st_size})


def main():
    cfg = load_config()
    setup_logging(cfg)
    url = cfg.data["url"]
    raw_dir = Path(cfg.paths["raw_dir"])
    out = raw_dir / url.split("/")[-1]
    if out.exists():
        log.warning("raw file exists; skipping download", extra={"path": str(out)})
        return str(out)
    log.info("downloading", extra={"url": url, "out": str(out)})
    download_file(url, out)
    log.info(
        "saved",
        extra={"path": str(out), "mb": round(out.stat().st_size / 1e6, 2)},
    )
    return str(out)


if __name__ == "__main__":
    main()
