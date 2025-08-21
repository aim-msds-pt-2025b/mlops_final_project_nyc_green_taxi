from pathlib import Path

from src.config import load_config


def test_config_exists():
    assert Path("config.yaml").exists()


def test_url_https():
    cfg = load_config()
    assert cfg.data["url"].startswith("https://")
