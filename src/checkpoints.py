from pathlib import Path

def get_checkpoints(path: str):
    return list(Path(path).rglob("*.ckpt"))

def get_latest_checkpoint(path: str):
    checkpoints = get_checkpoints(path)
    return max(checkpoints, key=lambda p: p.stat().st_mtime, default=None)