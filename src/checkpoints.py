from pathlib import Path

def get_checkpoints():
    return list(Path("./lightning_logs").rglob("*.ckpt"))

def get_latest_checkpoint():
    checkpoints = get_checkpoints()
    return max(checkpoints, key=lambda p: p.stat().st_mtime)