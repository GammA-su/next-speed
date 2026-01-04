import json
import os
import random
import sys
import time
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional for non-ML scripts
    torch = None

if torch is not None and not hasattr(torch, "Tensor"):
    sys.modules.pop("torch", None)
    torch = None


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None or not hasattr(torch, "manual_seed"):
        return
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
