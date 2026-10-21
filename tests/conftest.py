"""Put the repo root on sys.path so tests can ``import ask_llm`` and ``import experiments``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
