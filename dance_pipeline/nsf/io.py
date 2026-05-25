"""NSF sequence serialization."""

import numpy as np
from pathlib import Path
from .format import NSFSequence


def save_nsf(seq: NSFSequence, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=seq.positions,
        rotations=seq.rotations,
        contacts=seq.contacts,
        fps=np.float32(seq.fps),
        source=np.str_(seq.source),
        name=np.str_(seq.name),
    )


def load_nsf(path: str | Path) -> NSFSequence:
    d = np.load(path, allow_pickle=False)
    return NSFSequence(
        positions=d["positions"].astype(np.float32),
        rotations=d["rotations"].astype(np.float32),
        contacts=d["contacts"].astype(np.float32),
        fps=float(d["fps"]),
        source=str(d["source"]),
        name=str(d["name"]),
    )
