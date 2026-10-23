import hashlib
from pathlib import Path

import pandas as pd

_HASH_CHUNK_BYTES = 1024 * 1024


def file_fingerprint(path: Path) -> dict[str, object]:
    """Fingerprint a Parquet data file: name, time range, row count, content hash.

    The hash is computed from the raw file bytes (not a DataFrame
    re-serialization), so it is independently reproducible with any
    standard sha256 tool outside pandas.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path!r}.")

    df = pd.read_parquet(path)
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{path!r} does not have a DatetimeIndex (got {type(index).__name__}).")

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)

    return {
        "filename": path.name,
        "first_timestamp": index.min().isoformat(),
        "last_timestamp": index.max().isoformat(),
        "row_count": len(df),
        "sha256": digest.hexdigest(),
    }
