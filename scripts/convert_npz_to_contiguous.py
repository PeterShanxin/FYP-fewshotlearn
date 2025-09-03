#!/usr/bin/env python
"""Convert legacy embeddings.npz → embeddings.X.npy + embeddings.keys.npy.

Usage:
  python scripts/convert_npz_to_contiguous.py data/emb/embeddings.npz

Optionally specify an explicit output base (without suffix):
  python scripts/convert_npz_to_contiguous.py data/emb/embeddings.npz data/emb/embeddings
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm.auto import tqdm


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python scripts/convert_npz_to_contiguous.py <embeddings.npz> [<output_base>]")
        sys.exit(2)
    in_npz = Path(sys.argv[1])
    if not in_npz.exists():
        print(f"Input not found: {in_npz}")
        sys.exit(1)
    if len(sys.argv) == 3:
        base = Path(sys.argv[2])
    else:
        s = str(in_npz)
        base = Path(s[:-4]) if s.endswith('.npz') else in_npz

    X_path = base.with_suffix('.X.npy')
    keys_path = base.with_suffix('.keys.npy')

    print(f"[convert] loading {in_npz} …")
    npz = np.load(in_npz, allow_pickle=False)
    keys: List[str] = sorted(npz.files)
    print(f"[convert] found {len(keys)} entries; stacking …")
    X = np.stack([npz[k] for k in tqdm(keys, desc='[convert] stack', dynamic_ncols=True)]).astype('float32', copy=False)
    print(f"[convert] X shape = {X.shape}")
    np.save(X_path, X)
    np.save(keys_path, np.array(keys, dtype='U'))
    print(f"[convert] wrote → {X_path}, {keys_path}")


if __name__ == "__main__":
    main()

