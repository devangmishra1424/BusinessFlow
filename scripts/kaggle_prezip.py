"""Zips kaggle/training_assets/audio and english_regression into flat-root
zips (audio.zip, english_regression.zip) using ZIP_STORED (no compression),
matching the internal layout shutil.make_archive would produce (kaggle's own
dir_mode="zip" path) but skipping DEFLATE entirely.

Why: the kaggle package's dir_mode="zip" upload path re-zips these folders
itself via shutil.make_archive with ZIP_DEFLATED. WAV/PCM audio barely
compresses (it's already dense samples, not text/structured data), so that
compression pass burns ~10+ minutes of pure CPU for a low-single-digit-percent
size reduction. Pre-zipping here with ZIP_STORED and uploading the resulting
flat files (dir_mode="skip", the default) gets the identical uploaded bytes
layout without paying for compression that doesn't pay for itself.

Run: python scripts/kaggle_prezip.py
"""

import time
import zipfile
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "kaggle" / "training_assets"


def _zip_flat(source_dir: Path, zip_path: Path) -> None:
    files = sorted(p for p in source_dir.iterdir() if p.is_file())
    start = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, path in enumerate(files):
            zf.write(path, arcname=path.name)
            if (i + 1) % 5000 == 0:
                print(f"  {zip_path.name}: {i + 1}/{len(files)} written")
    print(f"{zip_path.name}: {len(files)} files, {zip_path.stat().st_size / 1e9:.2f} GB, {time.time() - start:.1f}s")


def main():
    _zip_flat(_ASSETS_DIR / "audio", _ASSETS_DIR / "audio.zip")
    _zip_flat(_ASSETS_DIR / "english_regression", _ASSETS_DIR / "english_regression.zip")


if __name__ == "__main__":
    main()
