"""One-off: zip and push kaggle/splicing_fuel_assets/ as a brand-new Kaggle
dataset (mishradevang14/splicing-fuel). Same ZIP_STORED flat-zip approach as
scripts/kaggle_prezip.py (WAV audio doesn't compress worth the CPU time --
see that script's own docstring), and the same resumable-upload retry as
scripts/kaggle_upload_dataset.py (a real, reproduced-on-this-machine bug in
the installed kaggle package, not a data/auth problem -- see that script's
own docstring). dataset_create_new, not dataset_create_version -- this
dataset doesn't exist on Kaggle yet.

Run: python kaggle/upload_splicing_fuel.py
"""

import tempfile
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402  (needs KAGGLE_API_TOKEN loaded first)

_ASSETS_DIR = _PROJECT_ROOT / "kaggle" / "splicing_fuel_assets"
UPLOADS_DIR = Path(tempfile.gettempdir()) / ".kaggle" / "uploads"
MAX_ATTEMPTS = 3


def _zip_flat(source_dir: Path, zip_path: Path) -> None:
    files = sorted(p for p in source_dir.iterdir() if p.is_file())
    start = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, path in enumerate(files):
            zf.write(path, arcname=path.name)
            if (i + 1) % 1000 == 0:
                print(f"  {zip_path.name}: {i + 1}/{len(files)} written")
    print(f"{zip_path.name}: {len(files)} files, {zip_path.stat().st_size / 1e9:.2f} GB, {time.time() - start:.1f}s")


def main():
    _zip_flat(_ASSETS_DIR / "audio", _ASSETS_DIR / "audio.zip")

    api = KaggleApi()
    api.authenticate()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            response = api.dataset_create_new(
                folder=str(_ASSETS_DIR),
                dir_mode="skip",  # audio.zip is pre-made; manifest.json/dataset-metadata.json go up as-is
                convert_to_csv=False,
                public=False,
            )
            print(response)
            return
        except FileNotFoundError as e:
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"attempt {attempt}/{MAX_ATTEMPTS} hit {e!r}, retrying after recreating {UPLOADS_DIR}")
            time.sleep(2 * attempt)


if __name__ == "__main__":
    main()
