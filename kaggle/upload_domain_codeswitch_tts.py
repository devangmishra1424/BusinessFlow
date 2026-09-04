"""One-off: zip and push kaggle/domain_codeswitch_tts_assets/ as a brand-new
Kaggle dataset (mishradevang14/domain-codeswitch-tts). Same ZIP_STORED
flat-zip + resumable-upload-retry pattern as kaggle/upload_splicing_fuel.py
-- see that script's own docstring for why each piece is shaped this way.

Run: python kaggle/upload_domain_codeswitch_tts.py
"""

import tempfile
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402  (needs KAGGLE_API_TOKEN loaded first)

_ASSETS_DIR = _PROJECT_ROOT / "kaggle" / "domain_codeswitch_tts_assets"
UPLOADS_DIR = Path(tempfile.gettempdir()) / ".kaggle" / "uploads"
MAX_ATTEMPTS = 3


def _zip_flat(source_dir: Path, zip_path: Path) -> None:
    files = sorted(p for p in source_dir.iterdir() if p.is_file())
    start = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, path in enumerate(files):
            zf.write(path, arcname=path.name)
            if (i + 1) % 500 == 0:
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
                dir_mode="skip",
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
