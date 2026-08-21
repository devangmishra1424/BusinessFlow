"""Downloads output files from a Kaggle kernel via the API instead of the
website's per-file download button, which has been failing silently on
large files (e.g. the fine-tuned model zip). Auth comes from KAGGLE_API_TOKEN
in .env -- this project's kaggle package (2.2.4) supports that env var
directly, no kaggle.json needed.

Run from anywhere: python kaggle/download_kernel_output.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402  (needs KAGGLE_API_TOKEN loaded first)

KERNEL = "mishradevang14/notebookd3491b2506"
FILE_PATTERN = r"businessflow-whisper-finetuned\.zip"
OUTPUT_DIR = _PROJECT_ROOT / "kaggle" / "downloaded_model"


def main():
    api = KaggleApi()
    api.authenticate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files, next_token = api.kernels_output(
        KERNEL, str(OUTPUT_DIR), file_pattern=FILE_PATTERN, force=True, quiet=False,
    )

    if not files:
        print("no matching files found in kernel output -- check the notebook has actually "
              "written businessflow-whisper-finetuned.zip to /kaggle/working/ in its latest saved version")
        return

    for f in files:
        full_path = OUTPUT_DIR / f
        size_mb = full_path.stat().st_size / (1024 * 1024) if full_path.exists() else 0
        print(f"downloaded: {full_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
