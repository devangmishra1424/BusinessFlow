"""One-off: push the full 55,147-utterance extraction (kaggle/training_assets/)
as a new version of the existing mishradevang14/training-set Kaggle dataset.

audio.zip and english_regression.zip are pre-made (see scripts/kaggle_prezip.py,
ZIP_STORED, no compression -- WAV audio doesn't compress worth the CPU time)
and already sit as flat files in this folder, so dir_mode="skip" (the default)
is correct here: it uploads those two zips plus manifest.json as-is and
ignores the original audio/ and english_regression/ subdirectories, rather
than dir_mode="zip" re-zipping them with kaggle's internal ZIP_DEFLATED path.

Retries: the installed kaggle package (2.2.4) writes a per-file resumable-
upload tracking json into %TEMP%/.kaggle/uploads, created via os.makedirs on
context entry. On this machine that directory has been observed to vanish
between creation and the first write a moment later (most likely Windows
Defender's real-time scan touching a just-created temp dir) -- reproducibly
enough to fail the very first blob every time, but not a real data or auth
problem, so a short bounded retry with the directory freshly (re-)created
each attempt is the right fix, not disabling resumable uploads.
"""

import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402  (needs KAGGLE_API_TOKEN loaded first)

UPLOADS_DIR = Path(tempfile.gettempdir()) / ".kaggle" / "uploads"
MAX_ATTEMPTS = 3

api = KaggleApi()
api.authenticate()

folder = str(_PROJECT_ROOT / "kaggle" / "training_assets")

for attempt in range(1, MAX_ATTEMPTS + 1):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        response = api.dataset_create_version(
            folder=folder,
            version_notes="Full corpus: 52,825 MUCS + 2,322 HiACC utterances (was a 3,500-utterance sample)",
            dir_mode="skip",
            convert_to_csv=False,
        )
        print(response)
        break
    except FileNotFoundError as e:
        if attempt == MAX_ATTEMPTS:
            raise
        print(f"attempt {attempt}/{MAX_ATTEMPTS} hit {e!r}, retrying after recreating {UPLOADS_DIR}")
        time.sleep(2 * attempt)
