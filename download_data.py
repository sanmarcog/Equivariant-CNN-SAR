"""
download_data.py

Download and verify the AvalCD dataset from Zenodo (record 15863589).
Extracts to data/raw/AvalCD/ by default.

Usage:
    python download_data.py
    python download_data.py --data-dir /gscratch/your_group/avalanche/data
    python download_data.py --skip-download   # verify + extract only

Dependencies: requests, tqdm  (pip install requests tqdm)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

ZENODO_RECORD_ID = "15863589"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
CHUNK_SIZE = 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# Zenodo metadata
# ---------------------------------------------------------------------------

def fetch_record_metadata() -> dict:
    """Fetch file list and checksums from the Zenodo API."""
    log.info("Fetching record metadata from Zenodo (record %s)...", ZENODO_RECORD_ID)
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_file_info(metadata: dict, filename: str) -> tuple[str, str]:
    """Return (download_url, checksum_string) for *filename* in the record."""
    for f in metadata["files"]:
        if f["key"] == filename:
            return f["links"]["self"], f["checksum"]
    raise FileNotFoundError(
        f"{filename!r} not found in Zenodo record {ZENODO_RECORD_ID}. "
        f"Available files: {[f['key'] for f in metadata['files']]}"
    )


# ---------------------------------------------------------------------------
# Download with resume support
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    """
    Stream-download *url* to *dest* with a progress bar.
    Resumes a partial download if *dest* already exists and is incomplete.
    """
    existing_bytes = dest.stat().st_size if dest.exists() else 0

    if expected_size and existing_bytes == expected_size:
        log.info("File already fully downloaded: %s", dest)
        return

    headers = {}
    if existing_bytes > 0:
        log.info("Resuming download from byte %d...", existing_bytes)
        headers["Range"] = f"bytes={existing_bytes}-"

    resp = requests.get(url, stream=True, headers=headers, timeout=60)

    # 416 = range not satisfiable → server doesn't support resume, restart
    if resp.status_code == 416:
        log.warning("Server does not support range requests — restarting download.")
        existing_bytes = 0
        resp = requests.get(url, stream=True, timeout=60)

    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + existing_bytes

    mode = "ab" if existing_bytes > 0 else "wb"
    with open(dest, mode) as f, tqdm(
        total=total,
        initial=existing_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------

def verify_checksum(path: Path, checksum_str: str) -> bool:
    """
    Verify *path* against a Zenodo checksum string like 'md5:abc123...' or
    'sha256:abc123...'. Returns True if valid.
    """
    algo, expected = checksum_str.split(":", 1)
    h = hashlib.new(algo)
    with open(path, "rb") as f, tqdm(
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Verifying {path.name}",
    ) as bar:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
            bar.update(len(chunk))
    actual = h.hexdigest()
    if actual == expected:
        log.info("Checksum OK (%s:%s)", algo, actual)
        return True
    else:
        log.error("Checksum MISMATCH for %s", path)
        log.error("  Expected: %s", expected)
        log.error("  Got:      %s", actual)
        return False


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract *zip_path* into *dest_dir*, showing a file-count progress bar."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        log.info("Extracting %d files to %s...", len(members), dest_dir)
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, dest_dir)
    log.info("Extraction complete: %s", dest_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download AvalCD dataset from Zenodo.")
    p.add_argument(
        "--data-dir", default=Path("data"), type=Path,
        help="Root data directory (default: ./data). Zip saved to <data-dir>/raw/, "
             "extracted to <data-dir>/raw/AvalCD/.",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; verify and extract an existing zip file.",
    )
    p.add_argument(
        "--skip-extract", action="store_true",
        help="Download and verify only; do not extract.",
    )
    p.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the zip file after extraction (default: delete it).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir: Path = args.data_dir.resolve() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Fetch metadata -------------------------------------------------
    metadata = fetch_record_metadata()

    # Dump a small provenance file alongside the data
    provenance_path = raw_dir / "zenodo_metadata.json"
    with open(provenance_path, "w") as f:
        json.dump(
            {
                "record_id": ZENODO_RECORD_ID,
                "doi": metadata.get("doi"),
                "title": metadata.get("metadata", {}).get("title"),
                "version": metadata.get("metadata", {}).get("version"),
                "files": [
                    {"key": f["key"], "size": f["size"], "checksum": f["checksum"]}
                    for f in metadata["files"]
                ],
            },
            f,
            indent=2,
        )
    log.info("Provenance written: %s", provenance_path)

    # --- Identify the zip -----------------------------------------------
    zip_filename = "AvalCD.zip"
    download_url, checksum_str = get_file_info(metadata, zip_filename)
    zip_path = raw_dir / zip_filename
    expected_size = next(
        f["size"] for f in metadata["files"] if f["key"] == zip_filename
    )

    # --- Download -------------------------------------------------------
    if not args.skip_download:
        download_file(download_url, zip_path, expected_size=expected_size)
    else:
        if not zip_path.exists():
            log.error("--skip-download set but zip not found: %s", zip_path)
            sys.exit(1)
        log.info("Skipping download, using existing file: %s", zip_path)

    # --- Verify ---------------------------------------------------------
    if not verify_checksum(zip_path, checksum_str):
        log.error("Checksum failed. Delete %s and re-run to download again.", zip_path)
        sys.exit(1)

    # --- Extract --------------------------------------------------------
    if not args.skip_extract:
        extract_dir = raw_dir / "AvalCD"
        if extract_dir.exists():
            log.info("Extraction target already exists: %s", extract_dir)
            log.info("Remove it manually if you want to re-extract.")
        else:
            extract_zip(zip_path, raw_dir)

        if not args.keep_zip:
            zip_path.unlink()
            log.info("Deleted zip: %s", zip_path)

    log.info("Done. Dataset available at: %s", raw_dir / "AvalCD")


if __name__ == "__main__":
    main()
