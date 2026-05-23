#!/usr/bin/env python3
"""Download GDSC data files for the real-data application."""

from __future__ import annotations

from pathlib import Path
import sys
import urllib.error
import urllib.request


RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

FILES = {
    "GDSC1_fitted_dose_response_27Oct23.xlsx": [
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx"
    ],
    "GDSC2_fitted_dose_response_27Oct23.xlsx": [
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx"
    ],
    "Cell_Lines_Details.xlsx": [
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx"
    ],
    "Cell_line_RMA_proc_basalExp.txt.zip": [
        "https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/"
        "Cell_line_RMA_proc_basalExp.txt.zip",
        "https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/DeepTTC/"
        "Cell_line_RMA_proc_basalExp.txt.zip",
    ],
}


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download_file(name: str, urls: list[str]) -> bool:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / name
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"skip existing {name}: {format_size(output_path.stat().st_size)}")
        return True

    for url in urls:
        print(f"downloading {name}")
        print(f"  {url}")
        try:
            headers = {} if "ftp.mcs.anl.gov" in url else {"User-Agent": "Mozilla/5.0"}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, output_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"downloaded {name}: {format_size(output_path.stat().st_size)}")
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if output_path.exists():
                output_path.unlink()
            print(f"FAILED {name} from {url}: {exc}", file=sys.stderr)

    print("Manual download may be needed from https://www.cancerrxgene.org/downloads/bulk_download")
    return False


def main() -> None:
    successes = [download_file(name, url) for name, url in FILES.items()]
    if not all(successes):
        failed = [name for name, ok in zip(FILES, successes) if not ok]
        print("\nSome files failed to download:", ", ".join(failed), file=sys.stderr)
        print("Place the missing files in real_app_gdsc/data/raw/ and rerun prepare_gdsc.py.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
