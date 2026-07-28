"""
Download LaCADIAN daily data files from Azure Blob Storage.

The LaCADIAN portal (sharelab.data.lsuagcenter.com) backs onto Azure Blob
Storage at sharelabdata.blob.core.windows.net/data/.  Daily files are
tab-separated .txt at:
    LaCADIAN/Daily/{year}/{site_id}_{site_name}_DAILY_{year}.txt

Usage:
    python -m swapstress.sources.lacadian_download
"""

import os
import urllib.request

BLOB_BASE = "https://sharelabdata.blob.core.windows.net/data/LaCADIAN/Daily"

# (site_id, blob_name_stem) — blob names verified against the portal
SITES = [
    ("LAC001", "LAC001_Ben_Hur"),
    ("LAC002", "LAC002_Hammond"),
    ("LAC003", "LAC003_StGabriel"),
    ("LAC004", "LAC004_Homer"),
    ("LAC005", "LAC005_BossierCity"),
    ("LAC006", "LAC006_Winnsboro"),
    ("LAC007", "LAC007_Chase"),
    ("LAC008", "LAC008_Alexandria_West"),
    ("LAC009", "LAC009_Alexandria_East"),
    ("LAC010", "LAC010_StJoseph"),
    ("LAC011", "LAC011_Franklinton"),
    ("LAC012", "LAC012_Clinton"),
    ("LAC013", "LAC013_Jeanerette"),
    ("LAC014", "LAC014_Crowley"),
    ("LAC015", "LAC015_Amite"),
    ("LAC016", "LAC016_LAHouse"),
    ("LAC017", "LAC017_LakeProvidence"),
    ("LAC018", "LAC018_Tallulah"),
    ("LAC019", "LAC019_Newellton"),
    ("LAC020", "LAC020_Deridder"),
    ("LAC021", "LAC021_Kinder"),
    ("LAC022", "LAC022_Leeville"),
    ("LAC023", "LAC023_StFrancisville"),
]

YEARS = ["2025", "2026"]

OUT_DIR = os.path.join("/nas", "soils", "soil_potential_obs", "lacadian", "raw")


def download_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    succeeded, skipped, failed = 0, 0, 0

    for site_id, stem in SITES:
        for year in YEARS:
            fname = f"{stem}_DAILY_{year}.txt"
            dest = os.path.join(OUT_DIR, fname)
            if os.path.exists(dest):
                print(f"  skip {fname} (exists)")
                skipped += 1
                continue

            url = f"{BLOB_BASE}/{year}/{fname}"
            try:
                urllib.request.urlretrieve(url, dest)
                size_kb = os.path.getsize(dest) / 1024
                print(f"  {fname}  ({size_kb:.0f} KB)")
                succeeded += 1
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"  {fname}  (not found, skipping)")
                else:
                    print(f"  {fname}  HTTP {e.code}")
                failed += 1
            except Exception as e:
                print(f"  {fname}  error: {e}")
                failed += 1

    print(f"\nDone: {succeeded} downloaded, {skipped} skipped, {failed} not found")


if __name__ == "__main__":
    download_all()

# ========================= EOF ====================================================================
