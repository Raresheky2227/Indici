#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE, env=env or os.environ.copy(), check=True)
    return r.returncode

def main():
    ap = argparse.ArgumentParser(description="Run: 1) opcom_scraper.py  2) indici.py")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate email HTML preview only (sets INDICI_DRY_RUN=1).")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.dry_run:
        env["INDICI_DRY_RUN"] = "1"

    # 1) Scrape & save CSVs (defaults are aligned with indici.py expectations)
    run([sys.executable, "opcom_scraper.py"], env=env)

    # 2) Build & send email (reads CSVs from data/opcom_csv/<YYYY-MM-DD>/)
    run([sys.executable, "indici.py"], env=env)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
