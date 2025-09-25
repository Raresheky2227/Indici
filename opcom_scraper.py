#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OPCOM tables scraper (dynamic)
URL: https://www.opcom.ro/grafice-ip-raportPIP-si-volumTranzactionat/en

- Randează pagina cu Selenium (inclusiv iframes), extrage toate <table>.
- Salvează CSV (cu data în nume) + SQLite (o masă per tabel) + ingestion_log.
"""

import os
import re
import sys
import time
import sqlite3
import argparse
import datetime as dt
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
from bs4 import BeautifulSoup

# --- Selenium ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchFrameException

DEFAULT_URL = "https://www.opcom.ro/grafice-ip-raportPIP-si-volumTranzactionat/en"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

def guess_report_date(html: str) -> dt.date:
    m = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", html)
    if m:
        d, mth, y = map(int, m.groups())
        try:
            return dt.date(y, mth, d)
        except ValueError:
            pass
    return dt.date.today()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def html_tables_to_dfs(raw_html: str) -> List[pd.DataFrame]:
    """Extrage tabele din HTML literal (folosind StringIO pentru pandas)."""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    cleaned = str(soup)
    try:
        dfs = pd.read_html(StringIO(cleaned), flavor="bs4")
    except ValueError:
        return []
    out = []
    for df in dfs:
        if df.shape[0] >= 3 and df.shape[1] >= 3:
            out.append(normalize_columns(df))
    return out

def scrape_with_selenium(url: str, headless: bool = True, wait_secs: int = 20) -> tuple[list[pd.DataFrame], str]:
    """Deschide pagina, așteaptă tabelele (și în iframes), returnează DF-urile și page_source-ul principal."""
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument(f"user-agent={UA}")
    opts.add_argument("--window-size=1400,1000")

    driver = webdriver.Chrome(options=opts)  # Selenium Manager rezolvă driverul
    try:
        driver.get(url)

        # 1) Așteaptă apariția a cel puțin unui <table> în pagina principală
        try:
            WebDriverWait(driver, wait_secs).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except TimeoutException:
            pass  # poate sunt doar în iframe – continuăm

        all_tables_html = []

        # 2) Adaugă tabelele din pagina principală
        main_tables = driver.find_elements(By.TAG_NAME, "table")
        for el in main_tables:
            all_tables_html.append(el.get_attribute("outerHTML"))

        # 3) Verifică și toate iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for i, frame in enumerate(iframes):
            try:
                driver.switch_to.frame(frame)
            except NoSuchFrameException:
                continue

            try:
                # Așteaptă până apare un <table> în cadrul curent (dacă există)
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "table"))
                    )
                except TimeoutException:
                    pass

                frame_tables = driver.find_elements(By.TAG_NAME, "table")
                for el in frame_tables:
                    all_tables_html.append(el.get_attribute("outerHTML"))
            finally:
                driver.switch_to.default_content()

        # Dacă încă n-am strâns tabele, mai așteptăm puțin (site-ul poate încărca lent)
        if not all_tables_html:
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                main_tables = driver.find_elements(By.TAG_NAME, "table")
                for el in main_tables:
                    all_tables_html.append(el.get_attribute("outerHTML"))
            except TimeoutException:
                pass

        # Convertește fiecare HTML de tabel în DataFrame
        dfs: List[pd.DataFrame] = []
        for tbl_html in all_tables_html:
            dfs.extend(html_tables_to_dfs(tbl_html))

        return dfs, driver.page_source
    finally:
        driver.quit()

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_csvs(dfs: List[pd.DataFrame], base_dir: Path, day: dt.date) -> List[Path]:
    ensure_dir(base_dir)
    out_paths = []
    for idx, df in enumerate(dfs, start=1):
        name = f"opcom_table_{idx:02d}_{day.isoformat()}.csv"
        out_path = base_dir / name
        df.to_csv(out_path, index=False)
        out_paths.append(out_path)
    return out_paths

def save_sqlite(dfs: List[pd.DataFrame], sqlite_path: Path, day: dt.date) -> None:
    ensure_dir(sqlite_path.parent)
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                table_name TEXT NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                ingested_at TEXT NOT NULL,
                report_date TEXT NOT NULL
            )
        """)
        for idx, df in enumerate(dfs, start=1):
            header = "_".join([re.sub(r"\W+", "_", c).strip("_") for c in df.columns])[:60]
            tbl = f"opcom_{idx:02d}__{header}" if header else f"opcom_{idx:02d}"
            df_sql = df.rename(columns=lambda c: re.sub(r"\W+", "_", str(c)).strip("_") or f"col_{hash(c)&0xffff}")
            df_sql.to_sql(tbl, conn, if_exists="append", index=False)
            conn.execute(
                "INSERT INTO ingestion_log (source, table_name, rows, cols, ingested_at, report_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "opcom_pip_volum",
                    tbl,
                    int(df_sql.shape[0]),
                    int(df_sql.shape[1]),
                    dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    day.isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()

def main():
    ap = argparse.ArgumentParser(description="Scrape OPCOM PIP & traded volume (dynamic).")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out-dir", default="data/opcom_csv")
    ap.add_argument("--sqlite", default="data/opcom/opcom.db")
    ap.add_argument("--no-headless", action="store_true", help="Rulează cu UI vizibil (debug).")
    ap.add_argument("--print", action="store_true", help="Arată un preview în terminal.")
    args = ap.parse_args()

    dfs, page_html = scrape_with_selenium(args.url, headless=not args.no_headless, wait_secs=25)
    if not dfs:
        print("Nu am găsit tabele. Posibil ca site-ul să blocheze accesul sau să întârzie încărcarea.")
        sys.exit(2)

    report_date = guess_report_date(page_html)
    csv_dir = Path(args.out_dir) / report_date.isoformat()
    out_paths = save_csvs(dfs, csv_dir, report_date)
    save_sqlite(dfs, Path(args.sqlite), report_date)

    print(f"[OK] Găsite {len(dfs)} tabele. Salvate CSV în: {csv_dir}")
    for p in out_paths:
        print(f"  - {p}")
    print(f"[OK] Baza SQLite actualizată: {args.sqlite}")

    if args.print:
        for i, df in enumerate(dfs, 1):
            print("\n" + "="*80)
            print(f"Tabel #{i}  (primele 8 rânduri)")
            print(df.head(8).to_string(index=False))

if __name__ == "__main__":
    main()
