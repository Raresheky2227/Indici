#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Indici Zilnici Energie & Valută – OPCOM din CSV (fără Selenium)
- BRM logică veche + regex robust.
- OPCOM medii: dacă p03 e în RON îl folosesc direct; dacă e în EUR îl convertesc cu EUR/RON;
  dacă p03 lipsește, calculez din p02 (ponderat cu volum dacă există, altfel media simplă).
- Include în e-mail și tabelul „opcom_table_03_YYYY-MM-DD.csv” exact ca atare.

Setări utile (opționale):
  OPCOM_CSV_ROOT="C:\\Users\\Raresheky\\Desktop\\work\\Indici\\data\\opcom_csv"
  INDICI_DRY_RUN=1   # nu trimite mail, scrie indici_preview.html

Dependențe: requests, pandas, beautifulsoup4, lxml, resend
"""

import os, re, sys, io, json, ssl, time, glob, logging, asyncio, datetime as dt
from typing import Optional, Dict, Any, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup

# -------------------- Config --------------------

# Destinatari: implicit trimite la raresheky@gmail.com; poți suprascrie cu INDICI_TO="a@x.com,b@y.com"
TO = os.getenv("INDICI_TO", "raresheky@gmail.com").split(",")

# Resend
# Cheia API din env: RESEND_API (sau RESEND_API_KEY)
# Expeditorul trebuie să fie pe un domeniu verificat în Resend.
RESEND_FROM = os.getenv("RESEND_FROM", "noreply@dinergyai.com")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IndiciBot/3.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ro-RO;q=0.8",
}
REQ_TIMEOUT: Tuple[int, int] = (20, 50)

TODAY = dt.date.today()
SUBJECT = f"Indici Zilnici Energie și Valută – {TODAY.isoformat()}"

DEBUG = os.getenv("INDICI_DEBUG", "0") == "1"


# -------------------- Utils --------------------

def setup_log():
    level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

def safe_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("\xa0", " ").replace(" ", "").replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def fmt(v, unit=""):
    return "N/A" if v is None else f"{v:.2f}{(' ' + unit) if unit else ''}"

def http_get(url: str, headers: dict = None) -> requests.Response:
    r = requests.get(url, headers=headers or HEADERS, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r

def _detect_currency_from_header(colname: str) -> str:
    """Return 'EUR' sau 'RON' dacă găsește în denumirea coloanei."""
    c = (colname or "").upper()
    if "EUR/MWH" in c or "EURO/MWH" in c or "EUR" in c:
        return "EUR"
    if "RON/MWH" in c or "LEI/MWH" in c or "RON" in c or "LEI" in c:
        return "RON"
    return ""  # necunoscut / nementionat

def _html_table(df: pd.DataFrame, caption: str = "") -> str:
    if df is None or df.empty:
        return ""
    # normalizează numele de coloane pentru lizibilitate
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    tbl = df2.to_html(index=False, border=0, classes="tbl03", escape=False)
    cap = f"<div class='muted' style='margin:6px 0 2px'>{caption}</div>" if caption else ""
    return f"""
{cap}
<div style="overflow:auto; border:1px solid #eee; border-radius:6px;">
{tbl}
</div>
<style>
.tbl03 {{ width:100%; border-collapse:collapse; font-size:14px; }}
.tbl03 th, .tbl03 td {{ border-bottom:1px solid #f0f0f0; padding:6px 8px; text-align:left; }}
.tbl03 th {{ background:#fafafa; }}
</style>
"""

# -------------------- Surse: BNR / TTF / BRM --------------------

def fetch_curs_bnr_eur() -> Dict[str, Any]:
    try:
        r = http_get("https://www.cursbnr.ro/")
        soup = BeautifulSoup(r.text, "html.parser")
        eur = None
        for tr in soup.select("table tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if not cells:
                continue
            rowtxt = " ".join(cells)
            if re.search(r"\b(EUR|Euro)\b", rowtxt, re.I):
                for c in cells:
                    v = safe_float(c)
                    if v and v > 1:
                        eur = v
                        break
        return {"eur_ron": eur}
    except Exception:
        return {"eur_ron": None}

def fetch_ttf() -> Dict[str, Any]:
    """
    Extrage EU Natural Gas (TTF) în EUR/MWh din TradingEconomics, cu regex ancorat
    pe etichetele Price/Last și filtre împotriva valorilor de tip High/Low/52W.
    Acceptă override din env: INDICI_TTF_OVERRIDE.
    """
    override = os.getenv("INDICI_TTF_OVERRIDE", "").strip()
    if override:
        v = safe_float(override)
        return {"ttf_eur_mwh": v if v is not None else None}

    url = "https://tradingeconomics.com/commodity/eu-natural-gas"

    def plausible(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        return x if 5.0 <= x <= 60.0 else None

    try:
        r = http_get(url)
        txt = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    except Exception:
        return {"ttf_eur_mwh": None}

    candidates: list[Optional[float]] = []

    for label in ("Price", "Last"):
        m = re.search(rf"(?i)\b{label}\b[^0-9\-]{{0,10}}([0-9]+[.,][0-9]{{1,2}})\b(?!\s*(?:High|Low|52))", txt)
        if m:
            candidates.append(safe_float(m.group(1)))

    m = re.search(r"(?i)\b(EU|Dutch)\s*Natural\s*Gas\b.*?([0-9]+[.,][0-9]{1,2})\s*(?:EUR|€)\s*/?\s*MWh", txt)
    if m:
        candidates.append(safe_float(m.group(2)))

    for m in re.finditer(r"([0-9]+[.,][0-9]{1,2})", txt):
        span = txt[m.start():m.end()+8]
        if not re.search(r"(?i)(High|Low|52)", span):
            candidates.append(safe_float(m.group(1)))

    for c in candidates:
        v = plausible(c)
        if v is not None:
            return {"ttf_eur_mwh": v}

    return {"ttf_eur_mwh": None}


async def _render(url: str, wait_selector: Optional[str] = None, wait_ms: int = 1000, actions=None) -> str:
    exe = os.getenv("PYPPETEER_EXECUTABLE_PATH", "").strip()
    if not exe:
        raise RuntimeError("PYPPETEER_EXECUTABLE_PATH nu este setat.")
    from pyppeteer import launch

    browser = await launch(
        headless=True,
        executablePath=exe,
        args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage","--no-first-run","--no-default-browser-check"],
        handleSIGINT=False, handleSIGTERM=False, handleSIGHUP=False
    )
    try:
        page = await browser.newPage()
        await page.setUserAgent(HEADERS["User-Agent"])
        await page.setExtraHTTPHeaders({"Accept-Language": HEADERS["Accept-Language"]})
        await page.goto(url, {"waitUntil":"networkidle2", "timeout": 60000})
        if actions:
            await actions(page)
        if wait_selector:
            try:
                await page.waitForSelector(wait_selector, {"timeout": 15000})
            except Exception:
                pass
        await asyncio.sleep(wait_ms/1000)
        return await page.content()
    finally:
        await browser.close()

def render_js(url: str, wait_selector: Optional[str] = None, actions=None) -> str:
    try:
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())  # type: ignore[attr-defined]
            except Exception:
                pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_render(url, wait_selector, actions=actions))
    except Exception as e:
        logging.error("Randare JS eșuată pentru %s: %s. Fallback HTML simplu.", url, e)
        try:
            return http_get(url).text
        except Exception:
            return ""

def fetch_brm_spot_latest() -> Dict[str, Any]:
    try:
        html = render_js("https://brm.ro/piata-spot-gn/", wait_selector="table")
        if not html:
            return {"brm_product": None, "brm_price": None}
        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:
            tables = []
        price = product = None
        for df in tables:
            df.columns = [str(c).strip() for c in df.columns]
            mask = df.apply(lambda row: row.astype(str).str.contains(r"BRMGAS_DA", case=False).any(), axis=1)
            if mask.any():
                sub = df[mask].copy()
                cand = [c for c in df.columns if re.search(r"(Pre[tț]|Ultim|Price|Last)", str(c), re.I)]
                if not cand:
                    cand = [c for c in df.columns if df[c].apply(safe_float).notna().sum() > 0]
                if not sub.empty and cand:
                    product = str(sub.iloc[0, 0])
                    for c in cand:
                        val = safe_float(sub.iloc[0][c])
                        if val is not None:
                            price = val; break
                break
        if price is None:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            m = re.search(r"(BRMGAS_DA\w*)[^0-9]{0,40}([0-9]+[.,][0-9]+)", text, re.I)
            if m:
                product = product or m.group(1)
                price = safe_float(m.group(2))
        return {"brm_product": product, "brm_price": price}
    except Exception as e:
        logging.exception("Eroare BRM Spot: %s", e)
        return {"brm_product": None, "brm_price": None}


def _mk_brm_code(kind: str, d: dt.date) -> str:
    return f"BRMGAS_{kind}{d.day:02d}_M{d.month:02d}_{d.year}"

def _pick_col(cols, *patterns):
    for pat in patterns:
        for c in cols:
            if re.search(pat, str(c), flags=re.I):
                return c
    return None

def fetch_brm_cotatii_for_date(target_date: dt.date) -> Dict[str, Dict[str, Optional[float]]]:
    html = render_js("https://brm.ro/piata-spot-gn/", wait_selector="table")
    if not html:
        return {'DA': {'pmp': None, 'min': None, 'max': None},
                'WD': {'pmp': None, 'min': None, 'max': None}}

    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        tables = []

    code_da = _mk_brm_code("DA", target_date)
    code_wd = _mk_brm_code("WD", target_date)

    result = {'DA': {'pmp': None, 'min': None, 'max': None},
              'WD': {'pmp': None, 'min': None, 'max': None}}

    def safe_number(x):
        return safe_float(x)

    for df in tables:
        df.columns = [str(c).strip() for c in df.columns]
        first_text_col = None
        for c in df.columns:
            if df[c].astype(str).str.contains(r"BRMGAS_", case=False, na=False).any():
                first_text_col = c
                break
        if first_text_col is None:
            first_text_col = df.columns[0]

        col_pmp = _pick_col(
            df.columns,
            r"pret\s*mediu\s*ponderat", r"preț\s*mediu\s*ponderat",
            r"\bPMP\b", r"ponderat\s*lei/?mwh"
        )
        col_min = _pick_col(
            df.columns,
            r"pret\s*minim", r"preț\s*minim",
            r"minim\s*tranzactionat", r"minim\s*tranzacționat"
        )
        col_max = _pick_col(
            df.columns,
            r"pret\s*maxim", r"preț\s*maxim",
            r"maxim\s*tranzactionat", r"maxim\s*tranzacționat"
        )

        if col_pmp is None:
            col_pmp = _pick_col(df.columns, r"pret\s*mediu", r"preț\s*mediu")

        for kind, code in (("DA", code_da), ("WD", code_wd)):
            mask = df[first_text_col].astype(str).str.contains(re.escape(code), case=False, na=False)
            if not mask.any():
                continue
            row = df[mask].iloc[0]
            if col_pmp and pd.notna(row.get(col_pmp)):
                result[kind]['pmp'] = safe_number(row[col_pmp])
            if col_min and pd.notna(row.get(col_min)):
                result[kind]['min'] = safe_number(row[col_min])
            if col_max and pd.notna(row.get(col_max)):
                result[kind]['max'] = safe_number(row[col_max])

    return result


# -------------------- OPCOM din CSV --------------------

def _csv_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    default_root = os.path.join(here, "data", "opcom_csv")
    return os.getenv("OPCOM_CSV_ROOT", default_root)

def _find_csv_set(date_str: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    base = _csv_root()
    day_dir = os.path.join(base, date_str)
    p01 = os.path.join(day_dir, f"opcom_table_01_{date_str}.csv")
    p02 = os.path.join(day_dir, f"opcom_table_02_{date_str}.csv")
    p03 = os.path.join(day_dir, f"opcom_table_03_{date_str}.csv")
    return (p01 if os.path.isfile(p01) else None,
            p02 if os.path.isfile(p02) else None,
            p03 if os.path.isfile(p03) else None)

def _latest_available_date() -> Optional[str]:
    base = _csv_root()
    if not os.path.isdir(base):
        return None
    dirs = [d for d in os.listdir(base) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) and os.path.isdir(os.path.join(base, d))]
    return sorted(dirs)[-1] if dirs else None

def _weighted_or_mean(price: pd.Series, volume: Optional[pd.Series]) -> Optional[float]:
    p = pd.to_numeric(price, errors="coerce").dropna()
    if p.empty:
        return None
    if volume is None:
        return float(p.mean())
    v = pd.to_numeric(volume, errors="coerce").fillna(0)
    denom = v.sum()
    if denom and not pd.isna(denom):
        return float((p * v).sum() / denom)
    return float(p.mean())

def load_opcom_from_csvs(target_date: Optional[dt.date] = None):
    d = (target_date or dt.date.today()).isoformat()
    p01, p02, p03 = _find_csv_set(d)
    chosen = d

    if not (p02 or p03):
        last = _latest_available_date()
        if last:
            p01, p02, p03 = _find_csv_set(last)
            chosen = last

    hourly_df = None
    averages: Dict[str, float] = {}
    raw03_df = None

    if p02:
        df02 = pd.read_csv(p02)
        hour_col  = next((c for c in df02.columns if re.search(r"\b(interval|hour|ora)\b", c, re.I)), None)
        price_col = next((c for c in df02.columns if re.search(r"(ROPEX_DAM_H|\bprice\b|Pre[tț])", c, re.I)), None)
        vol_col   = next((c for c in df02.columns if re.search(r"(Traded\s*Volume|volume|MWh)", c, re.I)), None)

        if hour_col and price_col:
            hours = pd.to_numeric(df02[hour_col].astype(str).str.extract(r"(\d{1,2})")[0], errors="coerce")
            prices = df02[price_col].apply(safe_float)
            vols = df02[vol_col].apply(safe_float) if vol_col else None

            hourly_df = pd.DataFrame({
                "Hour": hours,
                "Price": prices,
                "Volume": vols if vol_col else None
            }).dropna(subset=["Hour", "Price"])
            hourly_df = hourly_df[(hourly_df["Hour"] >= 1) & (hourly_df["Hour"] <= 24)].sort_values("Hour").reset_index(drop=True)

    if p03:
        df03 = pd.read_csv(p03)
        raw03_df = df03.copy()
        name_col = df03.columns[0]

        candidate_cols = []
        for c in df03.columns[1:]:
            if re.search(r"ROPEX\s*_?DAM|Price|Pret|Preț", str(c), re.I):
                candidate_cols.append(c)
        if not candidate_cols:
            for c in df03.columns[1:]:
                s = pd.to_numeric(df03[c], errors="coerce")
                if s.notna().sum() >= 2:
                    candidate_cols.append(c); break

        price_col = candidate_cols[0] if candidate_cols else None
        if price_col:
            p03_currency = _detect_currency_from_header(price_col)

            def pick(label_regex):
                m = df03[name_col].astype(str).str.contains(label_regex, case=False, regex=True)
                if m.any():
                    return safe_float(df03.loc[m, price_col].iloc[0])
                return None

            base_v = pick(r"\bBase\b")
            peak_v = pick(r"\bPeak\b")
            off_v  = pick(r"Off[_\s-]*peak|Off\s*peak")

            if p03_currency == "RON":
                averages["base_avg_ron"] = base_v
                averages["peak_avg_ron"] = peak_v
                averages["off_avg_ron"]  = off_v
            else:
                averages["base_avg_eur"] = base_v
                averages["peak_avg_eur"] = peak_v
                averages["off_avg_eur"]  = off_v

    if hourly_df is not None and not averages:
        base = hourly_df[(hourly_df["Hour"] >= 1) & (hourly_df["Hour"] <= 24)]
        peak = hourly_df[(hourly_df["Hour"] >= 9) & (hourly_df["Hour"] <= 20)]
        off  = hourly_df[((hourly_df["Hour"] >= 1) & (hourly_df["Hour"] <= 8)) |
                         ((hourly_df["Hour"] >= 21) & (hourly_df["Hour"] <= 24))]
        v = hourly_df["Volume"] if "Volume" in hourly_df and hourly_df["Volume"] is not None else None
        averages = {
            "base_avg_ron": _weighted_or_mean(base["Price"], v.loc[base.index] if v is not None else None),
            "peak_avg_ron": _weighted_or_mean(peak["Price"], v.loc[peak.index] if v is not None else None),
            "off_avg_ron":  _weighted_or_mean(off["Price"],  v.loc[off.index]  if v is not None else None),
        }

    tag = f"csv:{chosen}" if (hourly_df is not None or averages) else "csv-missing"
    return hourly_df, averages, tag, raw03_df


# -------------------- Calcul & HTML --------------------

def compute_ropex(hourly: Optional[pd.DataFrame], averages: Dict[str, float], eur_ron: Optional[float]) -> Dict[str, Any]:
    have_eur = any(k.endswith("_eur") and averages.get(k) is not None for k in averages.keys())
    have_ron = any(k.endswith("_ron") and averages.get(k) is not None for k in averages.keys())

    def to_ron_from_eur(x):
        if x is None:
            return None
        if eur_ron is None:
            return None
        return float(x) * float(eur_ron)

    base_avg = peak_avg = off_avg = None
    if have_ron:
        base_avg = averages.get("base_avg_ron")
        peak_avg = averages.get("peak_avg_ron")
        off_avg  = averages.get("off_avg_ron")
    elif have_eur:
        base_avg = to_ron_from_eur(averages.get("base_avg_eur"))
        peak_avg = to_ron_from_eur(averages.get("peak_avg_eur"))
        off_avg  = to_ron_from_eur(averages.get("off_avg_eur"))

    def stats(hmin, hmax):
        if hourly is None or hourly.empty:
            return None, None
        sub = hourly[(hourly["Hour"] >= hmin) & (hourly["Hour"] <= hmax)]
        if sub.empty:
            return None, None
        return float(sub["Price"].min()), float(sub["Price"].max())

    base_min, base_max = stats(1, 24)
    peak_min, peak_max = stats(9, 20)
    off_min1, off_max1 = stats(1, 8)
    off_min2, off_max2 = stats(21, 24)
    off_min = min([v for v in [off_min1, off_min2] if v is not None], default=None)
    off_max = max([v for v in [off_max1, off_max2] if v is not None], default=None)

    return {
        "base_avg": base_avg,
        "base_min": base_min, "base_max": base_max,
        "peak_avg": peak_avg,
        "peak_min": peak_min, "peak_max": peak_max,
        "off_avg":  off_avg,
        "off_min":  off_min, "off_max":  off_max,
    }

def build_html(
    curs,
    ttf,
    brm,
    ropex,
    hourly_rows=0,
    opcom_source: str = "csv",
    opcom03_html: str = "",
    brm_cotatii: dict | None = None,
    brm_codes: tuple[str, str] | None = None,
) -> str:
    da_code, _ = brm_codes if brm_codes else (_mk_brm_code("DA", TODAY), _mk_brm_code("WD", TODAY))
    cot = brm_cotatii or {}
    da = cot.get("DA") or {}

    eur_ron_raw = curs.get("eur_ron")
    ttf_eur_raw = ttf.get("ttf_eur_mwh")
    eur_ron_disp = None if eur_ron_raw is None else round(float(eur_ron_raw), 2)
    ttf_eur_disp = None if ttf_eur_raw is None else round(float(ttf_eur_raw), 2)

    def margin_cell(brm_ron: float | None) -> str:
        if brm_ron is None or eur_ron_disp is None or ttf_eur_disp is None:
            return "N/A"
        margin_ron = round(brm_ron - (ttf_eur_disp * eur_ron_disp), 2)
        brm_eur = round(brm_ron / eur_ron_disp, 2)
        delta_eur = round(brm_eur - ttf_eur_disp, 2)
        sign = "+" if delta_eur >= 0 else "−"
        return f"{margin_ron:.2f} RON/MWh <span class='muted'>(TTF {sign} {abs(delta_eur):.2f} EUR ⇒ {brm_eur:.2f} EUR)</span>"

    return f"""
<html><head><meta charset="utf-8"/>
<title>{SUBJECT}</title>
<style>
body {{ font-family: Arial, sans-serif; color:#111; }}
h1 {{ font-size: 18px; }}
h2 {{ font-size: 16px; margin: 16px 0 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 18px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
th {{ background: #f4f4f6; text-align: left; }}
.muted {{ color:#666; font-size:12px; }}
.chip {{ display:inline-block; padding:2px 8px; border-radius:12px; background:#f1f5f9; }}
</style></head><body>
<h1>Indici Zilnici Energie și Valută – {TODAY:%Y-%m-%d}</h1>

<table>
<tr><th>Indicator</th><th>Valoare</th><th>Unitate</th><th>Observații</th></tr>
<tr><td>Curs BNR EUR/RON</td><td>{fmt(eur_ron_disp)}</td><td>RON/EUR</td><td class="muted">sursa: cursbnr.ro</td></tr>
<tr><td>EU Natural Gas TTF</td><td>{fmt(ttf_eur_disp)}</td><td>EUR/MWh</td><td class="muted">sursa: tradingeconomics.com/commodity/eu-natural-gas</td></tr>
</table>

<h2>BRM – Cotații (ziua curentă)</h2>
<table>
  <tr>
    <th>Produs</th>
    <th>Preț mediu ponderat</th>
    <th>Preț minim tranzacționat</th>
    <th>Preț maxim tranzacționat</th>
  </tr>
  <tr>
    <td>{da_code}</td>
    <td>{fmt(da.get("pmp"), "RON/MWh")}</td>
    <td>{fmt(da.get("min"), "RON/MWh")}</td>
    <td>{fmt(da.get("max"), "RON/MWh")}</td>
  </tr>
  <tr>
    <td>Margine</td>
    <td>{margin_cell(da.get("pmp"))}</td>
    <td>{margin_cell(da.get("min"))}</td>
    <td>{margin_cell(da.get("max"))}</td>
  </tr>
</table>

{ (f"<h2>OPCOM – Agregate (fișier 03)</h2>{opcom03_html}" if opcom03_html else "") }
</body></html>
"""


def send_mail(html_body: str):
    # Dry-run: nu trimite, doar salvează HTML-ul
    if os.getenv("INDICI_DRY_RUN", "0") == "1":
        with open("indici_preview.html", "w", encoding="utf-8") as f:
            f.write(html_body)
        logging.info("Dry-run: am scris indici_preview.html")
        return

    api_key = os.getenv("RESEND_API") or os.getenv("RESEND_API_KEY")
    if not api_key:
        raise RuntimeError("Setează RESEND_API (sau RESEND_API_KEY) cu cheia Resend.")

    if not RESEND_FROM:
        raise RuntimeError("Setează RESEND_FROM (expeditor pe un domeniu verificat în Resend).")

    try:
        import resend
    except ImportError as e:
        raise RuntimeError("Lipsește pachetul 'resend'. Instalează cu: pip install resend") from e

    resend.api_key = api_key

    payload = {
        "from": RESEND_FROM,
        "to": TO,
        "subject": SUBJECT,
        "html": html_body,
    }

    try:
        resp = resend.Emails.send(payload)
        # resp may be a dict or object with 'id'
        msg_id = getattr(resp, "id", None) or (resp.get("id") if isinstance(resp, dict) else "N/A")
        logging.info("Email trimis via Resend către: %s (id: %s)", ", ".join(TO), msg_id)
    except Exception as e:
        logging.exception("Eroare la trimiterea e-mailului via Resend: %s", e)
        raise


# -------------------- Main --------------------

def main():
    setup_log()
    logging.info("Colectez date...")

    curs = fetch_curs_bnr_eur()
    ttf = fetch_ttf()
    brm = fetch_brm_spot_latest()
    brm_codes = (_mk_brm_code("DA", TODAY), _mk_brm_code("WD", TODAY))
    brm_cotatii = fetch_brm_cotatii_for_date(TODAY)

    hourly, avgs, src_tag, raw03_df = load_opcom_from_csvs(TODAY)

    # HTML „ca atare” pentru p03 (dacă există)
    opcom03_html = ""
    try:
        if raw03_df is not None and not raw03_df.empty:
            d = TODAY.isoformat()
            _, _, p03_path_today = _find_csv_set(d)
            caption = os.path.basename(p03_path_today) if p03_path_today else "opcom_table_03.csv"
            opcom03_html = _html_table(raw03_df, caption=caption)
        else:
            last = _latest_available_date()
            if last:
                _, _, p03_last = _find_csv_set(last)
                if p03_last and os.path.isfile(p03_last):
                    df_last = pd.read_csv(p03_last)
                    opcom03_html = _html_table(df_last, caption=os.path.basename(p03_last))
    except Exception as e:
        logging.warning("Nu am putut genera HTML pentru p03: %s", e)

    if hourly is None and not avgs:
        logging.warning("OPCOM: CSV-urile nu au fost găsite (root: %s) – las N/A.", _csv_root())
        rows = 0
        ropex = compute_ropex(None, {}, curs.get("eur_ron"))
    else:
        rows = 0 if hourly is None else len(hourly)
        ropex = compute_ropex(hourly, avgs, curs.get("eur_ron"))

    html = build_html(
        curs, ttf, brm, ropex,
        hourly_rows=rows,
        opcom_source=src_tag,
        opcom03_html=opcom03_html,
        brm_cotatii=brm_cotatii,
        brm_codes=brm_codes
    )

    logging.info("Trimit email către: %s", ", ".join(TO) if os.getenv("INDICI_DRY_RUN","0")!="1" else "(dry-run)")
    send_mail(html)
    logging.info("GATA ✔️")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Eșec rulare: %s", e)
        sys.exit(1)
