from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

OPTION_TITLE_PATTERN = re.compile(
    r"اختیار\s*([خف])\s+(.+?)\s*-\s*(\d+)\s*-\s*(\d{4}/\d{2}/\d{2})"
)


def normalize_persian(text: str) -> str:
    """یکسان‌سازی کاراکترهای عربی و فارسی در خروجی‌های TSETMC."""
    return text.replace("ي", "ی").replace("ك", "ک").strip()


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> datetime.date:
    """تبدیل دقیق تاریخ شمسی به میلادی بدون وابستگی به کتابخانه خارجی."""
    jy += 1595
    days = (
        -355668
        + (365 * jy)
        + ((jy // 33) * 8)
        + (((jy % 33) + 3) // 4)
        + jd
        + ((jm - 1) * 31 if jm < 7 else ((jm - 7) * 30) + 186)
    )
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    sal_a = [
        0,
        31,
        29 if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)) else 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ]
    gm = 0
    while gm < 13 and gd > sal_a[gm]:
        gd -= sal_a[gm]
        gm += 1
    return datetime.date(gy, gm, gd)


def parse_option_title(lval30: str, ref_date: Optional[datetime.date] = None) -> Optional[Dict]:
    """استخراج نوع آپشن، قیمت اعمال و زمان باقی‌مانده تا سررسید از نام کامل قرارداد."""
    clean_title = normalize_persian(lval30)
    match = OPTION_TITLE_PATTERN.search(clean_title)
    if not match:
        return None

    opt_char, underlying_name, strike_str, exp_jalali = match.groups()
    option_type = "call" if opt_char == "خ" else "put"
    strike = float(strike_str)

    jy, jm, jd = map(int, exp_jalali.split("/"))
    exp_date = jalali_to_gregorian(jy, jm, jd)
    today = ref_date or datetime.date.today()
    days_to_exp = (exp_date - today).days

    return {
        "option_type": option_type,
        "underlying_parsed": underlying_name.strip(),
        "strike": strike,
        "exp_jalali": exp_jalali,
        "exp_date": exp_date.isoformat(),
        "days_to_exp": days_to_exp,
        "t_years": max(days_to_exp / 365.0, 0.0),
    }


def _fetch_contract_quote(ins_code: str, session: requests.Session) -> Dict:
    """دریافت قیمت پایانی، آخرین معامله و بهترین مظنه خرید/فروش یک قرارداد."""
    quote = {
        "last_price": 0.0,
        "close_price": 0.0,
        "volume": 0,
        "bid": 0.0,
        "ask": 0.0,
    }
    try:
        p_url = f"https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceInfo/{ins_code}"
        p_resp = session.get(p_url, headers=HEADERS, timeout=12)
        if p_resp.ok:
            info = p_resp.json().get("closingPriceInfo", {})
            quote["last_price"] = float(info.get("pDrCotVal") or 0.0)
            quote["close_price"] = float(info.get("pClosing") or 0.0)
            quote["volume"] = int(info.get("qTotTran5J") or 0)

        b_url = f"https://cdn.tsetmc.com/api/BestLimits/{ins_code}"
        b_resp = session.get(b_url, headers=HEADERS, timeout=12)
        if b_resp.ok:
            limits = b_resp.json().get("bestLimits", [])
            for row in limits:
                if row.get("number") == 1:
                    quote["bid"] = float(row.get("pMeDem") or 0.0)
                    quote["ask"] = float(row.get("pMeOf") or 0.0)
                    break
    except Exception:
        pass
    return quote


def _resolve_underlying_spot(ticker: str, session: requests.Session) -> Tuple[str, float]:
    """پیدا کردن کد ۱۷ رقمی و قیمت لحظه‌ای دارایی پایه."""
    norm_ticker = normalize_persian(ticker)
    url = f"https://cdn.tsetmc.com/api/Instrument/GetInstrumentSearch/{norm_ticker}"
    resp = session.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("instrumentSearch", [])

    ins_code = None
    for item in items:
        if normalize_persian(item.get("lVal18AFC", "")) == norm_ticker and item.get("flow") in (300, 303, 309):
            ins_code = str(item["insCode"])
            break
    if not ins_code and items:
        ins_code = str(items[0]["insCode"])

    if not ins_code:
        raise ValueError(f"Underlying ticker '{ticker}' not found on TSETMC.")

    quote = _fetch_contract_quote(ins_code, session)
    spot = quote["last_price"] if quote["last_price"] > 0 else quote["close_price"]
    return ins_code, spot


def fetch_options_chain(
    underlying_ticker: str = "اهرم",
    cache_dir: str = "data",
    force_refresh: bool = False,
) -> Tuple[float, pd.DataFrame]:
    """
    دریافت زنجیره کامل اختیار خرید و فروش برای یک دارایی پایه از بورس تهران.
    خروجی: (قیمت دارایی پایه, دیتافریم قراردادهای فعال)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"options_{underlying_ticker}.csv")

    if os.path.exists(cache_path) and not force_refresh:
        df_cached = pd.read_csv(cache_path)
        spot_cached = float(df_cached["spot"].iloc[0])
        return spot_cached, df_cached

    norm_ticker = normalize_persian(underlying_ticker)
    contracts: Dict[str, Dict] = {}

    with requests.Session() as session:
        _, spot = _resolve_underlying_spot(norm_ticker, session)

        # جستجوی هم‌زمان قراردادهای اختیار خرید (اختیارخ) و اختیار فروش (اختیارف)
        for prefix in ("اختیارخ", "اختیارف"):
            query = f"{prefix} {norm_ticker}"
            url = f"https://cdn.tsetmc.com/api/Instrument/GetInstrumentSearch/{query}"
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()

            for item in resp.json().get("instrumentSearch", []):
                ins_code = str(item.get("insCode", ""))
                symbol = normalize_persian(item.get("lVal18AFC", ""))
                title = normalize_persian(item.get("lVal30", ""))
                parsed = parse_option_title(title)

                if parsed and parsed["days_to_exp"] > 0 and norm_ticker in parsed["underlying_parsed"]:
                    contracts[ins_code] = {
                        "ins_code": ins_code,
                        "symbol": symbol,
                        "title": title,
                        "spot": spot,
                        **parsed,
                    }

        if not contracts:
            raise RuntimeError(f"No active option contracts found for '{underlying_ticker}'.")

        # دریافت هم‌زمان مظنه و قیمت تمام قراردادها
        with ThreadPoolExecutor(max_workers=10) as pool:
            future_map = {
                pool.submit(_fetch_contract_quote, code, session): code
                for code in contracts
            }
            for future in as_completed(future_map):
                code = future_map[future]
                contracts[code].update(future.result())

    df = pd.DataFrame(list(contracts.values()))
    df = df.sort_values(["exp_date", "option_type", "strike"]).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} active contracts for '{underlying_ticker}' (Spot={spot:,.0f}) -> {cache_path}")
    return spot, df


if __name__ == "__main__":
    spot_price, chain_df = fetch_options_chain("اهرم", force_refresh=True)
    print(f"\nUnderlying Spot Price: {spot_price:,.0f} Rials")
    print(chain_df[["symbol", "option_type", "strike", "exp_jalali", "days_to_exp", "bid", "ask", "close_price"]].head(12).to_string())