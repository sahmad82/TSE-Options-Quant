from typing import Dict, Literal
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _d1_d2(
    spot: np.ndarray,
    strike: np.ndarray,
    t_years: np.ndarray,
    r: float,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """محاسبه پارامترهای d1 و d2 در مدل بلک-شولز."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t_years = np.maximum(np.asarray(t_years, dtype=float), 1e-8)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)

    sqrt_t = np.sqrt(t_years)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bs_price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    t_years: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
) -> float | np.ndarray:
    """محاسبه قیمت تئوری اختیار خرید (Call) یا فروش (Put) اروپایی."""
    d1, d2 = _d1_d2(spot, strike, t_years, r, sigma)
    discount = np.exp(-r * np.maximum(t_years, 0.0))

    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    elif option_type == "put":
        price = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(price) if np.ndim(price) == 0 else price


def bs_greeks(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    t_years: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
) -> Dict[str, float | np.ndarray]:
    """
    محاسبه کامل یونانی‌های اختیار معامله:
    - Delta: حساسیت قیمت آپشن به تغییر ۱ ریالی دارایی پایه
    - Gamma: نرخ تغییر دلتا
    - Vega: تغییر قیمت آپشن به ازای ۱٪ تغییر در نوسان‌پذیری
    - Theta: افت ارزش زمانی روزانه (تقسیم بر ۳۶۵)
    - Rho: حساسیت به ۱٪ تغییر در نرخ بهره بدون ریسک
    """
    t_safe = np.maximum(np.asarray(t_years, dtype=float), 1e-8)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    sqrt_t = np.sqrt(t_safe)

    d1, d2 = _d1_d2(spot, strike, t_safe, r, sigma_safe)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * t_safe)

    gamma = pdf_d1 / (spot * sigma_safe * sqrt_t)
    vega = (spot * pdf_d1 * sqrt_t) / 100.0  # per 1% volatility change

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_annual = -(spot * pdf_d1 * sigma_safe) / (2.0 * sqrt_t) - r * strike * discount * norm.cdf(d2)
        rho = (strike * t_safe * discount * norm.cdf(d2)) / 100.0
    elif option_type == "put":
        delta = norm.cdf(d1) - 1.0
        theta_annual = -(spot * pdf_d1 * sigma_safe) / (2.0 * sqrt_t) + r * strike * discount * norm.cdf(-d2)
        rho = (-strike * t_safe * discount * norm.cdf(-d2)) / 100.0
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    theta_daily = theta_annual / 365.0

    def _fmt(val):
        return float(val) if np.ndim(val) == 0 else val

    return {
        "delta": _fmt(delta),
        "gamma": _fmt(gamma),
        "vega": _fmt(vega),
        "theta_daily": _fmt(theta_daily),
        "rho": _fmt(rho),
    }


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    option_type: OptionType = "call",
) -> float:
    """
    استخراج نوسان‌پذیری ضمنی (Implied Volatility) از قیمت بازار با الگوریتم Brent.
    در صورت نبود ارزش زمانی (قیمت بازار کمتر از ارزش ذاتی)، مقدار NaN برمی‌گرداند.
    """
    if market_price <= 0 or t_years <= 0 or spot <= 0 or strike <= 0:
        return float("nan")

    discount = np.exp(-r * t_years)
    intrinsic = (
        max(0.0, spot - strike * discount)
        if option_type == "call"
        else max(0.0, strike * discount - spot)
    )

    if market_price <= intrinsic + 1e-6:
        return float("nan")

    def objective(vol: float) -> float:
        return float(bs_price(spot, strike, t_years, r, vol, option_type)) - market_price

    try:
        iv = brentq(objective, a=1e-4, b=5.0, xtol=1e-6, maxiter=100)
        return float(iv)
    except ValueError:
        return float("nan")