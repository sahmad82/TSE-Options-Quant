import numpy as np
from src.black_scholes import bs_greeks, bs_price, implied_volatility


def test_put_call_parity():
    """بررسی برقراری رابطه بنیادی C - P = S - K * exp(-rT)."""
    s, k, t, r, sigma = 15000.0, 14500.0, 0.25, 0.30, 0.45
    call_p = bs_price(s, k, t, r, sigma, "call")
    put_p = bs_price(s, k, t, r, sigma, "put")

    lhs = call_p - put_p
    rhs = s - k * np.exp(-r * t)
    assert np.isclose(lhs, rhs, atol=1e-6)


def test_implied_volatility_round_trip():
    """بررسی بازیابی دقیق نوسان‌پذیری اولیه از قیمت تئوری."""
    s, k, t, r, true_vol = 20000.0, 22000.0, 60 / 365.0, 0.32, 0.68
    market_c = bs_price(s, k, t, r, true_vol, "call")
    recovered_iv = implied_volatility(market_c, s, k, t, r, "call")
    assert np.isclose(recovered_iv, true_vol, atol=1e-5)


def test_greeks_bounds():
    """بررسی محدوده منطقی دلتا و مثبت بودن گاما و وگا."""
    g_call = bs_greeks(10000, 10000, 0.5, 0.30, 0.40, "call")
    g_put = bs_greeks(10000, 10000, 0.5, 0.30, 0.40, "put")

    assert 0.0 < g_call["delta"] < 1.0
    assert -1.0 < g_put["delta"] < 0.0
    assert g_call["gamma"] > 0.0
    assert g_call["vega"] > 0.0