import numpy as np
import pandas as pd

from src.black_scholes import bs_greeks, bs_price, implied_volatility
from src.tse_options_loader import fetch_options_chain

# کارمزدهای استاندارد معاملات در بورس تهران
TSE_EQUITY_BUY_FEE = 0.003712   # کارمزد خرید سهام/صندوق (0.3712%)
TSE_OPTION_FEE = 0.00103        # کارمزد معامله اختیار معامله (0.103%)
TSE_EXERCISE_FEE = 0.0005       # کارمزد اعمال قرارداد (0.05%)


def analyze_options_chain(
    chain_df: pd.DataFrame,
    spot: float,
    rf_annual: float = 0.30,
    only_liquid: bool = True,
) -> pd.DataFrame:
    """غنی‌سازی زنجیره آپشن با نوسان‌پذیری ضمنی (IV)، قیمت تئوری بلک-شولز و یونانی‌ها."""
    df = chain_df.copy()

    has_spread = (df["bid"] > 0) & (df["ask"] > 0)
    df["market_price"] = np.where(
        has_spread,
        0.5 * (df["bid"] + df["ask"]),
        np.where(df["last_price"] > 0, df["last_price"], df["close_price"]),
    )

    if only_liquid:
        # حذف قراردادهای غیرنقدشونده و قیمت‌های اسمی ۱ ریالی بدون معامله
        is_liquid = (df["market_price"] > 10.0) & ((df["volume"] > 0) | (df["bid"] > 1.0))
        df = df[is_liquid].copy()
    else:
        df = df[df["market_price"] > 0].copy()

    if df.empty:
        return df

    df["iv"] = [
        implied_volatility(
            market_price=row["market_price"],
            spot=spot,
            strike=row["strike"],
            t_years=row["t_years"],
            r=rf_annual,
            option_type=row["option_type"],
        )
        for _, row in df.iterrows()
    ]

    median_iv = float(df["iv"].dropna().median()) if df["iv"].notna().any() else 0.45
    pricing_vol = df["iv"].fillna(median_iv).to_numpy(dtype=float, copy=True)

    df["bs_price"] = 0.0
    df["delta"] = 0.0
    df["gamma"] = 0.0
    df["vega"] = 0.0
    df["theta_daily"] = 0.0

    for opt_type in ("call", "put"):
        mask = df["option_type"] == opt_type
        if not mask.any():
            continue
        k_arr = df.loc[mask, "strike"].to_numpy(dtype=float)
        t_arr = df.loc[mask, "t_years"].to_numpy(dtype=float)
        v_arr = pricing_vol[mask.to_numpy()]

        df.loc[mask, "bs_price"] = bs_price(spot, k_arr, t_arr, rf_annual, v_arr, opt_type)
        greeks = bs_greeks(spot, k_arr, t_arr, rf_annual, v_arr, opt_type)
        df.loc[mask, "delta"] = greeks["delta"]
        df.loc[mask, "gamma"] = greeks["gamma"]
        df.loc[mask, "vega"] = greeks["vega"]
        df.loc[mask, "theta_daily"] = greeks["theta_daily"]

    df["mispricing_pct"] = ((df["market_price"] - df["bs_price"]) / df["bs_price"]) * 100.0
    df["moneyness"] = spot / df["strike"]
    return df.reset_index(drop=True)


def scan_covered_calls(
    enriched_df: pd.DataFrame,
    spot: float,
    rf_annual: float = 0.30,
) -> pd.DataFrame:
    """اسکنر استراتژی Covered Call با کسر کارمزدهای واقعی بورس تهران."""
    calls = enriched_df[enriched_df["option_type"] == "call"].copy()
    if calls.empty:
        return pd.DataFrame()

    calls["sell_premium"] = np.where(calls["bid"] > 0, calls["bid"], calls["market_price"])
    calls = calls[calls["sell_premium"] > 10.0].copy()

    buy_cost = spot * (1.0 + TSE_EQUITY_BUY_FEE)
    net_premium = calls["sell_premium"] * (1.0 - TSE_OPTION_FEE)
    calls["net_debit"] = buy_cost - net_premium

    calls["breakeven"] = calls["net_debit"]
    calls["downside_protection_pct"] = ((spot - calls["breakeven"]) / spot) * 100.0

    net_strike_received = calls["strike"] * (1.0 - TSE_EXERCISE_FEE)
    calls["max_return_pct"] = ((net_strike_received - calls["net_debit"]) / calls["net_debit"]) * 100.0
    calls["annualized_yield_pct"] = calls["max_return_pct"] * (365.0 / calls["days_to_exp"])
    calls["excess_over_rf_pct"] = calls["annualized_yield_pct"] - (rf_annual * 100.0)

    cols = [
        "symbol", "strike", "exp_jalali", "days_to_exp", "sell_premium",
        "iv", "delta", "breakeven", "downside_protection_pct",
        "max_return_pct", "annualized_yield_pct", "excess_over_rf_pct",
    ]
    return calls[cols].sort_values("annualized_yield_pct", ascending=False).reset_index(drop=True)


def scan_put_call_parity(
    enriched_df: pd.DataFrame,
    spot: float,
    rf_annual: float = 0.30,
) -> pd.DataFrame:
    """بررسی انحراف از Put-Call Parity بر اساس مظنه‌های واقعی Bid/Ask و کارمزد."""
    calls = enriched_df[enriched_df["option_type"] == "call"][
        ["strike", "exp_jalali", "days_to_exp", "t_years", "symbol", "bid", "ask", "market_price"]
    ].rename(columns={"symbol": "call_symbol", "bid": "call_bid", "ask": "call_ask", "market_price": "call_mid"})

    puts = enriched_df[enriched_df["option_type"] == "put"][
        ["strike", "exp_jalali", "symbol", "bid", "ask", "market_price"]
    ].rename(columns={"symbol": "put_symbol", "bid": "put_bid", "ask": "put_ask", "market_price": "put_mid"})

    pairs = pd.merge(calls, puts, on=["strike", "exp_jalali"], how="inner")
    if pairs.empty:
        return pd.DataFrame()

    pairs["lhs_c_minus_p"] = pairs["call_mid"] - pairs["put_mid"]
    pairs["rhs_pv_forward"] = spot - pairs["strike"] * np.exp(-rf_annual * pairs["t_years"])
    pairs["parity_spread"] = pairs["lhs_c_minus_p"] - pairs["rhs_pv_forward"]

    # سود خالص آربیتراژ با لحاظ کردن مظنه خرید/فروش (Bid/Ask) و کارمزدها
    est_fee = spot * (TSE_EQUITY_BUY_FEE + 2 * TSE_OPTION_FEE)
    conv_profit = np.where(
        (pairs["call_bid"] > 0) & (pairs["put_ask"] > 0),
        (pairs["call_bid"] - pairs["put_ask"]) - pairs["rhs_pv_forward"] - est_fee,
        -np.inf,
    )
    rev_profit = np.where(
        (pairs["call_ask"] > 0) & (pairs["put_bid"] > 0),
        pairs["rhs_pv_forward"] - (pairs["call_ask"] - pairs["put_bid"]) - est_fee,
        -np.inf,
    )

    pairs["executable_edge"] = np.maximum(conv_profit, rev_profit)
    pairs["signal"] = np.where(
        pairs["executable_edge"] <= 0,
        "No Executable Arbitrage",
        np.where(conv_profit > rev_profit, "Conversion (Sell Call Bid, Buy Put Ask)", "Reversal (Buy Call Ask, Sell Put Bid)"),
    )

    cols = [
        "strike", "exp_jalali", "days_to_exp", "call_symbol", "call_mid",
        "put_symbol", "put_mid", "parity_spread", "executable_edge", "signal",
    ]
    return pairs[cols].sort_values("parity_spread", key=np.abs, ascending=False).reset_index(drop=True)