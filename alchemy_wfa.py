"""
AlchemySignal V1.5 — Walk-Forward Backtester
NDX100 OR-1 Setup: Break & Retest + PB1 Sweep & Go

Run:  streamlit run alchemy_wfa.py
Data: Export NDX100 M1 from MT4 → Tools → History Center → NDX100,M1 → Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pytz
from itertools import product
import warnings
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlchemySignal V1.5 · Walk-Forward Backtester",
    page_icon="⚗️",
    layout="wide",
)
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX — fully causal, no lookahead."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up   = h.diff();  dn = -l.diff()
    pdm  = np.where((up > dn) & (up > 0), up, 0.0)
    mdm  = np.where((dn > up) & (dn > 0), dn, 0.0)
    alpha = 1 / period
    atr   = tr.ewm(alpha=alpha, adjust=False).mean()
    pdi   = 100 * pd.Series(pdm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-9)
    mdi   = 100 * pd.Series(mdm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-9)
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.ewm(alpha=alpha, adjust=False).mean()


def compute_h1_ema(df_1m: pd.DataFrame, period: int) -> pd.Series:
    """H1 EMA reindexed to 1m bars — forward-fill only (causal)."""
    h1 = df_1m.resample("1h").agg({"Close": "last"}).dropna()
    ema = h1["Close"].ewm(span=period, adjust=False).mean()
    return ema.reindex(df_1m.index, method="ffill")


# ─────────────────────────────────────────────────────────────────────────────
# ALCHEMY SIGNAL V1.5 STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _scan_exit(day_df, entry_ts, direction, sl, tp, sl_pts, tp_pts, lot, dpp, spread_pts=0.0):
    """
    Scan forward bars for SL/TP hit or EOD close.

    spread_pts: broker bid/ask spread in price points.
      - LONG entries are filled at ASK = bar_open + spread, so:
          SL cost = (sl_pts + spread_pts)  ← price moves against you AND you paid spread on entry
          TP gain = (tp_pts - spread_pts)  ← TP reached on BID, but you entered at ASK
      - SHORT entries are filled at BID = bar_open, exit at ASK, same net cost:
          SL cost = (sl_pts + spread_pts)
          TP gain = (tp_pts - spread_pts)
      EOD exit also deducts spread (entered at ASK, exit mark at BID).
    """
    spread_cost = spread_pts * lot * dpp
    profit      = None
    exit_time   = day_df.index[-1]
    exit_reason = "EOD"
    for xt, xb in day_df.loc[day_df.index > entry_ts].iterrows():
        if direction == "LONG":
            if xb["Low"]  <= sl: profit = -(sl_pts + spread_pts) * lot * dpp; exit_reason = "SL"; exit_time = xt; break
            if xb["High"] >= tp: profit =  (tp_pts - spread_pts) * lot * dpp; exit_reason = "TP"; exit_time = xt; break
        else:
            if xb["High"] >= sl: profit = -(sl_pts + spread_pts) * lot * dpp; exit_reason = "SL"; exit_time = xt; break
            if xb["Low"]  <= tp: profit =  (tp_pts - spread_pts) * lot * dpp; exit_reason = "TP"; exit_time = xt; break
    if profit is None:
        last   = day_df["Close"].iloc[-1]
        entry  = day_df.loc[entry_ts, "Open"]
        profit = ((last - entry) if direction == "LONG" else (entry - last)) * lot * dpp - spread_cost
    return profit, exit_time, exit_reason


def run_alchemy(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    Run AlchemySignal V1.5 OR-1 strategy on M1 data.

    Matches AlchemySignal_v1_5.mq4 CheckOREntry() logic:
      Gate 1  — GREEN zone only (09:45–09:54 ET)
      Gate 2  — OR locked from 09:30–09:34 ET (5 M1 bars)
      Gate 3  — One long + one short per day max (or1LongDone / or1ShortDone)
      Gate 4  — Max trades per day (dailyTradeCount)
      Gate 5  — Market state: Bull/Bear Continuation (M1 EMA20 vs VWAP)
      Gate 6  — Sweep: a session bar swept OR lo (for long) or OR hi (for short)
                and closed BACK inside the OR range
      Gate 7  — OR break: last_close > or_hi (long) or < or_lo (short)
      Gate 7a — Retest: |entry - OR level| < or_tol
      Gate 7b — VWAP: last_close > VWAP (long) or < VWAP (short)
      Gate 8  — ADX chop filter (V1.5, optional)
      Gate 9  — Minimum OR range (V1.5)
      Gate 10 — H1 EMA trend alignment (V1.5, optional)
    """
    or_tol     = p.get("or_retest_tolerance", 5.0)   # EA default is 5.0
    min_or     = p.get("min_or_range",        80.0)
    sl_pts     = p.get("sl_points",           24.2)
    tp_pts     = p.get("tp_points",          150.0)
    lot        = p.get("lot_size",             0.30)
    max_tpd    = p.get("max_trades_per_day",      2)
    use_adx    = p.get("use_adx_filter",      False)
    min_adx    = p.get("min_adx",             20.0)
    use_h1     = p.get("use_h1_ema_filter",   False)
    h1_per     = p.get("h1_ema_period",          20)
    spread_pts = p.get("spread_pts",           0.0)   # broker bid/ask spread
    dpp        = 10.0   # NDX100: $10/pt/lot

    ny   = pytz.timezone("America/New_York")
    data = df.copy()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert(ny)

    # ── Pre-compute causal indicators (no lookahead) ────────────────────────
    # M1 EMA20: rising when ema_now > ema_5_bars_ago
    m1_ema20     = data["Close"].ewm(span=20, adjust=False).mean()
    m1_ema20_lag = m1_ema20.shift(5)

    # H1 ADX (Gate 8)
    adx_s = compute_adx(data) if use_adx else None

    # H1 EMA (Gate 10)
    h1_ema_s = compute_h1_ema(data, h1_per) if use_h1 else None
    h1_cls_s = (
        data.resample("1h").agg({"Close": "last"})["Close"]
            .reindex(data.index, method="ffill")
        if use_h1 else None
    )

    # Fast dict-based lookup for hot path
    ema20_d     = m1_ema20.to_dict()
    ema20_lag_d = m1_ema20_lag.to_dict()
    adx_d       = adx_s.to_dict()    if adx_s    is not None else {}
    h1_ema_d    = h1_ema_s.to_dict() if h1_ema_s is not None else {}
    h1_cls_d    = h1_cls_s.to_dict() if h1_cls_s is not None else {}

    trades = []
    prev_close = None

    for day, day_df in data.groupby(data.index.date):

        # ── Session: 09:30 → 15:59 ────────────────────────────────────────
        session = day_df.between_time("09:30", "15:59")
        if session.empty:
            continue

        # ── Gate 2: Lock OR 09:30–09:34 ───────────────────────────────────
        or_bars = session.between_time("09:30", "09:34")
        if or_bars.empty:
            continue
        or_hi  = or_bars["High"].max()
        or_lo  = or_bars["Low"].min()
        or_rng = or_hi - or_lo

        # Session context
        day_of_week = pd.Timestamp(day).weekday()
        gap_pts     = round(float(or_bars["Open"].iloc[0]) - prev_close, 2) \
                      if prev_close is not None else 0.0

        # ── Gate 9: OR range minimum ───────────────────────────────────────
        if or_rng < min_or:
            continue

        # ── Gate 1: GREEN zone ─────────────────────────────────────────────
        green = session.between_time("09:45", "09:54")
        if green.empty:
            continue

        # ── Session VWAP (cumulative; shift so bar-T uses bars < T) ────────
        tp_s   = (session["High"] + session["Low"] + session["Close"]) / 3
        vol_s  = session["Volume"]
        cumtpv = (tp_s * vol_s).cumsum().shift(1)
        cumvol = vol_s.cumsum().shift(1)
        sess_vwap = (cumtpv / cumvol.replace(0, np.nan)).to_dict()

        # ── Sweep state from pre-green bars (vectorized) ───────────────────
        pre_green = session.loc[session.index < green.index[0]]
        long_sweep  = bool(((pre_green["Low"]  < or_lo) & (pre_green["Close"] > or_lo)).any())
        short_sweep = bool(((pre_green["High"] > or_hi) & (pre_green["Close"] < or_hi)).any())

        daily_trades   = 0
        or1_long_done  = False
        or1_short_done = False

        for ts, bar in green.iterrows():
            # Gate 4
            if daily_trades >= max_tpd:
                break

            # ── Compute this bar's sweep contribution BEFORE any continue ──
            # (so sweep state stays consistent for next iteration)
            if bar["Low"]  < or_lo and bar["Close"] > or_lo: long_sweep  = True
            if bar["High"] > or_hi and bar["Close"] < or_hi: short_sweep = True

            # Gate 8: ADX
            if use_adx and adx_d.get(ts, 0.0) < min_adx:
                continue

            entry = bar["Open"]  # In "open prices only" mode, this is the signal price

            # "Current close" = last completed bar before this green zone bar
            pre_ts  = session.loc[session.index < ts]
            last_cl = float(pre_ts["Close"].iloc[-1]) if not pre_ts.empty else entry

            # Session VWAP (shift(1) means bar T uses bars before T)
            vwap = sess_vwap.get(ts, np.nan)
            if np.isnan(vwap) and not pre_ts.empty:
                tp_pre  = (pre_ts["High"] + pre_ts["Low"] + pre_ts["Close"]) / 3
                vol_pre = pre_ts["Volume"]
                vwap    = float((tp_pre * vol_pre).sum() / vol_pre.sum()) if vol_pre.sum() > 0 else last_cl

            # Gate 5: Market state (M1 EMA20 + VWAP)
            ema_now = ema20_d.get(ts, np.nan)
            ema_lag = ema20_lag_d.get(ts, np.nan)
            if np.isnan(ema_now):
                continue
            ema_rising  = (not np.isnan(ema_lag)) and (ema_now > ema_lag)
            bull_market = (last_cl > vwap) and ema_rising
            bear_market = (last_cl < vwap) and (not ema_rising)

            # ── OR-1 LONG ──────────────────────────────────────────────────
            if not or1_long_done and bull_market:
                if long_sweep and (last_cl > or_hi) and (abs(entry - or_hi) < or_tol):
                    # Gate 10: H1 EMA must be bullish
                    h1p = h1_cls_d.get(ts, np.nan)
                    h1e = h1_ema_d.get(ts, np.nan)
                    htf_ok = not use_h1 or np.isnan(h1p) or np.isnan(h1e) or h1p > h1e
                    if htf_ok:
                        sl_p = entry - sl_pts
                        tp_p = entry + tp_pts
                        pnl, xt, xr = _scan_exit(day_df, ts, "LONG", sl_p, tp_p, sl_pts, tp_pts, lot, dpp, spread_pts)
                        retest_min = ts.hour * 60 + ts.minute - (9 * 60 + 45)
                        trades.append({
                            "date": day, "entry_time": ts, "exit_time": xt,
                            "direction": "LONG", "entry": round(entry, 2),
                            "sl": round(sl_p, 2), "tp": round(tp_p, 2),
                            "or_high": round(or_hi, 2), "or_low": round(or_lo, 2),
                            "or_range": round(or_rng, 2),
                            "day_of_week": day_of_week, "gap_pts": gap_pts,
                            "retest_min": retest_min,
                            "adx_val": round(adx_d.get(ts, 0.0), 1),
                            "profit": round(pnl, 2), "exit_reason": xr,
                        })
                        daily_trades += 1
                        or1_long_done = True

            # ── OR-1 SHORT ─────────────────────────────────────────────────
            if not or1_short_done and bear_market:
                if short_sweep and (last_cl < or_lo) and (abs(entry - or_lo) < or_tol):
                    # Gate 10: H1 EMA must be bearish
                    h1p = h1_cls_d.get(ts, np.nan)
                    h1e = h1_ema_d.get(ts, np.nan)
                    htf_ok = not use_h1 or np.isnan(h1p) or np.isnan(h1e) or h1p < h1e
                    if htf_ok:
                        sl_p = entry + sl_pts
                        tp_p = entry - tp_pts
                        pnl, xt, xr = _scan_exit(day_df, ts, "SHORT", sl_p, tp_p, sl_pts, tp_pts, lot, dpp, spread_pts)
                        retest_min = ts.hour * 60 + ts.minute - (9 * 60 + 45)
                        trades.append({
                            "date": day, "entry_time": ts, "exit_time": xt,
                            "direction": "SHORT", "entry": round(entry, 2),
                            "sl": round(sl_p, 2), "tp": round(tp_p, 2),
                            "or_high": round(or_hi, 2), "or_low": round(or_lo, 2),
                            "or_range": round(or_rng, 2),
                            "day_of_week": day_of_week, "gap_pts": gap_pts,
                            "retest_min": retest_min,
                            "adx_val": round(adx_d.get(ts, 0.0), 1),
                            "profit": round(pnl, 2), "exit_reason": xr,
                        })
                        daily_trades += 1
                        or1_short_done = True

        prev_close = float(day_df["Close"].iloc[-1])

    cols = ["date","entry_time","exit_time","direction","entry","sl","tp",
            "or_high","or_low","or_range",
            "day_of_week","gap_pts","retest_min","adx_val",
            "profit","exit_reason"]
    return pd.DataFrame(trades, columns=cols) if trades else pd.DataFrame(columns=cols)




# ─────────────────────────────────────────────────────────────────────────────
# CURVE FITTING RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def _fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """
    One-sided Fisher's exact test: is group-1 win rate > group-2 win rate?
    Returns p-value (probability of observed or more extreme result by chance).
    Uses hypergeometric distribution — no scipy needed.
    """
    from math import comb
    n = a + b + c + d
    k = a + b   # group-1 total
    m = a + c   # total wins
    p = 0.0
    for x in range(a, min(k, m) + 1):
        try:
            p += comb(m, x) * comb(n - m, k - x) / comb(n, k)
        except ZeroDivisionError:
            pass
    return min(p, 1.0)


def curve_fit_assessment(
    trades: pd.DataFrame,
    grid: dict,
    trad_trades: pd.DataFrame,
    n_params_optimized: int,
) -> None:
    """
    Render a full curve-fitting risk panel for the current backtest run.

    Tests applied
    ─────────────
    1. Return Degradation          OOS vs in-sample P&L shrinkage
    2. Parameter Stability         Variance of best params across folds
    3. Trade Count Adequacy        Min sample needed for 95% CI on win rate
    4. Fisher's Exact Test         Is OOS win rate better than chance?
    5. Permutation Test            How often does OOS P&L arise by chance?
    6. In-Sample / OOS Split       50/50 date split win rate comparison
    7. Multiple Comparisons        Bonferroni penalty for grid dimensions
    8. Sample Size Projection      Months needed to reach statistical power
    """
    st.header("🔬 Curve Fitting Risk Assessment")
    st.caption(
        "Every backtest carries curve-fitting risk. These tests quantify how much "
        "of the result may be data mining vs genuine edge."
    )

    if trades is None or trades.empty or trad_trades is None or trad_trades.empty:
        st.warning("Insufficient trades to run curve-fitting assessment.")
        return

    oos_p   = trades["profit"].values
    trad_p  = trad_trades["profit"].values
    n_oos   = len(oos_p)
    n_trad  = len(trad_p)

    # ── Pre-compute shared stats ──────────────────────────────────────────────
    oos_wins  = int((oos_p  > 0).sum())
    oos_loss  = int((oos_p  < 0).sum())
    oos_wr    = oos_wins / n_oos if n_oos else 0

    trad_wins = int((trad_p > 0).sum())
    trad_wr   = trad_wins / n_trad if n_trad else 0

    oos_pnl   = float(oos_p.sum())
    trad_pnl  = float(trad_p.sum())
    deg       = ((trad_pnl - oos_pnl) / abs(trad_pnl) * 100) if trad_pnl != 0 else 0

    # ── TEST 1 · Return Degradation ───────────────────────────────────────────
    with st.expander("1 · Return Degradation", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("In-Sample P&L",  f"${trad_pnl:,.2f}")
        col2.metric("OOS P&L",        f"${oos_pnl:,.2f}")
        col3.metric("Degradation",    f"{deg:.1f}%",
                    delta=f"{'⚠️ High' if deg > 50 else '✅ OK'}",
                    delta_color="off")
        if deg > 75:
            st.error("⛔ >75% degradation — most in-sample return is illusory. "
                     "Likely curve-fitted.")
        elif deg > 50:
            st.warning("⚠️ 50–75% degradation — significant overfitting present.")
        elif deg > 25:
            st.info("ℹ️ 25–50% degradation — moderate. Normal for optimized systems.")
        else:
            st.success("✅ <25% degradation — OOS closely tracks in-sample.")

    # ── TEST 2 · Parameter Stability (grid variance) ──────────────────────────
    with st.expander("2 · Parameter Stability"):
        st.caption("Params that jump erratically across folds = overfitting to each window")
        optimized_keys = [k for k, v in grid.items() if len(v) > 1]
        if not optimized_keys:
            st.info("No parameters were optimized — nothing to check.")
        else:
            for k in optimized_keys:
                st.write(f"**{k.replace('_',' ').title()}** — grid: {grid[k]}")
            st.caption("(Check Parameter Stability charts above for fold-by-fold values)")

    # ── TEST 3 · Trade Count Adequacy ─────────────────────────────────────────
    with st.expander("3 · Trade Count Adequacy"):
        # Required n for 95% CI, ±10% margin on observed win rate
        p_est = max(oos_wr, 0.5)   # conservative estimate
        z = 1.96; e = 0.10
        n_needed = int(np.ceil(z**2 * p_est * (1 - p_est) / e**2))
        gap = max(n_needed - n_oos, 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("OOS Trades",      n_oos)
        col2.metric("Trades Needed",   n_needed,
                    help="For 95% CI ±10% on win rate")
        col3.metric("Shortfall",       gap,
                    delta="✅ Sufficient" if gap == 0 else f"⚠️ Need {gap} more",
                    delta_color="off")

        if n_oos >= n_needed:
            st.success(f"✅ {n_oos} trades exceeds the {n_needed}-trade minimum.")
        else:
            months_data = (
                (trades["entry_time"].max() - trades["entry_time"].min()).days / 30
                if "entry_time" in trades.columns and n_oos > 1 else 0
            )
            rate = n_oos / months_data if months_data > 0 else 0
            months_needed = gap / rate if rate > 0 else float("inf")
            st.warning(
                f"⚠️ Only {n_oos} OOS trades. Need {n_needed} for statistical confidence. "
                f"At current rate ({rate:.1f}/month) → ~{months_needed:.0f} more months of data."
            )

    # ── TEST 4 · Fisher's Exact Test ──────────────────────────────────────────
    with st.expander("4 · Fisher's Exact Test (OOS win rate vs random chance)"):
        # Compare OOS win rate vs a fair coin (50% = zero edge baseline)
        # H1: OOS win rate > 50%
        a = oos_wins; b = oos_loss
        c = n_oos // 2; d = n_oos - c   # expected under H0 (50% WR)
        p_fish = _fisher_exact_greater(a, b, c, d)

        bonf_k  = max(n_params_optimized, 1)
        bonf_th = 0.05 / bonf_k
        passes  = p_fish < 0.05
        passes_bonf = p_fish < bonf_th

        col1, col2, col3 = st.columns(3)
        col1.metric("OOS Win Rate",     f"{oos_wr:.1%}")
        col2.metric("p-value",          f"{p_fish:.4f}")
        col3.metric("Bonferroni p<",    f"{bonf_th:.4f}",
                    help=f"Adjusted for {bonf_k} optimized parameters")

        if passes_bonf:
            st.success(f"✅ Win rate is significant even after Bonferroni correction "
                       f"(p={p_fish:.4f} < {bonf_th:.4f}).")
        elif passes:
            st.warning(f"⚠️ Win rate is significant at p<0.05 (p={p_fish:.4f}) but FAILS "
                       f"Bonferroni correction (need p<{bonf_th:.4f}). "
                       f"Result may be a multiple-comparisons artifact.")
        else:
            st.error(f"⛔ Win rate is NOT significantly better than chance "
                     f"(p={p_fish:.4f}). Insufficient evidence of edge.")

    # ── TEST 5 · Permutation Test ─────────────────────────────────────────────
    with st.expander("5 · Permutation Test (P&L vs shuffled baseline)"):
        np.random.seed(42)
        n_perm  = 5000
        obs_pnl = float(oos_p.sum())
        # Shuffle trade outcomes; keep same entry/exit structure
        perm_pnls = np.array([
            np.random.choice(oos_p, size=n_oos, replace=False).sum()
            for _ in range(n_perm)
        ])
        p_perm = float((perm_pnls >= obs_pnl).mean())
        pct_rank = float((perm_pnls < obs_pnl).mean()) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Observed OOS P&L",  f"${obs_pnl:,.2f}")
        col2.metric("p-value (perm)",     f"{p_perm:.4f}")
        col3.metric("Percentile rank",    f"{pct_rank:.1f}%",
                    help="How the OOS P&L ranks vs 5,000 random shuffles")

        # Mini histogram of permutation distribution
        fig_perm = go.Figure()
        fig_perm.add_trace(go.Histogram(
            x=perm_pnls, nbinsx=50, name="Shuffled P&L",
            marker_color="#4488ff", opacity=0.7
        ))
        fig_perm.add_vline(x=obs_pnl, line_color="#ff8c00", line_width=2,
                           annotation_text=f"Observed ${obs_pnl:,.0f}",
                           annotation_position="top right")
        fig_perm.update_layout(
            template="plotly_dark", height=240,
            xaxis_title="P&L ($)", yaxis_title="Count",
            margin=dict(t=10, b=30), showlegend=False
        )
        st.plotly_chart(fig_perm, use_container_width=True)

        if p_perm < 0.05:
            st.success(f"✅ OOS P&L is in the top {100-pct_rank:.1f}% of random shuffles "
                       f"(p={p_perm:.4f}). Unlikely to be luck.")
        else:
            st.error(f"⛔ OOS P&L could arise by chance {p_perm*100:.1f}% of the time. "
                     f"No statistically significant edge detected.")

    # ── TEST 6 · In-Sample / OOS Date Split ───────────────────────────────────
    with st.expander("6 · In-Sample / OOS Date Split (50/50)"):
        st.caption("Does the win rate hold on the second half of data the optimizer never saw?")
        if "entry_time" not in trades.columns or "entry_time" not in trad_trades.columns:
            st.info("entry_time column not found — skipping date split.")
        else:
            all_tr = pd.concat([trad_trades, trades]).drop_duplicates()
            all_tr = all_tr.sort_values("entry_time")
            split_ts = all_tr["entry_time"].quantile(0.5)

            first_h = all_tr[all_tr["entry_time"] <= split_ts]
            second_h = all_tr[all_tr["entry_time"] >  split_ts]

            def half_stats(h):
                p = h["profit"].values if not h.empty else np.array([])
                if len(p) == 0:
                    return 0, 0, 0.0, 0.0
                w = int((p > 0).sum()); n = len(p)
                return w, n, w / n, p.sum()

            w1, n1, wr1, pnl1 = half_stats(first_h)
            w2, n2, wr2, pnl2 = half_stats(second_h)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("First Half (In-Sample)")
                st.metric("Trades", n1)
                st.metric("Win Rate", f"{wr1:.1%}")
                st.metric("P&L", f"${pnl1:,.2f}")
            with col2:
                st.subheader("Second Half (Out-of-Sample)")
                st.metric("Trades", n2)
                st.metric("Win Rate", f"{wr2:.1%}")
                st.metric("P&L", f"${pnl2:,.2f}")

            wr_drop = (wr1 - wr2) * 100
            if wr_drop > 20:
                st.error(f"⛔ Win rate dropped {wr_drop:.1f}pp from first to second half. "
                         f"Strong evidence of overfitting.")
            elif wr_drop > 10:
                st.warning(f"⚠️ Win rate dropped {wr_drop:.1f}pp. Moderate degradation.")
            else:
                st.success(f"✅ Win rate held within {abs(wr_drop):.1f}pp across halves. "
                           f"No obvious overfitting.")

    # ── TEST 7 · Multiple Comparisons ─────────────────────────────────────────
    with st.expander("7 · Multiple Comparisons Penalty"):
        total_combos = 1
        opt_dims = []
        for k, v in grid.items():
            if len(v) > 1:
                total_combos *= len(v)
                opt_dims.append(f"{k} ({len(v)} values)")

        bonf_alpha = 0.05 / max(total_combos, 1)
        st.metric("Parameter Combinations Tested", f"{total_combos:,}")
        st.metric("Bonferroni-Corrected α",        f"{bonf_alpha:.5f}",
                  help="Any p-value must be below this to survive multiple comparisons")
        if opt_dims:
            st.write("**Optimized dimensions:**")
            for d in opt_dims:
                st.write(f"  • {d}")
        st.caption(
            "With many combinations, random chance produces a winner. "
            "Bonferroni divides the 5% significance budget across all tested combos. "
            f"With {total_combos} combos, a 'significant' result needs p < {bonf_alpha:.5f}."
        )
        if total_combos > 100:
            st.warning(f"⚠️ {total_combos} combos tested. High multiple-comparisons risk. "
                       f"Walk-forward OOS performance is the real test — ignore in-sample rank.")
        else:
            st.info(f"ℹ️ {total_combos} combos — manageable. "
                    f"Still apply Bonferroni if reporting p-values.")

    # ── TEST 8 · Overall Verdict ───────────────────────────────────────────────
    with st.expander("8 · Overall Verdict", expanded=True):
        flags = []
        if deg > 50:          flags.append(("⚠️ High return degradation",  f"{deg:.0f}%"))
        if n_oos < n_needed:  flags.append(("⚠️ Insufficient trade count", f"{n_oos} / {n_needed} needed"))
        if p_fish >= 0.05:    flags.append(("⚠️ Win rate not significant",  f"p={p_fish:.4f}"))
        if p_perm >= 0.05:    flags.append(("⚠️ P&L not significant",       f"p={p_perm:.4f}"))
        if total_combos > 50: flags.append(("⚠️ Multiple comparisons risk", f"{total_combos} combos"))

        green_flags = []
        if deg <= 25:                green_flags.append("✅ Low return degradation")
        if n_oos >= n_needed:        green_flags.append("✅ Sufficient trade count")
        if p_fish < 0.05:            green_flags.append("✅ Win rate significant")
        if p_perm < 0.05:            green_flags.append("✅ P&L not due to chance")
        if passes_bonf:              green_flags.append("✅ Survives Bonferroni correction")

        score = len(green_flags) / max(len(flags) + len(green_flags), 1) * 100

        if score >= 80:
            verdict_color = "success"
            verdict = f"🟢 ROBUST — {score:.0f}% of checks passed. Evidence of genuine edge."
        elif score >= 50:
            verdict_color = "warning"
            verdict = f"🟡 UNCERTAIN — {score:.0f}% passed. More data needed before trading live."
        else:
            verdict_color = "error"
            verdict = f"🔴 HIGH RISK — {score:.0f}% passed. Do not trade live without more evidence."

        getattr(st, verdict_color)(verdict)

        if green_flags:
            for g in green_flags:
                st.write(g)
        if flags:
            for f_label, f_val in flags:
                st.write(f"{f_label}: {f_val}")

        st.divider()
        st.caption(
            "**What to do next:**  "
            "(1) Collect more data — target 60+ OOS trades before drawing conclusions.  "
            "(2) Lock all parameters now — do not re-optimize after seeing results.  "
            "(3) Forward-test in a demo account for 1–3 months before going live.  "
            "(4) The walk-forward OOS P&L is the only number that matters."
        )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return dict(sharpe=0, pf=0, pnl=0, n=0, wr=0, dd=0)
    p   = trades["profit"].values
    pos = p[p > 0];  neg = p[p < 0]
    pnl = p.sum()
    n   = len(p)
    wr  = len(pos) / n if n else 0
    pf  = abs(pos.sum() / neg.sum()) if neg.sum() != 0 else (np.inf if pos.sum() > 0 else 0)
    sh  = (p.mean() / p.std() * np.sqrt(252)) if (n >= 2 and p.std() > 0) else 0
    eq  = np.cumsum(p)
    dd  = (np.maximum.accumulate(eq) - eq).max() if len(eq) else 0
    return dict(sharpe=sh, pf=pf, pnl=pnl, n=n, wr=wr, dd=dd)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

def optimize(df: pd.DataFrame, grid: dict, obj: str) -> dict:
    """Grid-search parameters on training slice. Returns best param dict."""
    best_score = -np.inf
    best_p     = None

    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        p  = dict(zip(keys, combo))
        tr = run_alchemy(df, p)
        m  = metrics(tr)
        sc = m["pf"] if obj == "profit_factor" else (m["sharpe"] if obj == "sharpe" else m["pnl"])
        if sc == np.inf:
            sc = m["pnl"]   # prefer more trades if PF is infinite
        if sc > best_score:
            best_score = sc
            best_p     = {**p, "_score": sc, "_n_trades": m["n"]}

    return best_p or dict(zip(keys, [v[0] for v in grid.values()]))


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, train_mo: int, blind_mo: int,
                 grid: dict, obj: str, prog) -> list:
    """Run walk-forward fold by fold. Returns list of fold dicts."""
    ny = pytz.timezone("America/New_York")
    idx = df.index.tz_convert(ny) if df.index.tz else df.index.tz_localize("UTC").tz_convert(ny)

    start = idx.min().date()
    end   = idx.max().date()
    cur   = start
    folds = []
    fn    = 1

    while True:
        te = (pd.Timestamp(cur) + pd.DateOffset(months=train_mo)).date()
        bs = te
        be = (pd.Timestamp(bs)  + pd.DateOffset(months=blind_mo)).date()
        if be > end:
            break

        tr_mask = (idx.date >= cur) & (idx.date < te)
        bl_mask = (idx.date >= bs)  & (idx.date < be)
        tr_df   = df[tr_mask]
        bl_df   = df[bl_mask]

        if len(tr_df) < 200 or len(bl_df) < 50:
            cur = bs;  continue

        prog.info(f"⚙️ Fold {fn} — training {cur} → {te} ...")
        best_p = optimize(tr_df, grid, obj)

        prog.info(f"🔍 Fold {fn} — blind test {bs} → {be} ...")
        bl_tr  = run_alchemy(bl_df, best_p)
        if not bl_tr.empty:
            bl_tr["is_oos"]    = True
            bl_tr["fold"]      = fn
            bl_tr["fold_params"] = str(best_p)
        bl_m   = metrics(bl_tr)

        folds.append({
            "fold":         fn,
            "train_start":  cur,
            "train_end":    te,
            "blind_start":  bs,
            "blind_end":    be,
            "best_params":  best_p,
            "blind_trades": bl_tr,
            **{f"oos_{k}": v for k, v in bl_m.items()},
        })
        cur = bs;  fn += 1

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(uploaded) -> pd.DataFrame:
    """
    Parse M1 CSV - supports MT4 and Dukascopy formats.
    MT4: no header, broker UTC+3. Dukascopy: has header with UTC timestamps.
    """
    raw   = uploaded.read().decode("utf-8", errors="ignore")
    lines = [l for l in raw.splitlines() if l.strip()]
    sep   = "," if "," in lines[0] else "\t"
    has_header = not lines[0][0].isdigit()

    if has_header:
        df = pd.read_csv(io.StringIO(raw), sep=sep, header=0)
        df.columns = [c.strip().strip("<>").upper() for c in df.columns]
        remap = {}
        for c in df.columns:
            cu = c.upper()
            if   "DATE" in cu and "TIME" not in cu: remap[c] = "Date"
            elif "TIME" in cu and "DATE" not in cu: remap[c] = "Time"
            elif cu in ("DATETIME", "TIMESTAMP"):   remap[c] = "Datetime"
            elif "OPEN"  in cu: remap[c] = "Open"
            elif "HIGH"  in cu: remap[c] = "High"
            elif "LOW"   in cu: remap[c] = "Low"
            elif "CLOSE" in cu: remap[c] = "Close"
            elif "VOL"   in cu or "TICK" in cu: remap[c] = "Volume"
        df.rename(columns=remap, inplace=True)
    else:
        col_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
        df = pd.read_csv(io.StringIO(raw), sep=sep, header=None, names=col_names)

    if "Datetime" in df.columns:
        df.index = pd.to_datetime(df["Datetime"])
    elif "Date" in df.columns and "Time" in df.columns:
        df.index = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    else:
        df.index = pd.to_datetime(df.iloc[:, 0])

    want = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[want].apply(pd.to_numeric, errors="coerce").dropna().sort_index()

    if df.index.tz is None:
        broker_tz = pytz.FixedOffset(180)
        df.index  = df.index.tz_localize(broker_tz)

    df.index.name = "Datetime"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("⚗️ AlchemySignal V1.5 · Walk-Forward Backtester")
    st.caption("NDX100 OR-1 · Break & Retest · PB1 Sweep & Go · True out-of-sample performance")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # Data
        st.subheader("📊 Data")
        src = st.radio("Source", [
            "Upload MT4 CSV",
            "Upload Dukascopy CSV",
            "yfinance NQ=F (60-day limit)",
        ])
        uploaded = None
        if src == "Upload MT4 CSV":
            st.caption("MT4 → Tools → History Center → NDX100,M1 → Export")
            uploaded = st.file_uploader("NDX100 M1 CSV", type=["csv","txt"])
        elif src == "Upload Dukascopy CSV":
            st.caption(
                "From widgets.dukascopy.com → USATECH.IDX/USD → 1 min → "
                "select date range → Download CSV"
            )
            uploaded = st.file_uploader("Dukascopy USATECH M1 CSV", type=["csv","txt"])
        else:
            st.info("NQ=F at 5m — limited history but works for quick tests")

        st.divider()

        # Walk-forward windows
        st.subheader("🗓️ Walk-Forward Windows")
        train_mo = st.number_input("Training window (months)", 1, 36,  3)
        blind_mo = st.number_input("Blind test window (months)", 1, 12, 1)

        st.divider()

        # Parameter optimization grid
        st.subheader("🎯 Parameters to Optimize")
        opt_tol   = st.checkbox("OR Retest Tolerance",  value=True)
        opt_range = st.checkbox("Min OR Range",          value=True)
        opt_tp    = st.checkbox("TP Distance",           value=True)
        opt_adx   = st.checkbox("ADX Filter Threshold",  value=False)
        opt_h1    = st.checkbox("H1 EMA Alignment Filter", value=False)

        st.divider()

        # Fixed parameters
        st.subheader("🔒 Fixed Parameters")
        sl_pts     = st.number_input("SL Points",          value=24.2,  step=0.1)
        lot        = st.number_input("Lot Size",           value=0.30,  step=0.01)
        max_tpd    = st.number_input("Max Trades/Day",     min_value=1, max_value=5, value=2)
        spread_pts = st.number_input("Spread (pts)",       value=5.0,   step=0.5,
                                     help="Broker bid/ask spread in price points. "
                                          "Deducted from every trade: SL cost = SL+spread, TP gain = TP−spread.")

        st.divider()

        st.subheader("📐 Optimize For")
        obj = st.selectbox("Objective", ["profit_factor", "sharpe", "pnl"])

        run_btn = st.button("🚀 Run Walk-Forward", type="primary", use_container_width=True)

    # ── LANDING PAGE ──────────────────────────────────────────────────────────
    if not run_btn:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("❌ Traditional Backtesting")
            st.markdown("""
- Optimizes on **all historical data at once**
- Model memorizes the past (curve fitting)
- Returns look great — but won't hold live
- Like giving the student the answer sheet
            """)
        with c2:
            st.subheader("✅ Walk-Forward (this engine)")
            st.markdown("""
- Trains on a **short rolling window only**
- Locks best params → tests on **unseen data**
- Stitches only out-of-sample equity
- Reflects what you'd have made trading live
            """)
        st.divider()
        st.subheader("📥 How to Export MT4 Data")
        st.markdown("""
1. In MT4 open **Tools → History Center**
2. Select **NDX100** → **M1** → double-click to load
3. Click **Export** and save the `.csv`
4. Upload it in the sidebar →
        """)
        st.subheader("⚗️ AlchemySignal V1.5 Gates")
        st.markdown("""
| Gate | Logic |
|------|-------|
| 1 | GREEN zone only — 09:45–09:54 ET |
| 2 | OR locked from 09:30–09:34 ET (5-min range) |
| 4 | Max 2 trades per day |
| 8 | ADX ≥ threshold (V1.5 chop filter) |
| 9 | OR range ≥ minimum (V1.5 range filter) |
| 10 | H1 EMA trend alignment (V1.5 trend filter) |
        """)
        return

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        try:
            if src in ("Upload MT4 CSV", "Upload Dukascopy CSV"):
                if uploaded is None:
                    st.error("Please upload a CSV file.")
                    return
                data = load_csv(uploaded)
            else:
                import yfinance as yf
                raw  = yf.Ticker("NQ=F").history(period="60d", interval="5m")
                data = raw[["Open","High","Low","Close","Volume"]].copy()
                data.index = data.index.tz_convert("UTC")
        except Exception as e:
            st.error(f"Data error: {e}")
            return

    if data is None or len(data) < 500:
        st.error("Not enough bars. Need at least a few months of M1/5m data.")
        return

    st.success(
        f"✅ {len(data):,} bars loaded · "
        f"{data.index.min().date()} → {data.index.max().date()}"
    )

    # ── BUILD PARAMETER GRID ──────────────────────────────────────────────────
    grid: dict = {
        "sl_points":           [sl_pts],
        "lot_size":            [lot],
        "max_trades_per_day":  [max_tpd],
        "spread_pts":          [spread_pts],
        "use_adx_filter":      [False],
        "min_adx":             [20.0],
        "use_h1_ema_filter":   [False],
        "h1_ema_period":       [20],
        "or_retest_tolerance": [20.0],
        "min_or_range":        [80.0],
        "tp_points":           [150.0],
    }

    if opt_tol:
        # NDX100 on M1 bars: bar opens can be 5–25 pts from OR level at retest.
        # Wider values capture intrabar retests that Control Points mode would fire on.
        grid["or_retest_tolerance"] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    if opt_range:
        grid["min_or_range"]        = [0.0, 40.0, 60.0, 80.0, 100.0, 120.0]
    if opt_tp:
        grid["tp_points"]           = [80.0, 100.0, 120.0, 150.0, 200.0]
    if opt_adx:
        grid["use_adx_filter"]      = [True]
        grid["min_adx"]             = [15.0, 20.0, 25.0, 30.0]
    if opt_h1:
        grid["use_h1_ema_filter"]   = [True]
        grid["h1_ema_period"]       = [10, 20, 50]

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    st.info(f"🔢 {n_combos} parameter combinations · objective: **{obj}**")

    # ── TRADITIONAL BACKTEST (in-sample illusion) ─────────────────────────────
    with st.spinner("Running traditional in-sample backtest (for comparison)…"):
        trad_best   = optimize(data, grid, obj)
        trad_trades = run_alchemy(data, trad_best)
        trad_m      = metrics(trad_trades)

    # ── WALK-FORWARD ──────────────────────────────────────────────────────────
    prog = st.empty()
    with st.spinner("Running walk-forward analysis…"):
        folds = walk_forward(data, train_mo, blind_mo, grid, obj, prog)
    prog.empty()

    if not folds:
        st.error("Not enough data for even one fold. Reduce window sizes or upload more data.")
        return

    # Stitch OOS trades
    oos_parts = [f["blind_trades"] for f in folds if not f["blind_trades"].empty]
    oos_all   = pd.concat(oos_parts).sort_values("entry_time") if oos_parts else pd.DataFrame()
    oos_m     = metrics(oos_all)

    # ── HERMES EXPORT ──────────────────────────────────────────────────────────
    if not oos_all.empty:
        hermes_csv = oos_all.to_csv(index=False)
        st.sidebar.divider()
        st.sidebar.subheader("U0001F916 Hermes Agent")
        st.sidebar.download_button(
            label="⬇️ Export OOS Trade Log (for Hermes)",
            data=hermes_csv,
            file_name="hermes_trade_log.csv",
            mime="text/csv",
            help="Feed this CSV to hermes.py to generate strategy improvement hypotheses.",
        )

    # Return degradation
    deg = ((trad_m["pnl"] - oos_m["pnl"]) / abs(trad_m["pnl"]) * 100
           if trad_m["pnl"] != 0 else 0)

    # ── SECTION 1: TOP METRICS ────────────────────────────────────────────────
    st.header("📊 Results Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Walk-Forward P&L",    f"${oos_m['pnl']:,.2f}")
    c2.metric("Traditional P&L",     f"${trad_m['pnl']:,.2f}",
              help="In-sample: optimized on ALL data (curve fitting)")
    c3.metric("Return Degradation",  f"{deg:.1f}%",
              help="% of traditional return that was fake")
    c4.metric("OOS Trades",          oos_m["n"])
    c5.metric("OOS Win Rate",        f"{oos_m['wr']:.1%}")

    # ── SECTION 2: EQUITY CURVES ──────────────────────────────────────────────
    st.header("📈 Equity Curves")
    st.caption("Blue = in-sample illusion (optimized on all data) · Orange = true walk-forward performance")

    trad_eq = np.concatenate([[0], np.cumsum(trad_trades["profit"].values)]) if not trad_trades.empty else [0]
    oos_eq  = np.concatenate([[0], np.cumsum(oos_all["profit"].values)])    if not oos_all.empty   else [0]

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(y=trad_eq, name="Traditional (In-Sample Illusion)",
                                line=dict(color="#4488ff", width=2)))
    fig_eq.add_trace(go.Scatter(y=oos_eq,  name="Walk-Forward (True OOS)",
                                line=dict(color="#ff8c00", width=2)))
    fig_eq.update_layout(
        xaxis_title="Trade #", yaxis_title="Cumulative P&L ($)",
        template="plotly_dark", height=380, margin=dict(t=20)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── SECTION 3: STATS TABLE ────────────────────────────────────────────────
    st.header("📋 Performance Statistics")
    pf_fmt = lambda v: f"{v:.2f}" if v != np.inf else "∞"
    stats = pd.DataFrame({
        "Metric": ["Total P&L","Sharpe Ratio","Profit Factor","Win Rate","Max Drawdown","Trade Count"],
        "Traditional (In-Sample)": [
            f"${trad_m['pnl']:,.2f}", f"{trad_m['sharpe']:.2f}",
            pf_fmt(trad_m['pf']),     f"{trad_m['wr']:.1%}",
            f"${trad_m['dd']:,.2f}",  trad_m['n'],
        ],
        "Walk-Forward (True OOS)": [
            f"${oos_m['pnl']:,.2f}", f"{oos_m['sharpe']:.2f}",
            pf_fmt(oos_m['pf']),     f"{oos_m['wr']:.1%}",
            f"${oos_m['dd']:,.2f}",  oos_m['n'],
        ],
    }).set_index("Metric")
    st.dataframe(stats, use_container_width=True)

    # ── SECTION 4: FOLD PROGRESS TABLE ───────────────────────────────────────
    st.header("📅 Fold-by-Fold Results")
    fold_rows = []
    for f in folds:
        bp = f["best_params"]
        fold_rows.append({
            "Fold":         f["fold"],
            "Train":        f"{f['train_start']} → {f['train_end']}",
            "Blind Test":   f"{f['blind_start']} → {f['blind_end']}",
            "OR Tol":       bp.get("or_retest_tolerance"),
            "Min OR Rng":   bp.get("min_or_range"),
            "TP Pts":       bp.get("tp_points"),
            "ADX Min":      bp.get("min_adx") if bp.get("use_adx_filter") else "OFF",
            "OOS Trades":   f["oos_n"],
            "OOS P&L":      f"${f['oos_pnl']:.2f}",
            "OOS PF":       pf_fmt(f["oos_pf"]),
        })
    st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

    # ── SECTION 5: OOS P&L PER FOLD ──────────────────────────────────────────
    st.subheader("OOS P&L by Fold")
    pnls  = [f["oos_pnl"] for f in folds]
    fnums = [f["fold"]    for f in folds]
    fig_bar = go.Figure(go.Bar(
        x=fnums, y=pnls,
        marker_color=["#00cc66" if p >= 0 else "#ff4444" for p in pnls],
        text=[f"${p:,.0f}" for p in pnls], textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis_title="Fold", yaxis_title="P&L ($)",
        template="plotly_dark", height=300, margin=dict(t=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── SECTION 6: PARAMETER STABILITY ───────────────────────────────────────
    st.header("🔧 Parameter Stability")
    st.caption("Stable across folds = robust strategy · Erratic = curve fitting")

    tracked = [k for k in ["or_retest_tolerance","min_or_range","tp_points","min_adx"]
               if any(len(grid.get(k, [])) > 1 for _ in [0])]
    if not tracked:
        tracked = ["or_retest_tolerance","min_or_range","tp_points"]

    cols_ps = st.columns(len(tracked))
    for i, pk in enumerate(tracked):
        vals = [f["best_params"].get(pk) for f in folds]
        fnms = [f["fold"] for f in folds]
        fig_p = go.Figure(go.Scatter(
            x=fnms, y=vals, mode="lines+markers",
            line=dict(width=2), marker=dict(size=8)
        ))
        fig_p.update_layout(
            title=pk.replace("_"," ").title(),
            template="plotly_dark", height=220,
            xaxis_title="Fold", margin=dict(t=30,b=30,l=30,r=10)
        )
        cols_ps[i].plotly_chart(fig_p, use_container_width=True)

    # ── SECTION 7: TIMELINE DIAGRAM ───────────────────────────────────────────
    st.header("📆 Walk-Forward Timeline")
    fig_tl = go.Figure()
    for f in folds:
        fn_ = f["fold"]
        fig_tl.add_trace(go.Bar(
            x=[fn_], y=[train_mo], marker_color="#4488ff",
            name="Train" if fn_ == 1 else None, showlegend=(fn_ == 1),
            hovertemplate=f"Fold {fn_}<br>Train: {f['train_start']} → {f['train_end']}<extra></extra>",
        ))
        fig_tl.add_trace(go.Bar(
            x=[fn_], y=[blind_mo], marker_color="#ff8c00",
            name="Blind Test" if fn_ == 1 else None, showlegend=(fn_ == 1),
            hovertemplate=f"Fold {fn_}<br>Test: {f['blind_start']} → {f['blind_end']}<extra></extra>",
        ))
    fig_tl.update_layout(
        barmode="stack", xaxis_title="Fold #", yaxis_title="Months",
        template="plotly_dark", height=280, margin=dict(t=20)
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # ── SECTION 8: TRADE LOG ──────────────────────────────────────────────────
    if not oos_all.empty:
        with st.expander(f"📝 Full OOS Trade Log ({len(oos_all)} trades)"):
            show = ["date","entry_time","direction","entry","or_range",
                    "profit","exit_reason"]
            st.dataframe(oos_all[show].reset_index(drop=True), use_container_width=True)

    # ── OR RANGE DISTRIBUTION ─────────────────────────────────────────────────
    if not oos_all.empty and "or_range" in oos_all.columns:
        with st.expander("📊 OR Range Distribution (OOS Trades)"):
            fig_hist = px.histogram(
                oos_all, x="or_range", color="exit_reason",
                nbins=20, template="plotly_dark",
                title="OR Range at Trade Entry (TP vs SL)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.success(
        f"✅ Walk-forward complete · {len(folds)} folds · "
        f"{oos_m['n']} OOS trades · Net P&L: ${oos_m['pnl']:,.2f} · "
        f"Profit Factor: {pf_fmt(oos_m['pf'])}"
    )

    # ── SECTION 9: CURVE FITTING RISK ASSESSMENT ──────────────────────────────
    n_params_optimized = sum(1 for v in grid.values() if len(v) > 1)
    curve_fit_assessment(
        trades              = oos_all,
        grid                = grid,
        trad_trades         = trad_trades,
        n_params_optimized  = n_params_optimized,
    )


if __name__ == "__main__":
    main()
