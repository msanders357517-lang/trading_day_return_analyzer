
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pathlib import Path

st.set_page_config(
    page_title="ETF Performance & Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# CONSTANTS
# ============================================================

PERIODS = [
    "1D", "1W", "2W", "1M", "3M", "6M", "1Y",
    "3Y CAGR", "5Y CAGR", "10Y CAGR"
]

WINDOWS = {
    "1D": 1,
    "1W": 5,
    "2W": 10,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}

STRATEGIES = {
    "Balanced": {
        "1D": 0.02, "1W": 0.04, "2W": 0.05, "1M": 0.08,
        "3M": 0.11, "6M": 0.15, "1Y": 0.16,
        "3Y CAGR": 0.15, "5Y CAGR": 0.13, "10Y CAGR": 0.11,
    },
    "Short-Term Momentum": {
        "1D": 0.05, "1W": 0.10, "2W": 0.12, "1M": 0.18,
        "3M": 0.22, "6M": 0.20, "1Y": 0.13,
    },
    "Long-Term Growth": {
        "1Y": 0.15, "3Y CAGR": 0.25, "5Y CAGR": 0.30, "10Y CAGR": 0.30,
    },
    "Consistency": {
        "1M": 0.08, "3M": 0.12, "6M": 0.15, "1Y": 0.18,
        "3Y CAGR": 0.18, "5Y CAGR": 0.16, "10Y CAGR": 0.13,
    },
    "Aggressive Growth": {
        "1W": 0.05, "2W": 0.08, "1M": 0.12, "3M": 0.20,
        "6M": 0.20, "1Y": 0.18, "3Y CAGR": 0.10, "5Y CAGR": 0.07,
    },
}

# ============================================================
# HELPERS
# ============================================================

def pct(v):
    return "—" if pd.isna(v) else f"{v:.2%}"

def money(v):
    return "—" if pd.isna(v) else f"${v:,.2f}"

def fmt_num(v, suffix=""):
    if pd.isna(v):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.2f}M{suffix}"
    if abs(v) >= 1_000:
        return f"{v/1_000:,.2f}K{suffix}"
    return f"{v:,.2f}{suffix}"

@st.cache_data(show_spinner=False)
def load_universe():
    root_file = Path("ETF_1000.xlsx")
    data_file = Path("data") / "ETF_1000.xlsx"

    if root_file.exists():
        workbook_path = root_file
    elif data_file.exists():
        workbook_path = data_file
    else:
        raise FileNotFoundError(
            "ETF_1000.xlsx was not found. Place it either at the repository root "
            "or inside a data folder."
        )

    df = pd.read_excel(workbook_path)
    df.columns = [str(c).strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"].notna() & (df["Symbol"] != "") & (df["Symbol"] != "NAN")]
    df = df.drop_duplicates("Symbol")

    for col in ["Assets", "Stock Price", "% Change", "CAGR 1Y", "CAGR 3Y", "CAGR 5Y", "CAGR 10Y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Always include WLDU even if it has not yet been added to the workbook.
    if "WLDU" not in set(df["Symbol"]):
        extra = {col: np.nan for col in df.columns}
        extra["Symbol"] = "WLDU"
        if "Fund Name" in df.columns:
            extra["Fund Name"] = "Leverage Shares 2x Long World Stock Daily ETF"
        if "Leverage" in df.columns:
            extra["Leverage"] = "2x Long"
        df = pd.concat([df, pd.DataFrame([extra])], ignore_index=True)

    if "Assets" in df.columns:
        df = df.sort_values("Assets", ascending=False, na_position="last")

    return df.reset_index(drop=True)

def _close_frame(raw, tickers):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        out = {}
        level0 = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in level0:
                sub = raw[t]
                if "Close" in sub.columns:
                    out[t] = sub["Close"]
        return pd.DataFrame(out)
    if len(tickers) == 1 and "Close" in raw.columns:
        return pd.DataFrame({tickers[0]: raw["Close"]})
    return pd.DataFrame()

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_prices(tickers_tuple, period="10y"):
    tickers = list(tickers_tuple)
    frames = []
    failed = []

    for i in range(0, len(tickers), 75):
        batch = tickers[i:i+75]
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            frame = _close_frame(raw, batch)
            if not frame.empty:
                frames.append(frame)
                failed.extend([t for t in batch if t not in frame.columns])
            else:
                failed.extend(batch)
        except Exception:
            failed.extend(batch)

    if not frames:
        return pd.DataFrame(), sorted(set(failed))

    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.sort_index(), sorted(set(failed))

def trailing_return(s, sessions):
    s = s.dropna()
    if len(s) <= sessions:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-(sessions + 1)] - 1)

def cagr(s, years):
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    target = s.index[-1] - pd.DateOffset(years=years)
    x = s[s.index >= target]
    if len(x) < 2:
        return np.nan
    actual_years = (x.index[-1] - x.index[0]).days / 365.25
    if actual_years < years * 0.85:
        return np.nan
    if x.iloc[0] <= 0 or x.iloc[-1] <= 0:
        return np.nan
    return float((x.iloc[-1] / x.iloc[0]) ** (1 / actual_years) - 1)

def volatility(s):
    r = s.dropna().pct_change().dropna()
    return np.nan if len(r) < 30 else float(r.std() * np.sqrt(252))

def max_drawdown(s):
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    return float((s / s.cummax() - 1).min())

def trading_day_stats(series):
    """Close-to-close win/loss analytics using adjusted prices."""
    s = series.dropna().sort_index()
    if len(s) < 2:
        return None

    returns = s.pct_change().dropna()
    positive_r = returns[returns > 0]
    negative_r = returns[returns < 0]
    flat = int((returns == 0).sum())
    positive = int(len(positive_r))
    negative = int(len(negative_r))
    total = int(len(returns))
    non_flat = positive + negative

    avg_win = float(positive_r.mean()) if positive else np.nan
    avg_loss = float(negative_r.mean()) if negative else np.nan
    payoff_ratio = (
        avg_win / abs(avg_loss)
        if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0
        else np.nan
    )

    start_price = float(s.iloc[0])
    end_price = float(s.iloc[-1])
    total_return = end_price / start_price - 1 if start_price > 0 else np.nan
    years = (s.index[-1] - s.index[0]).days / 365.25
    history_cagr = (
        (end_price / start_price) ** (1 / years) - 1
        if start_price > 0 and end_price > 0 and years > 0
        else np.nan
    )

    return {
        "Start Date": s.index[0].date(),
        "End Date": s.index[-1].date(),
        "Positive Days": positive,
        "Negative Days": negative,
        "Flat Days": flat,
        "Trading Days": total,
        "Non-Flat Days": non_flat,
        # Win Rate intentionally excludes flat days.
        "Win Rate": positive / non_flat if non_flat else np.nan,
        # Positive-Day Rate includes flat days in the denominator for reference.
        "Positive-Day Rate": positive / total if total else np.nan,
        "Average Winning Day": avg_win,
        "Average Losing Day": avg_loss,
        "Win/Loss Payoff Ratio": payoff_ratio,
        "Total Return": total_return,
        "CAGR": history_cagr,
        "Max Drawdown": max_drawdown(s),
    }


def trading_day_comparison(prices, tickers):
    """Return own-history and common-period trading-day comparisons."""
    own_rows = []
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        stats = trading_day_stats(prices[ticker])
        if stats:
            own_rows.append({"Symbol": ticker, **stats})

    own = pd.DataFrame(own_rows)
    if not own.empty:
        own = own.sort_values(["Win Rate", "CAGR"], ascending=[False, False]).reset_index(drop=True)
        own.insert(0, "Win-Rate Rank", range(1, len(own) + 1))

    available = [t for t in tickers if t in prices.columns]
    common_rows = []
    common_start = None
    common_end = None

    if len(available) >= 2:
        aligned = prices[available].dropna(how="any").sort_index()
        if len(aligned) >= 2:
            common_start = aligned.index[0].date()
            common_end = aligned.index[-1].date()

            for ticker in available:
                stats = trading_day_stats(aligned[ticker])
                if stats:
                    common_rows.append({"Symbol": ticker, **stats})

    common = pd.DataFrame(common_rows)
    if not common.empty:
        common = common.sort_values(["Win Rate", "CAGR"], ascending=[False, False]).reset_index(drop=True)
        common.insert(0, "Win-Rate Rank", range(1, len(common) + 1))

    return own, common, common_start, common_end


def build_metrics(prices, universe):
    meta = universe.set_index("Symbol", drop=False)
    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty:
            continue
        r = {"Symbol": t}
        for p, n in WINDOWS.items():
            r[p] = trailing_return(s, n)

        r["3Y CAGR"] = cagr(s, 3)
        r["5Y CAGR"] = cagr(s, 5)
        r["10Y CAGR"] = cagr(s, 10)
        r["Volatility"] = volatility(s)
        r["Max Drawdown"] = max_drawdown(s)
        r["Latest Price"] = float(s.iloc[-1])
        r["Price Date"] = s.index[-1].date()

        if t in meta.index:
            m = meta.loc[t]
            r["Fund Name"] = m.get("Fund Name", "")
            r["Assets"] = m.get("Assets", np.nan)
            r["Leverage"] = m.get("Leverage", "")

            fallbacks = {
                "1Y": m.get("CAGR 1Y", np.nan),
                "3Y CAGR": m.get("CAGR 3Y", np.nan),
                "5Y CAGR": m.get("CAGR 5Y", np.nan),
                "10Y CAGR": m.get("CAGR 10Y", np.nan),
            }
            for k, v in fallbacks.items():
                if pd.isna(r.get(k)) and pd.notna(v):
                    r[k] = float(v)

        rows.append(r)

    return pd.DataFrame(rows)

def score_strategy(metrics, strategy):
    df = metrics.copy()
    weights = STRATEGIES[strategy]
    score = pd.Series(0.0, index=df.index)
    weight_used = pd.Series(0.0, index=df.index)

    for c, w in weights.items():
        if c not in df.columns:
            continue
        valid = df[c].notna()
        rank = df[c].rank(pct=True, method="average")
        score += rank.fillna(0) * w
        weight_used += valid.astype(float) * w

    df["Strategy Score"] = np.where(weight_used > 0, score / weight_used, np.nan)
    df["Positive Periods"] = df[[c for c in PERIODS if c in df.columns]].gt(0).sum(axis=1)

    if strategy == "Consistency":
        avail = df[[c for c in PERIODS if c in df.columns]].notna().sum(axis=1)
        breadth = np.where(avail > 0, df["Positive Periods"] / avail, 0)
        vol_penalty = df["Volatility"].rank(pct=True).fillna(0.5)
        dd_penalty = df["Max Drawdown"].abs().rank(pct=True).fillna(0.5)
        df["Strategy Score"] = (
            0.65 * df["Strategy Score"]
            + 0.25 * breadth
            + 0.05 * (1 - vol_penalty)
            + 0.05 * (1 - dd_penalty)
        )

    return df.sort_values("Strategy Score", ascending=False, na_position="last")

def get_price_on_or_after(s, d):
    x = s.dropna()
    x = x[x.index >= pd.Timestamp(d)]
    if x.empty:
        return None, None
    return x.index[0], float(x.iloc[0])

def get_price_on_or_before(s, d):
    x = s.dropna()
    x = x[x.index <= pd.Timestamp(d)]
    if x.empty:
        return None, None
    return x.index[-1], float(x.iloc[-1])

def what_if(prices, ticker, amount, start, end):
    if ticker not in prices.columns:
        return None
    s = prices[ticker]
    sd, sp = get_price_on_or_after(s, start)
    ed, ep = get_price_on_or_before(s, end)
    if sp is None or ep is None or ed <= sd:
        return None
    shares = amount / sp
    ending = shares * ep
    years = (ed - sd).days / 365.25
    return {
        "Start Date": sd.date(),
        "End Date": ed.date(),
        "Start Price": sp,
        "End Price": ep,
        "Shares": shares,
        "Initial Investment": amount,
        "Ending Value": ending,
        "Profit": ending - amount,
        "Return": ending / amount - 1,
        "CAGR": (ending / amount) ** (1 / years) - 1 if years > 0 else np.nan,
    }

def portfolio_backtest(prices, allocations, start, end):
    details = []
    total_initial = 0.0
    total_ending = 0.0

    for ticker, amount in allocations.items():
        if amount <= 0:
            continue
        result = what_if(prices, ticker, amount, start, end)
        if result is None:
            continue
        details.append({"Symbol": ticker, **result})
        total_initial += amount
        total_ending += result["Ending Value"]

    if total_initial <= 0:
        return None, pd.DataFrame()

    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    summary = {
        "Initial": total_initial,
        "Ending": total_ending,
        "Profit": total_ending - total_initial,
        "Return": total_ending / total_initial - 1,
        "CAGR": (total_ending / total_initial) ** (1 / years) - 1 if years > 0 else np.nan,
    }
    return summary, pd.DataFrame(details)

# ============================================================
# SIDEBAR / UNIVERSE
# ============================================================

universe = load_universe()

st.sidebar.header("ETF Universe")
size_choice = st.sidebar.selectbox(
    "Ranking universe size",
    [50, 100, 250, 500, "All"],
    index=1,
)
if size_choice == "All":
    selected_universe = universe.copy()
else:
    selected_universe = universe.head(int(size_choice)).copy()

st.sidebar.caption(
    f"{len(selected_universe):,} ETFs selected from {len(universe):,} in your workbook."
)

if st.sidebar.button("Refresh market data"):
    fetch_prices.clear()
    st.rerun()

# Use 10Y for rankings; user controls universe size.
with st.spinner("Loading ETF price history..."):
    prices, failed = fetch_prices(tuple(selected_universe["Symbol"].tolist()), "10y")
    metrics = build_metrics(prices, selected_universe) if not prices.empty else pd.DataFrame()

# ============================================================
# APP
# ============================================================

st.title("📈 ETF Performance & Portfolio Analyzer")
st.caption(
    "Rank ETFs across multiple horizons, compare trading-day win/loss statistics, test hypothetical "
    "investments, backtest portfolios, and compare historical market performance."
)

tabs = st.tabs([
    "🏆 ETF Leaders",
    "🔎 ETF Analyzer",
    "💰 What-If",
    "📊 Portfolio",
    "⭐ Rankings",
])

# ---------------- Leaders ----------------
with tabs[0]:
    st.subheader("ETF Leaders by Time Period")

    if metrics.empty:
        st.error("Market data could not be loaded for the selected ETF universe.")
    else:
        top_n = st.slider("Show top", 5, 25, 10)
        period = st.selectbox("Period", PERIODS, index=5)

        table = metrics.dropna(subset=[period]).sort_values(period, ascending=False).head(top_n)
        cols = ["Symbol", "Fund Name", period, "Latest Price", "Assets", "Volatility", "Max Drawdown"]
        cols = [c for c in cols if c in table.columns]
        st.dataframe(
            table[cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                period: st.column_config.NumberColumn(format="%.2f%%"),
                "Latest Price": st.column_config.NumberColumn(format="$%.2f"),
                "Volatility": st.column_config.NumberColumn(format="%.2f%%"),
                "Max Drawdown": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        st.markdown("#### Top ETF in every period")
        leader_rows = []
        for p in PERIODS:
            x = metrics.dropna(subset=[p]).sort_values(p, ascending=False)
            if not x.empty:
                r = x.iloc[0]
                leader_rows.append({
                    "Period": p,
                    "Symbol": r["Symbol"],
                    "Fund Name": r.get("Fund Name", ""),
                    "Return": r[p],
                })
        leaders = pd.DataFrame(leader_rows)
        st.dataframe(
            leaders,
            use_container_width=True,
            hide_index=True,
            column_config={"Return": st.column_config.NumberColumn(format="%.2f%%")},
        )

        st.markdown("#### Repeat leaders")
        top10_sets = []
        for p in PERIODS:
            top10_sets.extend(
                metrics.dropna(subset=[p]).nlargest(10, p)["Symbol"].tolist()
            )
        counts = pd.Series(top10_sets).value_counts().rename("Top-10 Appearances").reset_index()
        counts.columns = ["Symbol", "Top-10 Appearances"]
        counts = counts.merge(
            universe[["Symbol", "Fund Name"]].drop_duplicates(),
            on="Symbol",
            how="left",
        )
        st.dataframe(counts.head(25), use_container_width=True, hide_index=True)

# ---------------- Analyzer ----------------
with tabs[1]:
    st.subheader("ETF Analyzer")
    symbols = universe["Symbol"].tolist()
    ticker = st.selectbox("ETF", symbols, index=0)

    # Fetch ticker separately so any workbook ETF can be viewed even if outside ranking subset.
    t_prices, _ = fetch_prices((ticker,), "10y")
    if ticker not in t_prices.columns:
        st.error("Price history unavailable for this ETF.")
    else:
        s = t_prices[ticker].dropna()
        one = build_metrics(t_prices, universe)
        row = one.iloc[0] if not one.empty else None

        meta = universe[universe["Symbol"] == ticker].iloc[0]
        st.markdown(f"### {ticker} — {meta.get('Fund Name', '')}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest price", money(float(s.iloc[-1])))
        c2.metric("1-year return", pct(row["1Y"]) if row is not None else "—")
        c3.metric("5-year CAGR", pct(row["5Y CAGR"]) if row is not None else "—")
        c4.metric("Max drawdown", pct(row["Max Drawdown"]) if row is not None else "—")

        chart_period = st.selectbox("Chart history", ["1Y", "3Y", "5Y", "10Y"], index=2)
        days = {"1Y": 365, "3Y": 365*3, "5Y": 365*5, "10Y": 365*10}[chart_period]
        chart_s = s[s.index >= s.index[-1] - pd.Timedelta(days=days)]
        st.line_chart(chart_s)

        metric_table = pd.DataFrame({
            "Period": PERIODS,
            "Return": [row.get(p, np.nan) for p in PERIODS],
        })
        st.dataframe(
            metric_table,
            hide_index=True,
            use_container_width=True,
            column_config={"Return": st.column_config.NumberColumn(format="%.2f%%")},
        )

    st.markdown("---")
    st.markdown("### Trading-Day Win/Loss Comparison")
    st.caption(
        "Win day = adjusted closing price finished above the prior trading day's adjusted close. "
        "Win Rate excludes flat days. The table also shows average winning/losing day, payoff ratio, "
        "CAGR, total return, and max drawdown. Use Own History for each ETF's full available history "
        "and Common Period for a fair same-date comparison."
    )

    default_compare = [t for t in ["VT", "VOO", "SPY"] if t in universe["Symbol"].tolist()]
    compare_tickers = st.multiselect(
        "Compare ETFs",
        universe["Symbol"].tolist(),
        default=default_compare,
        max_selections=6,
        key="positive_day_tickers",
    )

    if st.button("Run trading-day comparison", key="run_positive_day_comparison"):
        if len(compare_tickers) < 2:
            st.warning("Choose at least two ETFs.")
        else:
            with st.spinner("Loading full daily price history..."):
                compare_prices, compare_failed = fetch_prices(tuple(compare_tickers), "max")

            if compare_prices.empty:
                st.error("Full price history could not be loaded.")
            else:
                own_stats, common_stats, common_start, common_end = trading_day_comparison(
                    compare_prices, compare_tickers
                )

                st.markdown("#### Since each ETF's own available inception")
                if not own_stats.empty:
                    st.dataframe(
                        own_stats,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Win Rate": st.column_config.NumberColumn(format="%.2f%%"),
                            "Positive-Day Rate": st.column_config.NumberColumn(format="%.2f%%"),
                            "Average Winning Day": st.column_config.NumberColumn(format="%.3f%%"),
                            "Average Losing Day": st.column_config.NumberColumn(format="%.3f%%"),
                            "Win/Loss Payoff Ratio": st.column_config.NumberColumn(format="%.2f"),
                            "Total Return": st.column_config.NumberColumn(format="%.2f%%"),
                            "CAGR": st.column_config.NumberColumn(format="%.2f%%"),
                            "Max Drawdown": st.column_config.NumberColumn(format="%.2f%%"),
                        },
                    )

                st.markdown("#### Same-date comparison")
                if not common_stats.empty:
                    st.caption(
                        f"Common trading period: {common_start} through {common_end}. "
                        "This removes the age difference between the ETFs."
                    )
                    st.dataframe(
                        common_stats,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Win Rate": st.column_config.NumberColumn(format="%.2f%%"),
                            "Positive-Day Rate": st.column_config.NumberColumn(format="%.2f%%"),
                            "Average Winning Day": st.column_config.NumberColumn(format="%.3f%%"),
                            "Average Losing Day": st.column_config.NumberColumn(format="%.3f%%"),
                            "Win/Loss Payoff Ratio": st.column_config.NumberColumn(format="%.2f"),
                            "Total Return": st.column_config.NumberColumn(format="%.2f%%"),
                            "CAGR": st.column_config.NumberColumn(format="%.2f%%"),
                            "Max Drawdown": st.column_config.NumberColumn(format="%.2f%%"),
                        },
                    )

                    chart_df = common_stats[["Symbol", "Win Rate"]].copy()
                    chart_df["Win Rate"] = chart_df["Win Rate"] * 100
                    st.markdown("#### Win-rate ranking")
                    st.bar_chart(
                        chart_df.set_index("Symbol"),
                        y="Win Rate",
                    )

                if compare_failed:
                    st.warning(
                        "Price history was unavailable for: " + ", ".join(compare_failed)
                    )

# ---------------- What If ----------------
with tabs[2]:
    st.subheader("What If I Had Invested?")
    c1, c2 = st.columns(2)
    with c1:
        w_ticker = st.selectbox("ETF", universe["Symbol"].tolist(), key="whatif_ticker")
        amount = st.number_input("Initial investment", min_value=1.0, value=10000.0, step=500.0)
    with c2:
        default_start = date.today() - timedelta(days=365*5)
        start = st.date_input("Start date", value=default_start)
        end = st.date_input("End date", value=date.today())

    wp, _ = fetch_prices((w_ticker,), "10y")
    result = what_if(wp, w_ticker, amount, start, end) if w_ticker in wp.columns else None

    if result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial", money(result["Initial Investment"]))
        c2.metric("Ending value", money(result["Ending Value"]))
        c3.metric("Profit", money(result["Profit"]))
        c4.metric("Return", pct(result["Return"]))
        st.write(
            f"Historical shares purchased: **{result['Shares']:,.4f}** | "
            f"Historical CAGR: **{pct(result['CAGR'])}**"
        )
        st.caption(
            f"Uses adjusted closing prices from {result['Start Date']} through {result['End Date']}."
        )
    else:
        st.info("Choose dates with available history.")

# ---------------- Portfolio ----------------
with tabs[3]:
    st.subheader("Historical Portfolio Backtest")

    portfolio_tickers = st.multiselect(
        "Choose ETFs",
        universe["Symbol"].tolist(),
        default=universe["Symbol"].tolist()[:3],
        max_selections=12,
    )

    p_start = st.date_input(
        "Portfolio start date",
        value=date.today() - timedelta(days=365*5),
        key="p_start",
    )
    p_end = st.date_input("Portfolio end date", value=date.today(), key="p_end")

    allocations = {}
    if portfolio_tickers:
        st.markdown("#### Starting allocations")
        cols = st.columns(min(4, len(portfolio_tickers)))
        for i, t in enumerate(portfolio_tickers):
            with cols[i % len(cols)]:
                allocations[t] = st.number_input(
                    f"{t} starting $",
                    min_value=0.0,
                    value=5000.0,
                    step=500.0,
                    key=f"alloc_{t}",
                )

        pp, _ = fetch_prices(tuple(portfolio_tickers), "10y")
        summary, details = portfolio_backtest(pp, allocations, p_start, p_end)

        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Starting portfolio", money(summary["Initial"]))
            c2.metric("Ending value", money(summary["Ending"]))
            c3.metric("Profit", money(summary["Profit"]))
            c4.metric("Portfolio return", pct(summary["Return"]))

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Start Price": st.column_config.NumberColumn(format="$%.2f"),
                    "End Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Initial Investment": st.column_config.NumberColumn(format="$%.2f"),
                    "Ending Value": st.column_config.NumberColumn(format="$%.2f"),
                    "Profit": st.column_config.NumberColumn(format="$%.2f"),
                    "Return": st.column_config.NumberColumn(format="%.2f%%"),
                    "CAGR": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )
        else:
            st.info("Enter at least one positive allocation with available price history.")

# ---------------- Rankings ----------------
with tabs[4]:
    st.subheader("Rankings & Recommendation Signals")
    st.caption(
        "These are quantitative ranking signals, not personalized investment advice. "
        "Scores are based on historical relative performance and, for Consistency, risk measures."
    )

    strategy = st.selectbox("Strategy", list(STRATEGIES.keys()))

    if metrics.empty:
        st.error("Ranking data unavailable.")
    else:
        ranked = score_strategy(metrics, strategy)
        top = ranked.head(25).copy()

        def label(row):
            score = row["Strategy Score"]
            if pd.isna(score):
                return "Insufficient data"
            if score >= 0.90:
                return "Top-ranked"
            if score >= 0.75:
                return "Strong"
            if score >= 0.55:
                return "Above average"
            if score >= 0.35:
                return "Middle"
            return "Weak relative score"

        top["Signal"] = top.apply(label, axis=1)
        show_cols = [
            "Symbol", "Fund Name", "Strategy Score", "Signal",
            "Positive Periods", "1M", "3M", "6M", "1Y",
            "3Y CAGR", "5Y CAGR", "10Y CAGR", "Volatility", "Max Drawdown"
        ]
        show_cols = [c for c in show_cols if c in top.columns]

        st.dataframe(
            top[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strategy Score": st.column_config.ProgressColumn(min_value=0, max_value=1),
                "1M": st.column_config.NumberColumn(format="%.2f%%"),
                "3M": st.column_config.NumberColumn(format="%.2f%%"),
                "6M": st.column_config.NumberColumn(format="%.2f%%"),
                "1Y": st.column_config.NumberColumn(format="%.2f%%"),
                "3Y CAGR": st.column_config.NumberColumn(format="%.2f%%"),
                "5Y CAGR": st.column_config.NumberColumn(format="%.2f%%"),
                "10Y CAGR": st.column_config.NumberColumn(format="%.2f%%"),
                "Volatility": st.column_config.NumberColumn(format="%.2f%%"),
                "Max Drawdown": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

# Footer
if failed:
    st.caption(
        f"Some ticker histories were unavailable in the selected ranking universe: "
        f"{', '.join(failed[:15])}"
        + ("..." if len(failed) > 15 else "")
    )

st.caption(
    "Historical performance does not guarantee future results. "
    "Ranking signals are informational and should not be treated as personalized investment advice."
)
