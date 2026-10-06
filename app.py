from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


APP_TITLE = "Trading Day & Investment Return Analyzer"
FREQUENCIES = [
    "Daily",
    "Weekly",
    "Semimonthly",
    "Monthly",
    "Quarterly",
    "Annually",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_tickers(raw: str) -> List[str]:
    """Accept comma-, semicolon-, space-, or newline-separated symbols."""
    cleaned = (
        raw.replace(",", " ")
        .replace(";", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    tickers: List[str] = []
    seen = set()

    for token in cleaned.split():
        ticker = token.strip().upper()
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)

    return tickers


@st.cache_data(ttl="30m", show_spinner=False)
def download_prices(
    tickers: Tuple[str, ...],
    start_date: str,
    inclusive_end_date: str,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices with a robust Yahoo/yfinance fallback.

    Flow:
      1. Try one multi-ticker yf.download request.
      2. If a ticker is missing, retry it individually with Ticker.history().
      3. If no ticker succeeds, raise an error instead of returning an empty
         DataFrame. Streamlit therefore will NOT cache an empty failure.

    yfinance treats `end` as exclusive, so one calendar day is added to make
    the app's end-date control inclusive.
    """
    if not tickers:
        raise ValueError("No ticker symbols were supplied.")

    yf_end = (
        pd.Timestamp(inclusive_end_date) + pd.Timedelta(days=1)
    ).date().isoformat()

    successful: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # ATTEMPT 1 — MULTI-TICKER DOWNLOAD
    # ------------------------------------------------------------------
    try:
        data = yf.download(
            tickers=list(tickers),
            start=start_date,
            end=yf_end,
            interval="1d",
            auto_adjust=True,
            actions=False,
                keepna=False,
            progress=False,
            threads=False,
            group_by="column",
            multi_level_index=True,
            timeout=30,
        )

        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                level0 = data.columns.get_level_values(0)

                if "Close" in level0:
                    close = data["Close"].copy()

                    if isinstance(close, pd.Series):
                        close = close.to_frame(name=tickers[0])

                    for ticker in tickers:
                        if ticker in close.columns:
                            s = pd.to_numeric(close[ticker], errors="coerce").dropna()
                            if not s.empty:
                                successful[ticker] = s

            else:
                # Defensive single-ticker shape.
                if "Close" in data.columns and len(tickers) == 1:
                    s = pd.to_numeric(data["Close"], errors="coerce").dropna()
                    if not s.empty:
                        successful[tickers[0]] = s

    except Exception as exc:
        errors["BATCH"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # ATTEMPT 2 — INDIVIDUAL TICKER FALLBACK
    # ------------------------------------------------------------------
    missing = [ticker for ticker in tickers if ticker not in successful]

    for ticker in missing:
        try:
            hist = yf.Ticker(ticker).history(
                start=start_date,
                end=yf_end,
                interval="1d",
                auto_adjust=True,
                actions=False,
                        keepna=False,
                timeout=30,
                raise_errors=True,
            )

            if hist is not None and not hist.empty and "Close" in hist.columns:
                s = pd.to_numeric(hist["Close"], errors="coerce").dropna()
                if not s.empty:
                    successful[ticker] = s
                    continue

            errors[ticker] = "Yahoo Finance returned no daily price rows."

        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"

    if not successful:
        details = " | ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(
            "Yahoo Finance/yfinance returned no usable history for any requested "
            f"ticker. {details}"
        )

    # ------------------------------------------------------------------
    # NORMALIZE OUTPUT
    # ------------------------------------------------------------------
    normalized: Dict[str, pd.Series] = {}

    for ticker, series in successful.items():
        s = series.copy()
        idx = pd.to_datetime(s.index)

        # Remove timezone safely if Yahoo supplied one.
        try:
            if idx.tz is not None:
                idx = idx.tz_convert(None)
        except Exception:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass

        s.index = idx
        s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()
        normalized[ticker] = s

    close = pd.DataFrame(normalized).sort_index()
    close = close.reindex(columns=[t for t in tickers if t in close.columns])

    if close.empty:
        raise RuntimeError(
            "Price rows were downloaded but could not be normalized into a usable table."
        )

    return close


def compute_daily_stats(series: pd.Series) -> Dict[str, float]:
    s = series.dropna().astype(float)
    returns = s.pct_change(fill_method=None).dropna()

    positive = int((returns > 0).sum())
    negative = int((returns < 0).sum())
    flat = int((returns == 0).sum())
    directional = positive + negative

    win_pct = (positive / directional * 100.0) if directional else np.nan
    loss_pct = (negative / directional * 100.0) if directional else np.nan

    if len(s) >= 2:
        elapsed_days = (s.index[-1] - s.index[0]).days
        years = elapsed_days / 365.2425
        total_return = s.iloc[-1] / s.iloc[0] - 1.0
        cagr = ((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    else:
        elapsed_days = 0
        years = 0
        total_return = np.nan
        cagr = np.nan

    return {
        "trading_days": int(len(s)),
        "return_observations": int(len(returns)),
        "positive_days": positive,
        "negative_days": negative,
        "flat_days": flat,
        "win_pct": win_pct,
        "loss_pct": loss_pct,
        "asset_total_return": total_return,
        "asset_cagr": cagr,
        "first_price": float(s.iloc[0]) if len(s) else np.nan,
        "last_price": float(s.iloc[-1]) if len(s) else np.nan,
        "first_date": s.index[0] if len(s) else pd.NaT,
        "last_date": s.index[-1] if len(s) else pd.NaT,
        "years": years,
    }


def _scheduled_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
) -> pd.DatetimeIndex:
    """
    Produce calendar contribution targets.

    Actual purchases are moved to the first available trading day ON OR AFTER
    each target date.

    Semimonthly = 1st and 15th of each month.
    Weekly       = every 7 calendar days from the first available trading date.
    Monthly      = first calendar day of each month.
    Quarterly    = Jan/Apr/Jul/Oct first calendar day.
    Annually     = Jan 1 of each year.
    Daily        = handled directly from the trading-day index.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if frequency == "Weekly":
        return pd.date_range(start=start, end=end, freq="7D")

    if frequency == "Semimonthly":
        months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")
        dates = []
        for month in months:
            first = month.start_time.normalize()
            fifteenth = first + pd.Timedelta(days=14)
            if start <= first <= end:
                dates.append(first)
            if start <= fifteenth <= end:
                dates.append(fifteenth)
        return pd.DatetimeIndex(dates)

    if frequency == "Monthly":
        dates = pd.date_range(start=start.to_period("M").start_time, end=end, freq="MS")
        return dates[dates >= start]

    if frequency == "Quarterly":
        dates = pd.date_range(start=start.to_period("Q").start_time, end=end, freq="QS")
        return dates[dates >= start]

    if frequency == "Annually":
        dates = pd.date_range(start=pd.Timestamp(year=start.year, month=1, day=1), end=end, freq="YS")
        return dates[dates >= start]

    raise ValueError(f"Unsupported frequency: {frequency}")


def map_targets_to_trading_days(
    trading_days: pd.DatetimeIndex,
    targets: Iterable[pd.Timestamp],
) -> pd.DatetimeIndex:
    """Map target dates to the first available trading day on/after each target."""
    trading_days = pd.DatetimeIndex(trading_days).sort_values().unique()
    if len(trading_days) == 0:
        return pd.DatetimeIndex([])

    mapped = []
    for target in targets:
        pos = trading_days.searchsorted(pd.Timestamp(target), side="left")
        if pos < len(trading_days):
            mapped.append(trading_days[pos])

    return pd.DatetimeIndex(mapped).unique().sort_values()


def contribution_dates(prices: pd.Series, frequency: str) -> pd.DatetimeIndex:
    s = prices.dropna()
    if s.empty:
        return pd.DatetimeIndex([])

    days = pd.DatetimeIndex(s.index)
    if frequency == "Daily":
        return days

    targets = _scheduled_dates(days[0], days[-1], frequency)
    return map_targets_to_trading_days(days, targets)


def xnpv(rate: float, cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    if rate <= -0.999999:
        return np.inf

    t0 = cashflows[0][0]
    return sum(
        amount / ((1.0 + rate) ** (((dt - t0).days / 365.2425)))
        for dt, amount in cashflows
    )


def xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    """Dependency-free XIRR via bracket expansion + bisection."""
    if len(cashflows) < 2:
        return np.nan

    values = [v for _, v in cashflows]
    if not (any(v < 0 for v in values) and any(v > 0 for v in values)):
        return np.nan

    low = -0.9999
    high = 1.0

    f_low = xnpv(low, cashflows)
    f_high = xnpv(high, cashflows)

    # Expand upper bracket for unusually strong returns.
    for _ in range(60):
        if np.sign(f_low) != np.sign(f_high):
            break
        high *= 2.0
        f_high = xnpv(high, cashflows)
        if high > 1_000_000:
            return np.nan
    else:
        return np.nan

    for _ in range(200):
        mid = (low + high) / 2.0
        f_mid = xnpv(mid, cashflows)

        if abs(f_mid) < 1e-8:
            return mid

        if np.sign(f_low) == np.sign(f_mid):
            low = mid
            f_low = f_mid
        else:
            high = mid

    return (low + high) / 2.0


@dataclass
class InvestmentResult:
    summary: Dict[str, float]
    history: pd.DataFrame


def simulate_investment(
    prices: pd.Series,
    initial_investment: float,
    recurring_investment: float,
    frequency: str,
) -> InvestmentResult:
    s = prices.dropna().astype(float).sort_index()
    if s.empty:
        return InvestmentResult({}, pd.DataFrame())

    recurring_days = set(contribution_dates(s, frequency))
    shares = 0.0
    total_contributed = 0.0
    rows = []
    cashflows: List[Tuple[pd.Timestamp, float]] = []

    # Initial investment occurs on the first available trading day.
    first_dt = s.index[0]
    first_price = float(s.iloc[0])
    if initial_investment > 0:
        shares += initial_investment / first_price
        total_contributed += initial_investment
        cashflows.append((first_dt, -float(initial_investment)))

    for dt, price in s.items():
        contribution = 0.0

        # Avoid double-counting the first day as both the initial deposit and a
        # recurring contribution only if the user entered zero recurring amount.
        # Otherwise, the recurring plan begins immediately, which is transparent
        # and consistent across frequencies.
        if dt in recurring_days and recurring_investment > 0:
            contribution = float(recurring_investment)
            shares += contribution / float(price)
            total_contributed += contribution
            cashflows.append((dt, -contribution))

        value = shares * float(price)
        rows.append(
            {
                "Date": dt,
                "Price": float(price),
                "Contribution": contribution,
                "Cumulative Contributions": total_contributed,
                "Shares": shares,
                "Portfolio Value": value,
            }
        )

    history = pd.DataFrame(rows).set_index("Date")
    ending_value = float(history["Portfolio Value"].iloc[-1])
    gain = ending_value - total_contributed
    return_on_contributed = (
        gain / total_contributed if total_contributed > 0 else np.nan
    )

    if ending_value > 0:
        cashflows.append((history.index[-1], ending_value))
    money_weighted_return = xirr(cashflows)

    summary = {
        "initial_investment": float(initial_investment),
        "recurring_investment": float(recurring_investment),
        "contribution_count": int((history["Contribution"] > 0).sum()),
        "total_contributed": float(total_contributed),
        "ending_value": ending_value,
        "gain": float(gain),
        "return_on_contributed": float(return_on_contributed),
        "money_weighted_return": float(money_weighted_return),
        "shares": float(shares),
    }
    return InvestmentResult(summary, history)


def pct(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def money(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def integer(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def build_csv(
    summary_df: pd.DataFrame,
    investment_df: pd.DataFrame,
) -> bytes:
    parts = [
        "TRADING DAY STATISTICS\n",
        summary_df.to_csv(index=False),
        "\nINVESTMENT STATISTICS\n",
        investment_df.to_csv(index=False),
    ]
    return "".join(parts).encode("utf-8")


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📈 Trading Day & Investment Return Analyzer")
st.caption(
    "Compare positive vs. negative trading days, historical return/CAGR, "
    "and recurring-investment outcomes for one or many tickers."
)

with st.sidebar:
    st.header("Analysis Inputs")

    ticker_text = st.text_area(
        "Ticker(s)",
        value="SPY, VT",
        help="Enter one or many Yahoo Finance symbols separated by commas, spaces, or new lines.",
        height=90,
    )

    today = date.today()
    default_start = date(today.year - 10, today.month, min(today.day, 28))

    start_date = st.date_input(
        "Start date",
        value=default_start,
        max_value=today,
    )
    end_date = st.date_input(
        "End date",
        value=today,
        max_value=today,
    )

    st.divider()
    st.header("Investment Simulation")

    initial_investment = st.number_input(
        "Initial investment per ticker",
        min_value=0.0,
        value=5000.0,
        step=500.0,
        format="%.2f",
    )
    recurring_investment = st.number_input(
        "Recurring contribution per ticker",
        min_value=0.0,
        value=300.0,
        step=25.0,
        format="%.2f",
    )
    frequency = st.selectbox(
        "Contribution frequency",
        FREQUENCIES,
        index=3,
    )

    analyze = st.button("Run Analysis", type="primary", use_container_width=True)

st.info(
    "**Method:** A positive day means adjusted close > prior adjusted close; "
    "a negative day means adjusted close < prior adjusted close. Flat days are "
    "reported separately and excluded from win/loss percentages. Contributions "
    "buy fractional shares at the adjusted closing price on the scheduled "
    "trading day."
)

if not analyze:
    st.stop()

tickers = parse_tickers(ticker_text)

if not tickers:
    st.error("Enter at least one ticker.")
    st.stop()

if len(tickers) > 25:
    st.error("Please analyze 25 or fewer tickers at one time.")
    st.stop()

if start_date >= end_date:
    st.error("The start date must be earlier than the end date.")
    st.stop()

with st.spinner("Downloading and analyzing market data..."):
    try:
        close = download_prices(
            tuple(tickers),
            start_date.isoformat(),
            end_date.isoformat(),
        )
    except Exception as exc:
        st.error("Yahoo Finance could not return usable market data.")
        st.code(str(exc), language=None)
        st.info(
            "Try Run Analysis again in a few seconds. If this works locally but "
            "not on Streamlit Community Cloud, Yahoo may be temporarily rate-limiting "
            "the cloud server. The app now retries each ticker individually before "
            "showing this message."
        )
        st.stop()

available = list(close.columns)
missing = [t for t in tickers if t not in available]

st.success(
    "Loaded Yahoo Finance history for: " + ", ".join(available)
)

if missing:
    st.warning(
        "No usable adjusted-close data was returned for: "
        + ", ".join(missing)
    )

stats_rows = []
investment_rows = []
histories: Dict[str, pd.DataFrame] = {}

for ticker in available:
    series = close[ticker].dropna()
    if len(series) < 2:
        continue

    stats = compute_daily_stats(series)
    investment = simulate_investment(
        series,
        float(initial_investment),
        float(recurring_investment),
        frequency,
    )
    histories[ticker] = investment.history

    stats_rows.append(
        {
            "Ticker": ticker,
            "First Trading Date": stats["first_date"].date(),
            "Last Trading Date": stats["last_date"].date(),
            "Trading Days": stats["trading_days"],
            "Positive Days": stats["positive_days"],
            "Negative Days": stats["negative_days"],
            "Flat Days": stats["flat_days"],
            "Win %": stats["win_pct"],
            "Loss %": stats["loss_pct"],
            "Buy & Hold Return %": stats["asset_total_return"] * 100,
            "CAGR %": stats["asset_cagr"] * 100,
            "Start Price": stats["first_price"],
            "End Price": stats["last_price"],
        }
    )

    inv = investment.summary
    investment_rows.append(
        {
            "Ticker": ticker,
            "Frequency": frequency,
            "Initial Investment": inv["initial_investment"],
            "Recurring Amount": inv["recurring_investment"],
            "Recurring Purchases": inv["contribution_count"],
            "Total Contributed": inv["total_contributed"],
            "Ending Value": inv["ending_value"],
            "Dollar Gain": inv["gain"],
            "Return on Contributions %": inv["return_on_contributed"] * 100,
            "Annualized Money-Weighted Return %": inv["money_weighted_return"] * 100,
            "Ending Shares": inv["shares"],
        }
    )

if not stats_rows:
    st.error("The returned symbols did not contain enough observations to analyze.")
    st.stop()

stats_df = pd.DataFrame(stats_rows)
investment_df = pd.DataFrame(investment_rows)

# -----------------------------------------------------------------------------
# Headline metrics
# -----------------------------------------------------------------------------
st.subheader("Overview")
best_win = stats_df.loc[stats_df["Win %"].idxmax()]
best_cagr = stats_df.loc[stats_df["CAGR %"].idxmax()]
best_ending = investment_df.loc[investment_df["Ending Value"].idxmax()]

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Highest Win Rate",
    f'{best_win["Win %"]:.2f}%',
    best_win["Ticker"],
)
m2.metric(
    "Highest CAGR",
    f'{best_cagr["CAGR %"]:.2f}%',
    best_cagr["Ticker"],
)
m3.metric(
    "Highest Ending Value",
    money(best_ending["Ending Value"]),
    best_ending["Ticker"],
)
m4.metric(
    "Contribution Schedule",
    frequency,
    f"{money(recurring_investment)} each",
)

# -----------------------------------------------------------------------------
# Trading-day statistics
# -----------------------------------------------------------------------------
st.subheader("Trading-Day Statistics")

display_stats = stats_df.copy()
for col in ["Win %", "Loss %", "Buy & Hold Return %", "CAGR %"]:
    display_stats[col] = display_stats[col].map(lambda x: f"{x:,.2f}%")
for col in ["Start Price", "End Price"]:
    display_stats[col] = display_stats[col].map(lambda x: f"${x:,.2f}")

st.dataframe(display_stats, use_container_width=True, hide_index=True)

day_chart = stats_df.melt(
    id_vars="Ticker",
    value_vars=["Positive Days", "Negative Days", "Flat Days"],
    var_name="Day Type",
    value_name="Count",
)
fig_days = px.bar(
    day_chart,
    x="Ticker",
    y="Count",
    color="Day Type",
    barmode="group",
    title="Positive vs. Negative vs. Flat Trading Days",
    text_auto=True,
)
fig_days.update_layout(legend_title_text="")
st.plotly_chart(fig_days, width="stretch")

win_chart = stats_df[["Ticker", "Win %", "Loss %"]].melt(
    id_vars="Ticker",
    var_name="Result",
    value_name="Percent",
)
fig_win = px.bar(
    win_chart,
    x="Ticker",
    y="Percent",
    color="Result",
    barmode="stack",
    title="Win vs. Loss Percentage (Flat Days Excluded)",
    text_auto=".2f",
)
fig_win.update_yaxes(range=[0, 100], ticksuffix="%")
fig_win.update_layout(legend_title_text="")
st.plotly_chart(fig_win, width="stretch")

# -----------------------------------------------------------------------------
# Historical growth line
# -----------------------------------------------------------------------------
st.subheader("Price Growth Comparison")
normalized = pd.DataFrame()
for ticker in available:
    s = close[ticker].dropna()
    if not s.empty:
        normalized[ticker] = s / s.iloc[0] * 100.0

normalized_long = (
    normalized.reset_index()
    .rename(columns={normalized.index.name or "index": "Date"})
    .melt(id_vars="Date", var_name="Ticker", value_name="Growth of $100")
    .dropna()
)

fig_growth = px.line(
    normalized_long,
    x="Date",
    y="Growth of $100",
    color="Ticker",
    title="Historical Price Growth — $100 Indexed at Each Ticker's First Available Date",
)
fig_growth.update_yaxes(title="Indexed Value")
st.plotly_chart(fig_growth, width="stretch")

# -----------------------------------------------------------------------------
# Investment simulation
# -----------------------------------------------------------------------------
st.subheader("Investment Simulation")
st.caption(
    f"Each ticker is simulated independently using {money(initial_investment)} "
    f"initially plus {money(recurring_investment)} {frequency.lower()}."
)

display_inv = investment_df.copy()
for col in [
    "Initial Investment",
    "Recurring Amount",
    "Total Contributed",
    "Ending Value",
    "Dollar Gain",
]:
    display_inv[col] = display_inv[col].map(lambda x: f"${x:,.2f}")
for col in [
    "Return on Contributions %",
    "Annualized Money-Weighted Return %",
]:
    display_inv[col] = display_inv[col].map(
        lambda x: "N/A" if pd.isna(x) else f"{x:,.2f}%"
    )
display_inv["Ending Shares"] = display_inv["Ending Shares"].map(lambda x: f"{x:,.6f}")

st.dataframe(display_inv, use_container_width=True, hide_index=True)

value_chart = investment_df.melt(
    id_vars="Ticker",
    value_vars=["Total Contributed", "Ending Value"],
    var_name="Measure",
    value_name="Dollars",
)
fig_value = px.bar(
    value_chart,
    x="Ticker",
    y="Dollars",
    color="Measure",
    barmode="group",
    title="Total Contributions vs. Ending Portfolio Value",
)
fig_value.update_yaxes(tickprefix="$")
fig_value.update_layout(legend_title_text="")
st.plotly_chart(fig_value, width="stretch")

portfolio_long_parts = []
for ticker, history in histories.items():
    if history.empty:
        continue
    piece = history[["Portfolio Value", "Cumulative Contributions"]].copy()
    piece["Ticker"] = ticker
    piece["Date"] = piece.index
    portfolio_long_parts.append(piece.reset_index(drop=True))

if portfolio_long_parts:
    portfolio_long = pd.concat(portfolio_long_parts, ignore_index=True)
    fig_portfolio = go.Figure()
    for ticker in portfolio_long["Ticker"].unique():
        subset = portfolio_long[portfolio_long["Ticker"] == ticker]
        fig_portfolio.add_trace(
            go.Scatter(
                x=subset["Date"],
                y=subset["Portfolio Value"],
                mode="lines",
                name=f"{ticker} value",
            )
        )
    fig_portfolio.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        yaxis_tickprefix="$",
        hovermode="x unified",
    )
    st.plotly_chart(fig_portfolio, width="stretch")

# -----------------------------------------------------------------------------
# Frequency comparison: all requested frequencies
# -----------------------------------------------------------------------------
st.subheader("All Contribution Frequencies")
st.caption(
    "This table applies the same recurring dollar amount to every schedule so "
    "you can compare daily, weekly, semimonthly, monthly, quarterly, and annual "
    "contribution patterns. Because the amount is per contribution, total dollars "
    "contributed differ substantially by frequency."
)

frequency_rows = []
for ticker in available:
    series = close[ticker].dropna()
    if len(series) < 2:
        continue
    for freq in FREQUENCIES:
        result = simulate_investment(
            series,
            float(initial_investment),
            float(recurring_investment),
            freq,
        ).summary
        frequency_rows.append(
            {
                "Ticker": ticker,
                "Frequency": freq,
                "Purchases": result["contribution_count"],
                "Total Contributed": result["total_contributed"],
                "Ending Value": result["ending_value"],
                "Dollar Gain": result["gain"],
                "Return on Contributions %": result["return_on_contributed"] * 100,
                "Annualized Money-Weighted Return %": result["money_weighted_return"] * 100,
            }
        )

freq_df = pd.DataFrame(frequency_rows)

freq_display = freq_df.copy()
for col in ["Total Contributed", "Ending Value", "Dollar Gain"]:
    freq_display[col] = freq_display[col].map(lambda x: f"${x:,.2f}")
for col in ["Return on Contributions %", "Annualized Money-Weighted Return %"]:
    freq_display[col] = freq_display[col].map(
        lambda x: "N/A" if pd.isna(x) else f"{x:,.2f}%"
    )

st.dataframe(freq_display, use_container_width=True, hide_index=True)

fig_freq = px.bar(
    freq_df,
    x="Frequency",
    y="Ending Value",
    color="Ticker",
    barmode="group",
    category_orders={"Frequency": FREQUENCIES},
    title="Ending Value by Contribution Frequency",
)
fig_freq.update_yaxes(tickprefix="$")
st.plotly_chart(fig_freq, width="stretch")

# -----------------------------------------------------------------------------
# Downloads and notes
# -----------------------------------------------------------------------------
st.subheader("Export")
st.download_button(
    "Download Summary CSV",
    data=build_csv(stats_df, investment_df),
    file_name="trading_day_return_analysis.csv",
    mime="text/csv",
)

with st.expander("Calculation Notes"):
    st.markdown(
        """
- **Adjusted prices:** The app requests auto-adjusted daily Yahoo Finance prices through `yfinance`.
- **Positive day:** Adjusted close is greater than the previous available trading day's adjusted close.
- **Negative day:** Adjusted close is lower than the previous available trading day's adjusted close.
- **Flat day:** Adjusted close is unchanged. Flat days do not count as wins or losses.
- **Win %:** Positive Days ÷ (Positive Days + Negative Days).
- **Buy & Hold Return:** End adjusted price ÷ Start adjusted price − 1.
- **CAGR:** Annualized buy-and-hold price return over the actual elapsed calendar time.
- **Recurring purchases:** Fractional shares are purchased using the adjusted close.
- **Semimonthly:** Targets the 1st and 15th of each month and moves non-trading dates to the next available trading day.
- **Other calendar schedules:** Weekly uses seven-day spacing from the first available trading date; monthly, quarterly, and annual schedules use the first applicable calendar date and move to the next available trading day.
- **Return on Contributions:** (Ending Value − Total Contributed) ÷ Total Contributed. This is not annualized.
- **Annualized Money-Weighted Return:** XIRR-style annualized return using the actual contribution dates and final portfolio value.
- **Multiple tickers:** Each ticker is modeled independently using the same starting and recurring dollar amounts. The app does not split one contribution across all tickers.
- **Data source:** Yahoo Finance through the open-source `yfinance` package. Market data can contain gaps, revisions, symbol changes, or provider limitations.
        """
    )

st.caption(
    "For research and educational use. Historical results do not guarantee future returns."
)
