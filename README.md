# Trading Day & Investment Return Analyzer

A GitHub-ready Streamlit app that compares one or many market tickers across a user-selected date range.

https://tradingdayreturnanalyzer.streamlit.app/

## What it does

- Accepts one ticker or multiple tickers.
- Accepts a start date and end date.
- Counts:
  - Positive trading days
  - Negative trading days
  - Flat trading days
- Calculates:
  - Win percentage
  - Loss percentage
  - Buy-and-hold total return
  - CAGR
- Simulates an initial investment plus recurring contributions.
- Supports:
  - Daily
  - Weekly
  - Semimonthly
  - Monthly
  - Quarterly
  - Annually
- Calculates:
  - Total contributed
  - Ending value
  - Dollar gain
  - Return on contributed capital
  - Annualized money-weighted return (XIRR-style)
- Shows:
  - Statistical tables
  - Positive/negative/flat day bar chart
  - Win/loss percentage chart
  - Indexed price-growth line chart
  - Contributions vs. ending value bar chart
  - Portfolio-value line chart
  - Contribution-frequency comparison
- Exports summary statistics to CSV.

## Important behavior

Each ticker is modeled independently. If you enter SPY, VT, and QQQ with a $5,000 starting amount and $300 monthly contribution, each ticker receives its own hypothetical $5,000 + $300/month simulation.

The app uses adjusted daily prices (`auto_adjust=True`) from Yahoo Finance through `yfinance`.

### Contribution timing

- Daily: every available trading day
- Weekly: every 7 calendar days beginning with the first available trading date
- Semimonthly: 1st and 15th
- Monthly: first day of each month
- Quarterly: first day of Jan/Apr/Jul/Oct
- Annually: January 1

When a target falls on a weekend/holiday, the investment is moved to the next available trading day.

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Deploy with GitHub + Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml` (optional, included here)
3. Push/commit the files.
4. Open Streamlit Community Cloud.
5. Create a new app from your GitHub repository.
6. Set the entry point to `app.py`.
7. Deploy.

No API key is required by this version.

## Notes

Yahoo Finance/yfinance data can contain gaps or provider limitations. This tool is intended for research and educational analysis, not as a brokerage statement or guarantee of future performance.
