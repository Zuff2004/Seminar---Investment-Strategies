from pathlib import Path
import pandas as pd
import yfinance as yf
import time


# ============================================================
# 1. Project paths and parameters
# ============================================================


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "on_pn_companies_filtered.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "on_pn_historical_liquidity_2010_2025.xlsx"

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# Liquidity thresholds in BRL
BASE_LIQUIDITY_THRESHOLD = 1_000_000
STRICT_LIQUIDITY_THRESHOLD = 5_000_000

# Data availability threshold
MIN_COMMON_TRADING_DAYS = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Helper functions
# ============================================================

def parse_ticker_list(ticker_string):
    """
    Converts a string like 'PETR4, PETR5' into ['PETR4', 'PETR5'].
    """
    if pd.isna(ticker_string):
        return []

    return [
        ticker.strip().upper()
        for ticker in str(ticker_string).replace(";", ",").split(",")
        if ticker.strip()
    ]


def to_yfinance_ticker(ticker):
    """
    Converts B3 ticker to Yahoo Finance format.

    Example:
    PETR4 -> PETR4.SA
    """
    ticker = str(ticker).strip().upper()

    if ticker.endswith(".SA"):
        return ticker

    return f"{ticker}.SA"


def download_price_volume(ticker):
    """
    Downloads daily close price and volume from Yahoo Finance.
    """
    yf_ticker = to_yfinance_ticker(ticker)

    try:
        data = yf.download(
            yf_ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if data.empty:
            return pd.DataFrame()

        # Handle possible multi-index columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Close", "Volume"]].copy()
        data = data.dropna(subset=["Close", "Volume"])

        data["ticker"] = ticker
        data["yf_ticker"] = yf_ticker
        data["daily_traded_value"] = data["Close"] * data["Volume"]

        return data

    except Exception as error:
        print(f"Error downloading {ticker}: {error}")
        return pd.DataFrame()


def calculate_liquidity_metrics(ticker):
    """
    Calculates historical liquidity metrics for one ticker.
    """
    data = download_price_volume(ticker)

    if data.empty:
        return {
            "ticker": ticker,
            "yf_ticker": to_yfinance_ticker(ticker),
            "first_date": None,
            "last_date": None,
            "n_observations": 0,
            "avg_daily_volume": None,
            "median_daily_volume": None,
            "avg_daily_traded_value": None,
            "median_daily_traded_value": None,
            "data_status": "No data"
        }

    return {
        "ticker": ticker,
        "yf_ticker": to_yfinance_ticker(ticker),
        "first_date": data.index.min().date(),
        "last_date": data.index.max().date(),
        "n_observations": len(data),
        "avg_daily_volume": data["Volume"].mean(),
        "median_daily_volume": data["Volume"].median(),
        "avg_daily_traded_value": data["daily_traded_value"].mean(),
        "median_daily_traded_value": data["daily_traded_value"].median(),
        "data_status": "OK"
    }


def calculate_common_period_metrics(on_ticker, pn_ticker):
    """
    Calculates liquidity over the overlapping period of ON and PN data.
    This is important because the strategy needs both legs simultaneously.
    """
    on_data = download_price_volume(on_ticker)
    pn_data = download_price_volume(pn_ticker)

    if on_data.empty or pn_data.empty:
        return {
            "on_ticker": on_ticker,
            "pn_ticker": pn_ticker,
            "common_start_date": None,
            "common_end_date": None,
            "common_trading_days": 0,
            "on_avg_daily_value_common": None,
            "pn_avg_daily_value_common": None,
            "on_median_daily_value_common": None,
            "pn_median_daily_value_common": None,
            "pair_avg_daily_value": None,
            "pair_median_daily_value": None,
            "pair_data_status": "Missing ON or PN data"
        }

    common_dates = on_data.index.intersection(pn_data.index)

    if len(common_dates) == 0:
        return {
            "on_ticker": on_ticker,
            "pn_ticker": pn_ticker,
            "common_start_date": None,
            "common_end_date": None,
            "common_trading_days": 0,
            "on_avg_daily_value_common": None,
            "pn_avg_daily_value_common": None,
            "on_median_daily_value_common": None,
            "pn_median_daily_value_common": None,
            "pair_avg_daily_value": None,
            "pair_median_daily_value": None,
            "pair_data_status": "No overlapping dates"
        }

    on_common = on_data.loc[common_dates]
    pn_common = pn_data.loc[common_dates]

    on_avg = on_common["daily_traded_value"].mean()
    pn_avg = pn_common["daily_traded_value"].mean()

    on_median = on_common["daily_traded_value"].median()
    pn_median = pn_common["daily_traded_value"].median()

    pair_avg = min(on_avg, pn_avg)
    pair_median = min(on_median, pn_median)

    return {
        "on_ticker": on_ticker,
        "pn_ticker": pn_ticker,
        "common_start_date": common_dates.min().date(),
        "common_end_date": common_dates.max().date(),
        "common_trading_days": len(common_dates),
        "on_avg_daily_value_common": on_avg,
        "pn_avg_daily_value_common": pn_avg,
        "on_median_daily_value_common": on_median,
        "pn_median_daily_value_common": pn_median,
        "pair_avg_daily_value": pair_avg,
        "pair_median_daily_value": pair_median,
        "pair_data_status": "OK"
    }


# ============================================================
# 3. Main script
# ============================================================

def main():
    print("Reading ON/PN company universe...")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Make sure on_pn_companies_filtered.xlsx is inside data/processed."
        )

    companies = pd.read_excel(INPUT_FILE, sheet_name="ON_PN_Companies")

    print(f"Companies loaded: {len(companies)}")
    print(f"Columns: {list(companies.columns)}")

    required_columns = ["ticker_prefix", "company_names", "on_tickers", "pn_tickers"]

    for col in required_columns:
        if col not in companies.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available columns: {list(companies.columns)}"
            )

    all_pair_results = []
    selected_pair_results = []
    ticker_metrics = []

    # Cache to avoid downloading the same ticker multiple times
    ticker_metric_cache = {}

    for idx, row in companies.iterrows():
        prefix = row["ticker_prefix"]
        company_names = row["company_names"]

        on_list = parse_ticker_list(row["on_tickers"])
        pn_list = parse_ticker_list(row["pn_tickers"])

        print(f"\nProcessing {idx + 1}/{len(companies)}: {prefix}")
        print(f"ON: {on_list}")
        print(f"PN: {pn_list}")

        if len(on_list) == 0 or len(pn_list) == 0:
            continue

        # Usually there is only one ON. If there are multiple, we test all combinations.
        pair_results_for_company = []

        for on_ticker in on_list:
            if on_ticker not in ticker_metric_cache:
                ticker_metric_cache[on_ticker] = calculate_liquidity_metrics(on_ticker)
                time.sleep(0.2)

            ticker_metrics.append(ticker_metric_cache[on_ticker])

            for pn_ticker in pn_list:
                if pn_ticker not in ticker_metric_cache:
                    ticker_metric_cache[pn_ticker] = calculate_liquidity_metrics(pn_ticker)
                    time.sleep(0.2)

                ticker_metrics.append(ticker_metric_cache[pn_ticker])

                pair_metrics = calculate_common_period_metrics(on_ticker, pn_ticker)

                pair_metrics["ticker_prefix"] = prefix
                pair_metrics["company_names"] = company_names
                pair_metrics["all_on_tickers"] = ", ".join(on_list)
                pair_metrics["all_pn_tickers"] = ", ".join(pn_list)

                pair_results_for_company.append(pair_metrics)
                all_pair_results.append(pair_metrics)

                time.sleep(0.2)



        if len(pair_results_for_company) == 0:
            continue

        pair_df = pd.DataFrame(pair_results_for_company)

        # Select the PN pair with the highest pair-level average traded value
        # This automatically chooses the most liquid PN when there are multiple PNs.
        pair_df_valid = pair_df[pair_df["pair_avg_daily_value"].notna()].copy()

        if pair_df_valid.empty:
            selected = pair_df.iloc[0].to_dict()
            selected["selected_pair_reason"] = "No valid liquidity data available"
        else:
            selected = (
                pair_df_valid
                .sort_values("pair_avg_daily_value", ascending=False)
                .iloc[0]
                .to_dict()
            )
            selected["selected_pair_reason"] = "Highest pair-level average daily traded value"

        selected_pair_results.append(selected)

    all_pairs_df = pd.DataFrame(all_pair_results)
    selected_pairs_df = pd.DataFrame(selected_pair_results)
    ticker_metrics_df = pd.DataFrame(ticker_metrics).drop_duplicates(subset=["ticker"])

    # ========================================================
    # 4. Add liquidity pass/fail flags
    # ========================================================

    if not selected_pairs_df.empty:
        selected_pairs_df["passes_base_liquidity"] = (
                (selected_pairs_df["common_trading_days"] >= MIN_COMMON_TRADING_DAYS) &
                (selected_pairs_df["pair_avg_daily_value"] >= BASE_LIQUIDITY_THRESHOLD)
        )

        selected_pairs_df["passes_strict_liquidity"] = (
                (selected_pairs_df["common_trading_days"] >= MIN_COMMON_TRADING_DAYS) &
                (selected_pairs_df["pair_avg_daily_value"] >= STRICT_LIQUIDITY_THRESHOLD)
        )

        selected_pairs_df = selected_pairs_df.sort_values(
            "pair_avg_daily_value",
            ascending=False
        )

    # ========================================================
    # 5. Methodology sheet
    # ========================================================

    methodology = pd.DataFrame({
        "Item": [
            "Study period",
            "Liquidity measure",
            "Pair-level liquidity",
            "Multiple PN classes",
            "Base liquidity threshold",
            "Strict liquidity threshold",
            "Minimum common trading days",
            "Important limitation"
        ],
        "Description": [
            f"{START_DATE} to {END_DATE}",
            "Daily traded value = Close price × Volume",
            "Pair liquidity is defined as the lower average daily traded value between the ON and PN legs.",
            "If a company has multiple PN classes, the PN with the highest pair-level average daily traded value is selected.",
            f"R$ {BASE_LIQUIDITY_THRESHOLD:,.0f} average daily traded value for the weaker leg.",
            f"R$ {STRICT_LIQUIDITY_THRESHOLD:,.0f} average daily traded value for the weaker leg.",
            f"{MIN_COMMON_TRADING_DAYS} overlapping trading days.",
            "Results depend on Yahoo Finance/yfinance historical data availability and should be validated for important edge cases."
        ]
    })

    # ========================================================
    # 6. Export to Excel
    # ========================================================

    print("\nWriting output Excel...")
    print(f"Output file: {OUTPUT_FILE}")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        methodology.to_excel(writer, sheet_name="Methodology", index=False)
        selected_pairs_df.to_excel(writer, sheet_name="Selected_Pairs", index=False)
        all_pairs_df.to_excel(writer, sheet_name="All_ON_PN_Combinations", index=False)
        ticker_metrics_df.to_excel(writer, sheet_name="Ticker_Level_Metrics", index=False)

    print("Done.")
    print(f"Created: {OUTPUT_FILE}")

    if not selected_pairs_df.empty:
        print("\nTop 15 most liquid ON/PN pairs:")
        columns_to_print = [
            "ticker_prefix",
            "company_names",
            "on_ticker",
            "pn_ticker",
            "common_trading_days",
            "on_avg_daily_value_common",
            "pn_avg_daily_value_common",
            "pair_avg_daily_value",
            "passes_base_liquidity",
            "passes_strict_liquidity",
        ]

        print(selected_pairs_df[columns_to_print].head(15))


if __name__ == "__main__":
    main()
