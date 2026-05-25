from pathlib import Path
import time

import pandas as pd
import yfinance as yf


# ============================================================
# 1. Project paths and parameters
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "on_pn_companies_filtered.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "on_pn_historical_liquidity_analysis_2010_2025.xlsx"


START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# These are NOT final filters.
# They are only used to create suggested flags and comments.
SUGGESTED_MIN_COMMON_TRADING_DAYS = 1000
SUGGESTED_BASE_AVG_LIQUIDITY = 1_000_000
SUGGESTED_BASE_MEDIAN_LIQUIDITY = 500_000
SUGGESTED_STRICT_AVG_LIQUIDITY = 5_000_000
SUGGESTED_STRICT_MEDIAN_LIQUIDITY = 1_000_000

SLEEP_SECONDS = 0.2

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


def safe_divide(numerator, denominator):
    """
    Avoids division-by-zero errors.
    """
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return None

    return numerator / denominator


def format_brl(value):
    """
    Formats large BRL values for terminal printouts.
    """
    if value is None or pd.isna(value):
        return "NA"

    return f"R$ {value:,.0f}"


# ============================================================
# 3. Data download
# ============================================================

def download_price_volume(ticker):
    """
    Downloads daily close price and volume from Yahoo Finance using yfinance.

    Important:
    - yfinance is a practical data source, not official B3 data.
    - Some tickers may have missing or incomplete histories.
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

        # yfinance sometimes returns MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_columns = ["Close", "Volume"]

        for col in required_columns:
            if col not in data.columns:
                print(f"Warning: {col} not found for {ticker}")
                return pd.DataFrame()

        data = data[["Close", "Volume"]].copy()
        data = data.dropna(subset=["Close", "Volume"])

        data["ticker"] = ticker
        data["yf_ticker"] = yf_ticker
        data["daily_traded_value"] = data["Close"] * data["Volume"]

        return data

    except Exception as error:
        print(f"Error downloading {ticker}: {error}")
        return pd.DataFrame()


# ============================================================
# 4. Liquidity calculation
# ============================================================

def empty_pair_result(on_ticker, pn_ticker, status):
    """
    Standard output when a pair cannot be calculated.
    """
    return {
        "on_ticker": on_ticker,
        "pn_ticker": pn_ticker,

        "common_start_date": None,
        "common_end_date": None,
        "common_trading_days": 0,

        "on_avg_daily_value": None,
        "pn_avg_daily_value": None,
        "on_median_daily_value": None,
        "pn_median_daily_value": None,
        "on_max_daily_value": None,
        "pn_max_daily_value": None,

        "on_avg_volume": None,
        "pn_avg_volume": None,
        "on_median_volume": None,
        "pn_median_volume": None,

        "on_zero_volume_days": None,
        "pn_zero_volume_days": None,
        "on_zero_volume_pct": None,
        "pn_zero_volume_pct": None,

        "pair_avg_daily_value": None,
        "pair_median_daily_value": None,
        "pair_max_daily_value": None,

        "pair_liquidity_ratio_pn_to_on": None,
        "less_liquid_leg_by_avg": None,
        "less_liquid_leg_by_median": None,

        "data_status": status,
    }


def calculate_pair_liquidity(on_ticker, pn_ticker, price_cache):
    """
    Calculates historical liquidity metrics for one ON/PN pair.

    Pair-level liquidity is defined conservatively:
    - pair_avg_daily_value = lower average daily traded value between ON and PN
    - pair_median_daily_value = lower median daily traded value between ON and PN
    """
    if on_ticker not in price_cache:
        price_cache[on_ticker] = download_price_volume(on_ticker)
        time.sleep(SLEEP_SECONDS)

    if pn_ticker not in price_cache:
        price_cache[pn_ticker] = download_price_volume(pn_ticker)
        time.sleep(SLEEP_SECONDS)

    on_data = price_cache[on_ticker]
    pn_data = price_cache[pn_ticker]

    if on_data.empty or pn_data.empty:
        return empty_pair_result(
            on_ticker=on_ticker,
            pn_ticker=pn_ticker,
            status="Missing ON or PN data",
        )

    common_dates = on_data.index.intersection(pn_data.index)

    if len(common_dates) == 0:
        return empty_pair_result(
            on_ticker=on_ticker,
            pn_ticker=pn_ticker,
            status="No overlapping dates",
        )

    on_common = on_data.loc[common_dates].copy()
    pn_common = pn_data.loc[common_dates].copy()

    on_avg_value = on_common["daily_traded_value"].mean()
    pn_avg_value = pn_common["daily_traded_value"].mean()

    on_median_value = on_common["daily_traded_value"].median()
    pn_median_value = pn_common["daily_traded_value"].median()

    on_max_value = on_common["daily_traded_value"].max()
    pn_max_value = pn_common["daily_traded_value"].max()

    on_avg_volume = on_common["Volume"].mean()
    pn_avg_volume = pn_common["Volume"].mean()

    on_median_volume = on_common["Volume"].median()
    pn_median_volume = pn_common["Volume"].median()

    on_zero_volume_days = int((on_common["Volume"] == 0).sum())
    pn_zero_volume_days = int((pn_common["Volume"] == 0).sum())

    on_zero_volume_pct = on_zero_volume_days / len(common_dates)
    pn_zero_volume_pct = pn_zero_volume_days / len(common_dates)

    pair_avg_value = min(on_avg_value, pn_avg_value)
    pair_median_value = min(on_median_value, pn_median_value)
    pair_max_value = min(on_max_value, pn_max_value)

    less_liquid_leg_by_avg = "ON" if on_avg_value <= pn_avg_value else "PN"
    less_liquid_leg_by_median = "ON" if on_median_value <= pn_median_value else "PN"

    liquidity_ratio = safe_divide(pn_avg_value, on_avg_value)

    return {
        "on_ticker": on_ticker,
        "pn_ticker": pn_ticker,

        "common_start_date": common_dates.min().date(),
        "common_end_date": common_dates.max().date(),
        "common_trading_days": len(common_dates),

        "on_avg_daily_value": on_avg_value,
        "pn_avg_daily_value": pn_avg_value,
        "on_median_daily_value": on_median_value,
        "pn_median_daily_value": pn_median_value,
        "on_max_daily_value": on_max_value,
        "pn_max_daily_value": pn_max_value,

        "on_avg_volume": on_avg_volume,
        "pn_avg_volume": pn_avg_volume,
        "on_median_volume": on_median_volume,
        "pn_median_volume": pn_median_volume,

        "on_zero_volume_days": on_zero_volume_days,
        "pn_zero_volume_days": pn_zero_volume_days,
        "on_zero_volume_pct": on_zero_volume_pct,
        "pn_zero_volume_pct": pn_zero_volume_pct,

        "pair_avg_daily_value": pair_avg_value,
        "pair_median_daily_value": pair_median_value,
        "pair_max_daily_value": pair_max_value,

        "pair_liquidity_ratio_pn_to_on": liquidity_ratio,
        "less_liquid_leg_by_avg": less_liquid_leg_by_avg,
        "less_liquid_leg_by_median": less_liquid_leg_by_median,

        "data_status": "OK",
    }


# ============================================================
# 5. Descriptive analysis columns
# ============================================================

def add_descriptive_liquidity_metrics(df):
    """
    Adds descriptive liquidity categories and suggested flags.

    Important:
    These suggested flags are NOT final inclusion/exclusion decisions.
    They are only analytical indicators.
    """
    if df.empty:
        return df

    df = df.copy()

    # Recalculate pair liquidity defensively
    if {"on_avg_daily_value", "pn_avg_daily_value"}.issubset(df.columns):
        df["pair_avg_daily_value"] = df[
            ["on_avg_daily_value", "pn_avg_daily_value"]
        ].min(axis=1)

    if {"on_median_daily_value", "pn_median_daily_value"}.issubset(df.columns):
        df["pair_median_daily_value"] = df[
            ["on_median_daily_value", "pn_median_daily_value"]
        ].min(axis=1)

    df["data_availability_bucket"] = pd.cut(
        df["common_trading_days"],
        bins=[-1, 250, 1000, 2500, 10000],
        labels=[
            "Very short history",
            "Short history",
            "Medium history",
            "Long history",
        ],
    )

    df["avg_liquidity_bucket"] = pd.cut(
        df["pair_avg_daily_value"],
        bins=[-1, 100_000, 500_000, 1_000_000, 5_000_000, 50_000_000, float("inf")],
        labels=[
            "< R$100k",
            "R$100k–500k",
            "R$500k–1m",
            "R$1m–5m",
            "R$5m–50m",
            "> R$50m",
        ],
    )

    df["median_liquidity_bucket"] = pd.cut(
        df["pair_median_daily_value"],
        bins=[-1, 100_000, 500_000, 1_000_000, 5_000_000, 50_000_000, float("inf")],
        labels=[
            "< R$100k",
            "R$100k–500k",
            "R$500k–1m",
            "R$1m–5m",
            "R$5m–50m",
            "> R$50m",
        ],
    )

    # Suggested flags, not final filters
    df["suggested_base_liquidity_flag"] = (
            (df["common_trading_days"] >= SUGGESTED_MIN_COMMON_TRADING_DAYS)
            & (df["pair_avg_daily_value"] >= SUGGESTED_BASE_AVG_LIQUIDITY)
            & (df["pair_median_daily_value"] >= SUGGESTED_BASE_MEDIAN_LIQUIDITY)
    )

    df["suggested_strict_liquidity_flag"] = (
            (df["common_trading_days"] >= SUGGESTED_MIN_COMMON_TRADING_DAYS)
            & (df["pair_avg_daily_value"] >= SUGGESTED_STRICT_AVG_LIQUIDITY)
            & (df["pair_median_daily_value"] >= SUGGESTED_STRICT_MEDIAN_LIQUIDITY)
    )

    df["liquidity_comment"] = ""

    df.loc[
        df["data_status"] != "OK",
        "liquidity_comment",
    ] += "Data issue. "

    df.loc[
        df["common_trading_days"] < SUGGESTED_MIN_COMMON_TRADING_DAYS,
        "liquidity_comment",
    ] += "Limited common history. "

    df.loc[
        df["pair_avg_daily_value"] < SUGGESTED_BASE_AVG_LIQUIDITY,
        "liquidity_comment",
    ] += "Low average pair liquidity. "

    df.loc[
        df["pair_median_daily_value"] < SUGGESTED_BASE_MEDIAN_LIQUIDITY,
        "liquidity_comment",
    ] += "Low median pair liquidity. "

    df.loc[
        df["liquidity_comment"] == "",
        "liquidity_comment",
    ] = "No major liquidity concern under suggested thresholds."

    return df


# ============================================================
# 6. Main script
# ============================================================

def main():
    print("Reading ON/PN companies file...")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

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
    price_cache = {}

    for idx, row in companies.iterrows():
        prefix = row["ticker_prefix"]
        company_names = row["company_names"]

        on_list = parse_ticker_list(row["on_tickers"])
        pn_list = parse_ticker_list(row["pn_tickers"])

        print(f"\nProcessing {idx + 1}/{len(companies)}: {prefix}")
        print(f"ON tickers: {on_list}")
        print(f"PN tickers: {pn_list}")

        company_pair_results = []

        for on_ticker in on_list:
            for pn_ticker in pn_list:
                metrics = calculate_pair_liquidity(
                    on_ticker=on_ticker,
                    pn_ticker=pn_ticker,
                    price_cache=price_cache,
                )

                metrics["ticker_prefix"] = prefix
                metrics["company_names"] = company_names
                metrics["all_on_tickers"] = ", ".join(on_list)
                metrics["all_pn_tickers"] = ", ".join(pn_list)

                company_pair_results.append(metrics)
                all_pair_results.append(metrics)

        if not company_pair_results:
            continue

        company_pair_df = pd.DataFrame(company_pair_results)

        valid_pairs = company_pair_df[
            company_pair_df["pair_avg_daily_value"].notna()
        ].copy()

        if valid_pairs.empty:
            selected_pair = company_pair_df.iloc[0].to_dict()
            selected_pair["selected_reason"] = "No valid liquidity data"
        else:
            selected_pair = (
                valid_pairs
                .sort_values(
                    ["pair_avg_daily_value", "pair_median_daily_value"],
                    ascending=False,
                )
                .iloc[0]
                .to_dict()
            )
            selected_pair["selected_reason"] = (
                "Highest pair-level average daily traded value among available PN classes"
            )

        selected_pair_results.append(selected_pair)

    all_pairs_df = pd.DataFrame(all_pair_results)
    selected_pairs_df = pd.DataFrame(selected_pair_results)

    all_pairs_df = add_descriptive_liquidity_metrics(all_pairs_df)
    selected_pairs_df = add_descriptive_liquidity_metrics(selected_pairs_df)

    if not selected_pairs_df.empty:
        selected_pairs_df = selected_pairs_df.sort_values(
            ["pair_avg_daily_value", "pair_median_daily_value"],
            ascending=False,
        )

    methodology = pd.DataFrame({
        "Item": [
            "Study period",
            "Data source",
            "Liquidity measure",
            "Pair-level average liquidity",
            "Pair-level median liquidity",
            "Multiple PN classes",
            "Suggested base flag",
            "Suggested strict flag",
            "Important note",
        ],
        "Description": [
            f"{START_DATE} to {END_DATE}",
            "Historical price and volume data are downloaded using yfinance/Yahoo Finance.",
            "Daily traded value = Close price × Volume.",
            "Lower average daily traded value between ON and PN.",
            "Lower median daily traded value between ON and PN.",
            "If several PN classes exist, the pair with the highest pair-level average daily traded value is selected.",
            (
                f"Common trading days >= {SUGGESTED_MIN_COMMON_TRADING_DAYS}, "
                f"average pair liquidity >= R$ {SUGGESTED_BASE_AVG_LIQUIDITY:,.0f}, "
                f"median pair liquidity >= R$ {SUGGESTED_BASE_MEDIAN_LIQUIDITY:,.0f}."
            ),
            (
                f"Common trading days >= {SUGGESTED_MIN_COMMON_TRADING_DAYS}, "
                f"average pair liquidity >= R$ {SUGGESTED_STRICT_AVG_LIQUIDITY:,.0f}, "
                f"median pair liquidity >= R$ {SUGGESTED_STRICT_MEDIAN_LIQUIDITY:,.0f}."
            ),
            (
                "The suggested flags are not final exclusion rules. "
                "They support later methodological decisions and manual review."
            ),
        ],
    })

    print("\nWriting Excel output...")
    print(f"Output file: {OUTPUT_FILE}")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        methodology.to_excel(writer, sheet_name="Methodology", index=False)
        selected_pairs_df.to_excel(writer, sheet_name="Selected_Pairs_Ranked", index=False)
        all_pairs_df.to_excel(writer, sheet_name="All_ON_PN_Combinations", index=False)

    print("Done.")
    print(f"Excel file created: {OUTPUT_FILE}")

    if not selected_pairs_df.empty:
        print("\nTop 20 most liquid pairs:")
        columns = [
            "ticker_prefix",
            "company_names",
            "on_ticker",
            "pn_ticker",
            "common_trading_days",
            "pair_avg_daily_value",
            "pair_median_daily_value",
            "avg_liquidity_bucket",
            "median_liquidity_bucket",
            "data_availability_bucket",
            "less_liquid_leg_by_avg",
            "suggested_base_liquidity_flag",
            "suggested_strict_liquidity_flag",
            "liquidity_comment",
        ]

        print(selected_pairs_df[columns].head(20).to_string(index=False))

        print("\nSuggested base liquidity candidates:")
        base_candidates = selected_pairs_df[
            selected_pairs_df["suggested_base_liquidity_flag"] == True
            ]

        if base_candidates.empty:
            print("No pairs meet the suggested base liquidity criteria.")
        else:
            print(
                base_candidates[
                    [
                        "ticker_prefix",
                        "company_names",
                        "on_ticker",
                        "pn_ticker",
                        "pair_avg_daily_value",
                        "pair_median_daily_value",
                    ]
                ].to_string(index=False)
            )




if __name__ == "__main__":
    main()
