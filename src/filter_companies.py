from pathlib import Path
import pandas as pd
import re


# ============================================================
# 1. Project paths
# ============================================================


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "acoes-listadas-b3.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "on_pn_companies_filtered.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Helper functions
# ============================================================

def clean_ticker(ticker):
    """
    Cleans ticker strings.

    Example:
    ' petr4 ' -> 'PETR4'
    """
    if pd.isna(ticker):
        return None

    return str(ticker).strip().upper()


def get_ticker_suffix(ticker):
    """
    Extracts the numeric ending of a Brazilian ticker.

    Examples:
    PETR3  -> 3
    PETR4  -> 4
    KLBN11 -> 11
    AAPL34 -> 34
    """
    if ticker is None:
        return None

    match = re.search(r"(\d+)$", ticker)

    if match:
        return match.group(1)

    return None


def get_ticker_prefix(ticker):
    """
    Extracts the first four letters of the ticker.

    This is used as a simple proxy for the company/issuer group.

    Examples:
    PETR3 -> PETR
    PETR4 -> PETR
    ITUB3 -> ITUB
    ITUB4 -> ITUB
    """
    if ticker is None:
        return None

    match = re.match(r"([A-Z]{4})", ticker)

    if match:
        return match.group(1)

    return None


def classify_share_class(ticker):
    """
    Classifies Brazilian share classes using the final ticker number.

    B3 convention used here:
    3 = ON
    4 = PN
    5 = PNA
    6 = PNB
    7 = PNC
    8 = PND

    Other endings, such as 11 or 34, are excluded from the pure ON/PN universe.
    """
    suffix = get_ticker_suffix(ticker)

    mapping = {
        "3": "ON",
        "4": "PN",
        "5": "PNA",
        "6": "PNB",
        "7": "PNC",
        "8": "PND",
    }

    return mapping.get(suffix, "OTHER")


def safe_join(values):
    """
    Joins unique values into a readable string.
    """
    cleaned = []

    for value in values:
        if pd.notna(value):
            cleaned.append(str(value))

    cleaned = sorted(set(cleaned))

    return ", ".join(cleaned)


# ============================================================
# 3. Main script
# ============================================================

def main():
    print("Reading raw CSV...")
    print(f"Input file: {RAW_FILE}")



    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {RAW_FILE}\n"
            "Check if the file is inside data/raw/ and has the correct name."
        )

    # The CSV from Dados de Mercado usually uses comma separator.
    # If your file opens incorrectly, try sep=';' instead.
    df = pd.read_csv(RAW_FILE)



    print(f"Raw rows read: {len(df)}")
    print(f"Columns found: {list(df.columns)}")

    # ========================================================
    # 4. Standardize column names
    # ========================================================

    # Your CSV should have a column called "Ticker".
    # If the column name is different, adjust here.
    if "Ticker" not in df.columns:
        raise ValueError(
            "Column 'Ticker' was not found in the CSV. "
            f"Available columns are: {list(df.columns)}"
        )

    if "Nome" not in df.columns:
        print("Warning: Column 'Nome' not found. Company name will be missing.")

    df["ticker"] = df["Ticker"].apply(clean_ticker)
    df["name"] = df["Nome"] if "Nome" in df.columns else None

    # ========================================================
    # 5. Classify ticker
    # ========================================================

    df["ticker_suffix"] = df["ticker"].apply(get_ticker_suffix)
    df["ticker_prefix"] = df["ticker"].apply(get_ticker_prefix)
    df["share_class"] = df["ticker"].apply(classify_share_class)

    # ========================================================
    # 6. Step 1 — Keep only Brazilian listed stock classes
    # ========================================================

    valid_share_classes = ["ON", "PN", "PNA", "PNB", "PNC", "PND"]

    stock_classes = df[df["share_class"].isin(valid_share_classes)].copy()

    print(f"Stock-class rows kept: {len(stock_classes)}")

    # ========================================================
    # 7. Step 2 — Identify companies with ON and PN
    # ========================================================

    grouped = (
        stock_classes
        .groupby("ticker_prefix")
        .agg(
            company_names=("name", safe_join),
            tickers=("ticker", lambda x: ", ".join(sorted(set(x)))),
            share_classes=("share_class", lambda x: ", ".join(sorted(set(x)))),
            on_tickers=("ticker", lambda x: ", ".join(sorted(
                set(stock_classes.loc[x.index][stock_classes.loc[x.index, "share_class"] == "ON"]["ticker"])
            ))),
            pn_tickers=("ticker", lambda x: ", ".join(sorted(
                set(stock_classes.loc[x.index][stock_classes.loc[x.index, "share_class"].isin(["PN", "PNA", "PNB", "PNC", "PND"])]["ticker"])
            ))),
        )
        .reset_index()
    )

    grouped["has_on"] = grouped["on_tickers"].apply(lambda x: len(str(x).strip()) > 0)
    grouped["has_pn"] = grouped["pn_tickers"].apply(lambda x: len(str(x).strip()) > 0)

    on_pn_companies = grouped[
        (grouped["has_on"] == True) &
        (grouped["has_pn"] == True)
        ].copy()

    on_pn_companies = on_pn_companies.sort_values("ticker_prefix")

    print(f"ON/PN company groups found: {len(on_pn_companies)}")

    # ========================================================
    # 8. Add methodology text
    # ========================================================

    methodology = pd.DataFrame({
        "Step": [
            "Step 1",
            "Step 2",
            "Ticker classification",
            "Important limitation"
        ],
        "Description": [
            "Start with Brazilian listed stocks traded on B3. Keep only individual share classes with ticker endings 3, 4, 5, 6, 7, or 8.",
            "Group tickers by the first four letters and keep only companies with at least one ON share and one PN-type share.",
            "3 = ON, 4 = PN, 5 = PNA, 6 = PNB, 7 = PNC, 8 = PND.",
            "This file only identifies ON/PN eligibility. It does not yet check liquidity, fundamentals, or statistical pair validity."
        ]
    })

    # ========================================================
    # 9. Export to Excel
    # ========================================================

    print(f"Writing Excel output to: {OUTPUT_FILE}")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        methodology.to_excel(writer, sheet_name="Methodology", index=False)
        on_pn_companies.to_excel(writer, sheet_name="ON_PN_Companies", index=False)
        stock_classes.to_excel(writer, sheet_name="Filtered_Stock_Classes", index=False)

    print("Done.")
    print(f"Excel file created: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
