from pathlib import Path

import pandas as pd
import yfinance as yf


class MarketDataLoader:
    """
    Loads and optionally downloads market data.

    This class is responsible only for data access:
    - downloading price data from Yahoo Finance;
    - saving raw CSV files;
    - loading local CSV files from data/raw;
    - returning clean price series for the rest of the pipeline.

    It does not calculate signals, taxes, backtest results or portfolio metrics.
    """

    def __init__(
        self,
        raw_data_dir: str | Path,
        download: bool = False,
    ):
        """
        Initializes the market data loader.

        Parameters
        ----------
        raw_data_dir:
            Folder where raw CSV files are stored.

        download:
            If True, the loader downloads data from Yahoo Finance.
            If False, the loader only reads existing local CSV files.
        """

        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self.download = bool(download)

    # ============================================================
    # Download methods
    # ============================================================

    def download_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Downloads one ticker from Yahoo Finance and saves it as CSV.

        Example:
            PETR3.SA -> data/raw/PETR3.csv
            ^BVSP    -> data/raw/BVSP.csv
        """

        print(f"Downloading {ticker}...")

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        if data is None or data.empty:
            raise ValueError(f"No data downloaded for ticker {ticker}")

        # yfinance may return MultiIndex columns in some versions.
        # This keeps only the price-field level.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.dropna(how="all")

        output_path = self.raw_data_dir / self._ticker_to_filename(ticker)
        data.to_csv(output_path)

        print(f"Saved {ticker} to {output_path}")

        return data

    def download_multiple_tickers(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Downloads multiple tickers from Yahoo Finance.

        Failed downloads are reported but do not stop the full pipeline.
        """

        downloaded_data = {}

        for ticker in tickers:
            try:
                downloaded_data[ticker] = self.download_single_ticker(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as error:
                print(f"Could not download {ticker}: {error}")

        return downloaded_data

    def download_project_universe(
        self,
        company_pairs: dict,
        ibovespa_ticker: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Downloads all ON/PN tickers and the Ibovespa benchmark.
        """

        tickers = set()

        for on_ticker, pn_ticker in company_pairs.values():
            tickers.add(on_ticker)
            tickers.add(pn_ticker)

        tickers.add(ibovespa_ticker)

        return self.download_multiple_tickers(
            tickers=sorted(tickers),
            start_date=start_date,
            end_date=end_date,
        )

    # ============================================================
    # Loading methods
    # ============================================================

    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Loads the full local CSV file for one ticker.

        The method cleans common CSV formatting issues and returns a DataFrame
        indexed by date.
        """

        file_path = self.raw_data_dir / self._ticker_to_filename(ticker)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing local data for {ticker}: {file_path}. "
                "Set download_data=True in project_config.py or add the CSV manually."
            )

        data = pd.read_csv(file_path)

        if data.empty:
            raise ValueError(f"Local CSV for {ticker} is empty: {file_path}")

        # Some yfinance CSVs may contain duplicated header rows.
        # These rows are removed before parsing dates.
        first_column = data.columns[0]

        data = data[data[first_column] != "Ticker"]
        data = data[data[first_column] != "Date"]

        data = data.rename(columns={first_column: "Date"})

        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])
        data = data.set_index("Date")
        data = data.sort_index()

        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        data = data.dropna(how="all")

        return data

    def load_close_price(self, ticker: str) -> pd.Series:
        """
        Loads the adjusted close/close price series for one ticker.

        Because yfinance with auto_adjust=True stores adjusted prices in
        the Close column, the project uses Close as the price series.
        """

        data = self.load_ticker_data(ticker)

        if "Close" not in data.columns:
            raise ValueError(f"'Close' column not found for ticker {ticker}")

        close = data["Close"].copy()
        close = pd.to_numeric(close, errors="coerce")
        close = close.dropna()
        close.name = ticker

        return close

    def load_volume(self, ticker: str) -> pd.Series | None:
        """
        Loads volume data for one ticker if available.

        The universe filter can use volume, but the final project can also
        run without requiring volume data.
        """

        data = self.load_ticker_data(ticker)

        if "Volume" not in data.columns:
            return None

        volume = data["Volume"].copy()
        volume = pd.to_numeric(volume, errors="coerce")
        volume = volume.dropna()
        volume.name = ticker

        return volume

    def load_pair_prices(
        self,
        company: str,
        on_ticker: str,
        pn_ticker: str,
        ibovespa_ticker: str,
    ) -> pd.DataFrame:
        """
        Loads and merges ON, PN and Ibovespa close prices.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by Date with columns:
            - ON
            - PN
            - IBOVESPA
        """

        on_close = self.load_close_price(on_ticker).rename("ON")
        pn_close = self.load_close_price(pn_ticker).rename("PN")
        ibovespa_close = self.load_close_price(ibovespa_ticker).rename("IBOVESPA")

        data = pd.concat(
            [on_close, pn_close, ibovespa_close],
            axis=1,
            join="inner",
        )

        data = data.dropna()
        data = data.sort_index()

        if data.empty:
            raise ValueError(f"No overlapping price data for {company}")

        return data

    def load_pair_volumes(
        self,
        on_ticker: str,
        pn_ticker: str,
    ) -> pd.DataFrame | None:
        """
        Loads and merges ON and PN volume data.

        Returns None if volume data is unavailable.
        """

        on_volume = self.load_volume(on_ticker)
        pn_volume = self.load_volume(pn_ticker)

        if on_volume is None or pn_volume is None:
            return None

        volumes = pd.concat(
            [
                on_volume.rename("ON"),
                pn_volume.rename("PN"),
            ],
            axis=1,
            join="inner",
        )

        volumes = volumes.dropna()
        volumes = volumes.sort_index()

        if volumes.empty:
            return None

        return volumes

    # ============================================================
    # Filename helper
    # ============================================================

    @staticmethod
    def _ticker_to_filename(ticker: str) -> str:
        """
        Converts a market ticker into a local CSV filename.

        Examples:
            PETR3.SA -> PETR3.csv
            ^BVSP    -> BVSP.csv
        """

        filename = ticker.replace(".SA", "")
        filename = filename.replace("^", "")

        return f"{filename}.csv"