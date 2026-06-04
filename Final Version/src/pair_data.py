import pandas as pd


class PairData:
    """
    Prepares the base dataset for one ON/PN company pair.

    This class receives already-loaded price data and transforms it into
    an analysis-ready dataset.

    The final dataset contains:
    - ON price
    - PN price
    - Ibovespa price
    - ON daily return
    - PN daily return
    - Ibovespa daily return

    This class does not generate trading signals.
    This class does not calculate taxes.
    This class does not run backtests.
    """

    def __init__(
        self,
        company: str,
        on_ticker: str,
        pn_ticker: str,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None,
    ):
        """
        Initializes the pair data object.

        Parameters
        ----------
        company:
            Company identifier, for example "PETR", "ITUB" or "BBDC".

        on_ticker:
            ON ticker, for example "PETR3.SA".

        pn_ticker:
            PN ticker, for example "PETR4.SA".

        price_data:
            DataFrame containing ON, PN and IBOVESPA price columns.

        volume_data:
            Optional DataFrame containing ON and PN volume columns.
        """

        self.company = company.upper()
        self.name = self.company

        self.on_ticker = on_ticker
        self.pn_ticker = pn_ticker

        self.raw_prices = price_data.copy()
        self.raw_volumes = volume_data.copy() if volume_data is not None else None

        self.data = self._prepare_price_data(self.raw_prices)
        self.prices = self.data[["ON", "PN"]].copy()
        self.volumes = self._prepare_volume_data(self.raw_volumes)

    def _prepare_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans price data and calculates daily returns.
        """

        required_columns = ["ON", "PN", "IBOVESPA"]

        for column in required_columns:
            if column not in price_data.columns:
                raise ValueError(
                    f"Missing column '{column}' in price data for {self.company}"
                )

        data = price_data.copy()
        data = data.sort_index()

        # Convert all price columns to numeric values.
        for column in required_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        # Remove rows with missing or invalid prices.
        data = data.dropna(subset=required_columns)

        data = data[
            (data["ON"] > 0)
            & (data["PN"] > 0)
            & (data["IBOVESPA"] > 0)
        ]

        if data.empty:
            raise ValueError(f"No valid price data available for {self.company}")

        # Calculate daily returns.
        data["return_on"] = data["ON"].pct_change()
        data["return_pn"] = data["PN"].pct_change()
        data["return_ibovespa"] = data["IBOVESPA"].pct_change()

        # The first row has missing returns because there is no previous price.
        data = data.dropna(
            subset=[
                "return_on",
                "return_pn",
                "return_ibovespa",
            ]
        )

        if data.empty:
            raise ValueError(
                f"No valid return data available for {self.company}"
            )

        return data

    def _prepare_volume_data(
        self,
        volume_data: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """
        Cleans optional ON/PN volume data.

        Volume is useful for the universe filter, but the project can still run
        without requiring volume data.
        """

        if volume_data is None:
            return None

        required_columns = ["ON", "PN"]

        for column in required_columns:
            if column not in volume_data.columns:
                return None

        volumes = volume_data.copy()
        volumes = volumes.sort_index()

        for column in required_columns:
            volumes[column] = pd.to_numeric(volumes[column], errors="coerce")

        volumes = volumes.dropna(subset=required_columns)

        if volumes.empty:
            return None

        return volumes

    def get_data(self) -> pd.DataFrame:
        """
        Returns the full analysis-ready dataset.
        """

        return self.data.copy()

    def get_train_test_ready_data(self) -> pd.DataFrame:
        """
        Returns the dataset used by the train-test splitter.

        This method exists mainly for readability in main.py.
        """

        return self.get_data()

    def get_price_data(self) -> pd.DataFrame:
        """
        Returns only ON and PN prices.

        This is mainly used by the universe filter.
        """

        return self.prices.copy()

    def get_volume_data(self) -> pd.DataFrame | None:
        """
        Returns ON and PN volume data if available.
        """

        if self.volumes is None:
            return None

        return self.volumes.copy()

    def summary(self) -> dict:
        """
        Returns a simple summary of the prepared pair data.
        """

        return {
            "company": self.company,
            "on_ticker": self.on_ticker,
            "pn_ticker": self.pn_ticker,
            "start_date": self.data.index.min(),
            "end_date": self.data.index.max(),
            "observations": len(self.data),
            "has_volume_data": self.volumes is not None,
        }