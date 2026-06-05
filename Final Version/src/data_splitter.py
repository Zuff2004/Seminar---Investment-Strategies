import pandas as pd


class TimeSeriesSplitter:
    """
    Splits financial time series into chronological train and test samples.

    Financial data must never be randomly shuffled because future information
    cannot be used to calibrate past decisions.

    In this project, the split is based on a fixed out-of-sample start date:

    - the training sample contains all observations before test_start_date;
    - the test sample contains all observations on or after test_start_date.

    This ensures that all companies use the same out-of-sample period whenever
    data is available.
    """

    def __init__(self, test_start_date: str = "2020-01-01"):
        """
        Initializes the splitter.

        Parameters
        ----------
        test_start_date:
            First calendar date assigned to the test/backtest sample.
            The actual first test observation will be the first available
            trading day on or after this date.
        """

        self.test_start_date = pd.to_datetime(test_start_date)

    def split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits one time series into train and test samples using a fixed date.

        Parameters
        ----------
        data:
            Chronologically indexed DataFrame.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Train data and test data.
        """

        if data is None or data.empty:
            raise ValueError("DataFrame is empty.")

        data = data.copy().sort_index()

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        train_data = data[data.index < self.test_start_date].copy()
        test_data = data[data.index >= self.test_start_date].copy()

        if train_data.empty:
            raise ValueError(
                f"Training sample would be empty. "
                f"No observations before {self.test_start_date.date()}."
            )

        if test_data.empty:
            raise ValueError(
                f"Test sample would be empty. "
                f"No observations on or after {self.test_start_date.date()}."
            )

        return train_data, test_data

    def split_pair_objects(self, pair_objects: list) -> dict:
        """
        Splits multiple PairData objects into train and test datasets.

        Parameters
        ----------
        pair_objects:
            List of PairData objects.

        Returns
        -------
        dict
            Dictionary with one entry per company:
            {
                "PETR": {
                    "pair_object": PairData,
                    "train": train_data,
                    "test": test_data,
                    "full": full_data
                }
            }
        """

        split_data = {}

        for pair_object in pair_objects:
            full_data = pair_object.get_train_test_ready_data()

            train_data, test_data = self.split(full_data)

            split_data[pair_object.company] = {
                "pair_object": pair_object,
                "train": train_data,
                "test": test_data,
                "full": full_data,
            }

        return split_data

    def print_summary(
        self,
        company: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ):
        """
        Prints a readable summary of the train-test split for one company.
        """

        print(f"\n{company} train-test split")
        print("-" * 60)

        print(
            "Train:",
            train_data.index.min(),
            "->",
            train_data.index.max(),
            "| Observations:",
            len(train_data),
        )

        print(
            "Test:",
            test_data.index.min(),
            "->",
            test_data.index.max(),
            "| Observations:",
            len(test_data),
        )

    def build_split_summary(self, split_data: dict) -> pd.DataFrame:
        """
        Builds a summary table with train and test periods for all companies.
        """

        rows = []

        for company, content in split_data.items():
            train_data = content["train"]
            test_data = content["test"]

            rows.append({
                "company": company,
                "train_start": train_data.index.min(),
                "train_end": train_data.index.max(),
                "train_observations": len(train_data),
                "test_start": test_data.index.min(),
                "test_end": test_data.index.max(),
                "test_observations": len(test_data),
            })

        return pd.DataFrame(rows)