import pandas as pd


class TimeSeriesSplitter:
    """
    Splits financial time series into chronological train and test samples.

    Financial data must never be randomly shuffled because future information
    cannot be used to calibrate past decisions.

    Therefore:
    - the training sample contains the oldest observations;
    - the test sample contains the most recent observations.
    """

    def __init__(self, train_ratio: float = 2 / 3):
        """
        Initializes the splitter.

        Parameters
        ----------
        train_ratio:
            Fraction of observations assigned to the training sample.
            The remaining observations are assigned to the test sample.
        """

        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        self.train_ratio = float(train_ratio)

    def split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits one time series into train and test samples.

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

        split_index = int(len(data) * self.train_ratio)

        if split_index <= 0:
            raise ValueError("Training sample would be empty.")

        if split_index >= len(data):
            raise ValueError("Test sample would be empty.")

        train_data = data.iloc[:split_index].copy()
        test_data = data.iloc[split_index:].copy()

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