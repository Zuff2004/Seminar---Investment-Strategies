import pandas as pd


class BenchmarkBuilder:
    """
    Builds passive benchmarks for the individual company comparison.

    The project uses two benchmarks:

    1. Passive 50/50 ON/PN buy-and-hold:
       - invests 50% in ON and 50% in PN on the first day;
       - never rebalances after the initial allocation;
       - holds fixed quantities until the end.

    2. Ibovespa buy-and-hold:
       - invests the same initial capital in Ibovespa on the first day;
       - holds the index exposure until the end.
    """

    def __init__(self, initial_capital: float = 1.0):
        """
        Initializes the benchmark builder.

        Parameters
        ----------
        initial_capital:
            Initial capital used for all benchmark curves.
        """

        self.initial_capital = float(initial_capital)

    def build_50_50_buy_and_hold(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Builds the passive 50/50 ON/PN buy-and-hold benchmark.

        Important:
        The benchmark is not rebalanced daily. It buys fixed quantities of
        ON and PN on the first day and holds these quantities until the end.

        Parameters
        ----------
        data:
            DataFrame containing ON and PN price columns.

        Returns
        -------
        pandas.DataFrame
            Benchmark value, daily return and cumulative return.
        """

        required_columns = ["ON", "PN"]

        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Missing column: {column}")

        benchmark = data.copy().sort_index()

        if benchmark.empty:
            raise ValueError("Benchmark data is empty.")

        first_on_price = float(benchmark["ON"].iloc[0])
        first_pn_price = float(benchmark["PN"].iloc[0])

        if first_on_price <= 0 or first_pn_price <= 0:
            raise ValueError("Initial ON and PN prices must be positive.")

        # ------------------------------------------------------------
        # Initial allocation:
        # 50% of capital goes to ON and 50% goes to PN.
        # ------------------------------------------------------------

        initial_on_value = self.initial_capital * 0.50
        initial_pn_value = self.initial_capital * 0.50

        quantity_on = initial_on_value / first_on_price
        quantity_pn = initial_pn_value / first_pn_price

        # ------------------------------------------------------------
        # Buy-and-hold value:
        # Quantities remain fixed until the end of the sample.
        # ------------------------------------------------------------

        benchmark["benchmark_50_50_value"] = (
            quantity_on * benchmark["ON"]
            + quantity_pn * benchmark["PN"]
        )

        benchmark["benchmark_50_50_return"] = (
            benchmark["benchmark_50_50_value"]
            .pct_change()
            .fillna(0.0)
        )

        benchmark["benchmark_50_50_cumulative_return"] = (
            benchmark["benchmark_50_50_value"]
            / self.initial_capital
            - 1.0
        )

        benchmark["benchmark_50_50_weight_on"] = (
            quantity_on * benchmark["ON"]
            / benchmark["benchmark_50_50_value"]
        )

        benchmark["benchmark_50_50_weight_pn"] = (
            quantity_pn * benchmark["PN"]
            / benchmark["benchmark_50_50_value"]
        )

        return benchmark[
            [
                "benchmark_50_50_value",
                "benchmark_50_50_return",
                "benchmark_50_50_cumulative_return",
                "benchmark_50_50_weight_on",
                "benchmark_50_50_weight_pn",
            ]
        ]

    def build_ibovespa_buy_and_hold(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Builds the Ibovespa buy-and-hold benchmark.

        Parameters
        ----------
        data:
            DataFrame containing the IBOVESPA price column.

        Returns
        -------
        pandas.DataFrame
            Ibovespa benchmark value, daily return and cumulative return.
        """

        if "IBOVESPA" not in data.columns:
            raise ValueError("Missing column: IBOVESPA")

        benchmark = data.copy().sort_index()

        if benchmark.empty:
            raise ValueError("Ibovespa benchmark data is empty.")

        first_ibovespa_price = float(benchmark["IBOVESPA"].iloc[0])

        if first_ibovespa_price <= 0:
            raise ValueError("Initial Ibovespa price must be positive.")

        # ------------------------------------------------------------
        # Initial investment:
        # The full initial capital is invested in Ibovespa.
        # ------------------------------------------------------------

        quantity_ibovespa = self.initial_capital / first_ibovespa_price

        benchmark["ibovespa_value"] = (
            quantity_ibovespa * benchmark["IBOVESPA"]
        )

        benchmark["ibovespa_return"] = (
            benchmark["ibovespa_value"]
            .pct_change()
            .fillna(0.0)
        )

        benchmark["ibovespa_cumulative_return"] = (
            benchmark["ibovespa_value"]
            / self.initial_capital
            - 1.0
        )

        return benchmark[
            [
                "ibovespa_value",
                "ibovespa_return",
                "ibovespa_cumulative_return",
            ]
        ]

    def build_all_benchmarks(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Builds both individual benchmarks and merges them into one DataFrame.

        Returns:
        - passive 50/50 ON/PN benchmark;
        - Ibovespa buy-and-hold benchmark.
        """

        benchmark_50_50 = self.build_50_50_buy_and_hold(data)
        ibovespa_benchmark = self.build_ibovespa_buy_and_hold(data)

        benchmarks = pd.concat(
            [
                benchmark_50_50,
                ibovespa_benchmark,
            ],
            axis=1,
            join="inner",
        )

        benchmarks = benchmarks.sort_index()

        return benchmarks