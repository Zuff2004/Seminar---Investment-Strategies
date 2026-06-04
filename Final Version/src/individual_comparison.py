import pandas as pd

from performance_metrics import PerformanceMetrics


class IndividualComparisonBuilder:
    """
    Builds the final individual comparison for each company.

    This class combines:
    - the active ON/PN rotation strategy;
    - the passive 50/50 ON/PN buy-and-hold benchmark;
    - the Ibovespa buy-and-hold benchmark.

    It also calculates the final metrics used in the project report.
    """

    def __init__(
        self,
        trading_days_per_year: int = 252,
    ):
        """
        Initializes the comparison builder.

        Parameters
        ----------
        trading_days_per_year:
            Number of trading days used for annualized metrics.
        """

        self.metrics_calculator = PerformanceMetrics(
            trading_days_per_year=trading_days_per_year
        )

    # ============================================================
    # Main comparison method
    # ============================================================

    def build_comparison(
        self,
        company: str,
        strategy_result: pd.DataFrame,
        benchmark_result: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Merges strategy and benchmark results for one company.

        Parameters
        ----------
        company:
            Company identifier, for example "PETR".

        strategy_result:
            Output from ShareClassRotationBacktester.

        benchmark_result:
            Output from BenchmarkBuilder.build_all_benchmarks.

        Returns
        -------
        tuple[pandas.DataFrame, dict]
            Full daily comparison DataFrame and final metrics dictionary.
        """

        if strategy_result is None or strategy_result.empty:
            raise ValueError(f"Strategy result is empty for {company}")

        if benchmark_result is None or benchmark_result.empty:
            raise ValueError(f"Benchmark result is empty for {company}")

        comparison = pd.concat(
            [
                strategy_result,
                benchmark_result,
            ],
            axis=1,
            join="inner",
        )

        comparison = comparison.sort_index()

        if comparison.empty:
            raise ValueError(
                f"No overlapping dates between strategy and benchmarks for {company}"
            )

        comparison["company"] = company

        # ------------------------------------------------------------
        # Relative performance columns
        # ------------------------------------------------------------

        comparison["strategy_minus_50_50"] = (
            comparison["strategy_cumulative_return"]
            - comparison["benchmark_50_50_cumulative_return"]
        )

        comparison["strategy_minus_ibovespa"] = (
            comparison["strategy_cumulative_return"]
            - comparison["ibovespa_cumulative_return"]
        )

        comparison["benchmark_50_50_minus_ibovespa"] = (
            comparison["benchmark_50_50_cumulative_return"]
            - comparison["ibovespa_cumulative_return"]
        )

        metrics = self.calculate_metrics(
            company=company,
            comparison=comparison,
        )

        return comparison, metrics

    # ============================================================
    # Metrics
    # ============================================================

    def calculate_metrics(
        self,
        company: str,
        comparison: pd.DataFrame,
    ) -> dict:
        """
        Calculates strategy, 50/50 and Ibovespa metrics for one company.
        """

        required_columns = [
            "strategy_value",
            "benchmark_50_50_value",
            "ibovespa_value",
            "strategy_cumulative_return",
            "benchmark_50_50_cumulative_return",
            "ibovespa_cumulative_return",
        ]

        for column in required_columns:
            if column not in comparison.columns:
                raise ValueError(f"Missing column for metrics: {column}")

        strategy_metrics = self.metrics_calculator.calculate_from_equity_curve(
            equity_curve=comparison["strategy_value"],
            label="strategy",
        )

        benchmark_50_50_metrics = self.metrics_calculator.calculate_from_equity_curve(
            equity_curve=comparison["benchmark_50_50_value"],
            label="benchmark_50_50",
        )

        ibovespa_metrics = self.metrics_calculator.calculate_from_equity_curve(
            equity_curve=comparison["ibovespa_value"],
            label="ibovespa",
        )

        final_row = comparison.iloc[-1]

        metrics = {
            "company": company,

            **strategy_metrics,
            **benchmark_50_50_metrics,
            **ibovespa_metrics,

            "strategy_final_value": final_row["strategy_value"],
            "benchmark_50_50_final_value": final_row["benchmark_50_50_value"],
            "ibovespa_final_value": final_row["ibovespa_value"],

            "strategy_final_cumulative_return": final_row[
                "strategy_cumulative_return"
            ],
            "benchmark_50_50_final_cumulative_return": final_row[
                "benchmark_50_50_cumulative_return"
            ],
            "ibovespa_final_cumulative_return": final_row[
                "ibovespa_cumulative_return"
            ],

            "strategy_excess_return_vs_50_50": (
                final_row["strategy_cumulative_return"]
                - final_row["benchmark_50_50_cumulative_return"]
            ),

            "strategy_excess_return_vs_ibovespa": (
                final_row["strategy_cumulative_return"]
                - final_row["ibovespa_cumulative_return"]
            ),

            "benchmark_50_50_excess_return_vs_ibovespa": (
                final_row["benchmark_50_50_cumulative_return"]
                - final_row["ibovespa_cumulative_return"]
            ),

            "total_tax_paid": self._safe_sum(
                comparison=comparison,
                column="tax_paid",
            ),

            "total_transaction_cost": self._safe_sum(
                comparison=comparison,
                column="transaction_cost",
            ),

            "total_realized_pnl": self._safe_sum(
                comparison=comparison,
                column="realized_pnl",
            ),

            "gross_sale_value": self._safe_sum(
                comparison=comparison,
                column="gross_sale_value",
            ),

            "gross_buy_value": self._safe_sum(
                comparison=comparison,
                column="gross_buy_value",
            ),

            "number_of_trading_days": len(comparison),

            "number_of_trade_days": self._safe_trade_count(
                comparison=comparison,
            ),

            "final_weight_on": self._safe_final_value(
                final_row=final_row,
                column="weight_on",
            ),

            "final_weight_pn": self._safe_final_value(
                final_row=final_row,
                column="weight_pn",
            ),

            "final_target_weight_on": self._safe_final_value(
                final_row=final_row,
                column="target_weight_on",
            ),

            "final_target_weight_pn": self._safe_final_value(
                final_row=final_row,
                column="target_weight_pn",
            ),

            "policy_group": self._safe_final_value(
                final_row=final_row,
                column="policy_group",
            ),

            "last_signal": self._safe_final_value(
                final_row=final_row,
                column="signal",
            ),

            "final_accumulated_loss": self._safe_final_value(
                final_row=final_row,
                column="accumulated_loss",
            ),
        }

        return metrics

    # ============================================================
    # Save helpers
    # ============================================================

    def save_individual_comparison(
        self,
        comparison: pd.DataFrame,
        output_path,
    ):
        """
        Saves one company's daily comparison table as CSV.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)

        comparison_to_save = comparison.copy()
        comparison_to_save.index.name = "date"

        comparison_to_save.to_csv(output_path)

    def build_metrics_table(
        self,
        metrics_by_company: list[dict],
    ) -> pd.DataFrame:
        """
        Builds the final company-level metrics table.
        """

        if not metrics_by_company:
            return pd.DataFrame()

        metrics_table = pd.DataFrame(metrics_by_company)

        if "strategy_excess_return_vs_50_50" in metrics_table.columns:
            metrics_table = metrics_table.sort_values(
                by="strategy_excess_return_vs_50_50",
                ascending=False,
            )

        return metrics_table

    # ============================================================
    # Internal helpers
    # ============================================================

    @staticmethod
    def _safe_sum(
        comparison: pd.DataFrame,
        column: str,
    ) -> float:
        """
        Safely sums a column if it exists.
        """

        if column not in comparison.columns:
            return 0.0

        return float(
            comparison[column]
            .fillna(0.0)
            .sum()
        )

    @staticmethod
    def _safe_trade_count(
        comparison: pd.DataFrame,
    ) -> int:
        """
        Counts the number of days with trades.
        """

        if "traded" not in comparison.columns:
            return 0

        return int(
            comparison["traded"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    @staticmethod
    def _safe_final_value(
        final_row: pd.Series,
        column: str,
    ):
        """
        Safely returns a value from the final row.
        """

        if column not in final_row.index:
            return None

        return final_row[column]