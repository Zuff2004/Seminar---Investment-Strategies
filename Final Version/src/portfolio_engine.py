import pandas as pd


class PortfolioEngine:
    """
    Builds portfolio-level results from individual company backtests.

    Important:
    This class is not the main focus of the first project stage.

    First stage:
    - compare each company individually against its own 50/50 benchmark
      and against Ibovespa.

    Second stage:
    - combine individual strategy curves into a portfolio;
    - apply portfolio weights;
    - compare the aggregate portfolio against aggregate benchmarks.
    """

    def __init__(
        self,
        weighting_method: str = "equal_weight",
    ):
        """
        Initializes the portfolio engine.

        Parameters
        ----------
        weighting_method:
            Portfolio weighting method.

            Currently supported:
            - "equal_weight"
        """

        self.weighting_method = weighting_method

        if self.weighting_method not in ["equal_weight"]:
            raise ValueError(
                "Unsupported weighting_method. Currently only 'equal_weight' is supported."
            )

    # ============================================================
    # Public methods
    # ============================================================

    def build_equal_weight_portfolio(
        self,
        individual_results: dict,
        value_column: str,
        portfolio_name: str = "portfolio",
    ) -> pd.DataFrame:
        """
        Builds an equal-weighted portfolio from individual equity curves.

        Parameters
        ----------
        individual_results:
            Dictionary mapping company code to result DataFrame.

            Example:
            {
                "PETR": petr_result,
                "ITUB": itub_result,
                "BBDC": bbdc_result,
            }

        value_column:
            Column containing the equity curve to aggregate.

            Examples:
            - "strategy_value"
            - "benchmark_50_50_value"
            - "ibovespa_value"

        portfolio_name:
            Prefix used for portfolio output columns.

        Returns
        -------
        pandas.DataFrame
            Portfolio value, daily return and cumulative return.
        """

        curves = []

        for company, result in individual_results.items():
            if result is None or result.empty:
                continue

            if value_column not in result.columns:
                continue

            curve = result[value_column].copy()
            curve.name = company
            curves.append(curve)

        if not curves:
            raise ValueError(
                f"No valid individual curves found for column '{value_column}'."
            )

        portfolio = pd.concat(curves, axis=1).sort_index()

        # Forward-fill values so that companies with missing dates do not
        # disappear from the portfolio after the first valid observation.
        portfolio = portfolio.ffill()

        # Drop dates where all companies are missing.
        portfolio = portfolio.dropna(how="all")

        if portfolio.empty:
            raise ValueError("Portfolio DataFrame is empty after alignment.")

        # Equal-weight portfolio:
        # Each available company has the same weight on each date.
        portfolio[f"{portfolio_name}_value"] = portfolio.mean(axis=1)

        portfolio[f"{portfolio_name}_return"] = (
            portfolio[f"{portfolio_name}_value"]
            .pct_change()
            .fillna(0.0)
        )

        first_value = portfolio[f"{portfolio_name}_value"].iloc[0]

        portfolio[f"{portfolio_name}_cumulative_return"] = (
            portfolio[f"{portfolio_name}_value"]
            / first_value
            - 1.0
        )

        return portfolio[
            [
                f"{portfolio_name}_value",
                f"{portfolio_name}_return",
                f"{portfolio_name}_cumulative_return",
            ]
        ]

    def build_portfolio_comparison(
        self,
        individual_comparisons: dict,
    ) -> pd.DataFrame:
        """
        Builds portfolio-level comparison from individual comparison tables.

        This creates:
        - aggregate strategy portfolio;
        - aggregate passive 50/50 portfolio;
        - aggregate Ibovespa benchmark.

        The method is useful for the second project stage.
        """

        strategy_portfolio = self.build_equal_weight_portfolio(
            individual_results=individual_comparisons,
            value_column="strategy_value",
            portfolio_name="strategy_portfolio",
        )

        benchmark_50_50_portfolio = self.build_equal_weight_portfolio(
            individual_results=individual_comparisons,
            value_column="benchmark_50_50_value",
            portfolio_name="benchmark_50_50_portfolio",
        )

        ibovespa_portfolio = self.build_equal_weight_portfolio(
            individual_results=individual_comparisons,
            value_column="ibovespa_value",
            portfolio_name="ibovespa_portfolio",
        )

        portfolio_comparison = pd.concat(
            [
                strategy_portfolio,
                benchmark_50_50_portfolio,
                ibovespa_portfolio,
            ],
            axis=1,
            join="inner",
        )

        portfolio_comparison = portfolio_comparison.sort_index()

        portfolio_comparison["strategy_minus_50_50"] = (
            portfolio_comparison["strategy_portfolio_cumulative_return"]
            - portfolio_comparison["benchmark_50_50_portfolio_cumulative_return"]
        )

        portfolio_comparison["strategy_minus_ibovespa"] = (
            portfolio_comparison["strategy_portfolio_cumulative_return"]
            - portfolio_comparison["ibovespa_portfolio_cumulative_return"]
        )

        return portfolio_comparison

    def build_custom_weight_portfolio(
        self,
        individual_results: dict,
        value_column: str,
        weights: dict,
        portfolio_name: str = "custom_portfolio",
    ) -> pd.DataFrame:
        """
        Builds a custom-weight portfolio from individual equity curves.

        This method is not used in the first stage, but it is useful later
        when company weights are defined manually or by an optimization rule.

        Parameters
        ----------
        individual_results:
            Dictionary mapping company code to result DataFrame.

        value_column:
            Column containing the equity curve to aggregate.

        weights:
            Dictionary mapping company code to portfolio weight.

            Example:
            {
                "PETR": 0.20,
                "ITUB": 0.15,
                "BBDC": 0.15,
            }

        portfolio_name:
            Prefix used for output columns.
        """

        if not weights:
            raise ValueError("Weights dictionary is empty.")

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError("Total portfolio weight must be positive.")

        normalized_weights = {
            company: weight / total_weight
            for company, weight in weights.items()
        }

        weighted_curves = []

        for company, weight in normalized_weights.items():
            if company not in individual_results:
                continue

            result = individual_results[company]

            if result is None or result.empty:
                continue

            if value_column not in result.columns:
                continue

            curve = result[value_column].copy()
            curve = curve / curve.iloc[0]

            weighted_curve = curve * weight
            weighted_curve.name = company

            weighted_curves.append(weighted_curve)

        if not weighted_curves:
            raise ValueError("No valid curves available for custom portfolio.")

        portfolio = pd.concat(weighted_curves, axis=1).sort_index()
        portfolio = portfolio.ffill().dropna(how="all")

        portfolio[f"{portfolio_name}_value"] = portfolio.sum(axis=1)

        portfolio[f"{portfolio_name}_return"] = (
            portfolio[f"{portfolio_name}_value"]
            .pct_change()
            .fillna(0.0)
        )

        portfolio[f"{portfolio_name}_cumulative_return"] = (
            portfolio[f"{portfolio_name}_value"]
            / portfolio[f"{portfolio_name}_value"].iloc[0]
            - 1.0
        )

        return portfolio[
            [
                f"{portfolio_name}_value",
                f"{portfolio_name}_return",
                f"{portfolio_name}_cumulative_return",
            ]
        ]