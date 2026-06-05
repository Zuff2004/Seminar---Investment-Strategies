from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class PlotBuilder:
    """
    Builds and saves project plots.

    This class is responsible only for visualization.
    It does not run backtests.
    It does not calculate metrics.
    It does not change strategy results.

    Main plots:
    - individual cumulative return comparison;
    - individual equity value comparison;
    - strategy excess return vs 50/50 and Ibovespa;
    - ON allocation through time;
    - z-score and trading signals;
    - portfolio-level cumulative return comparison.
    """

    def __init__(
        self,
        plots_dir: str | Path,
        figure_size: tuple = (12, 6),
        dpi: int = 150,
    ):
        """
        Initializes the plot builder.

        Parameters
        ----------
        plots_dir:
            Folder where all plots will be saved.

        figure_size:
            Default matplotlib figure size.

        dpi:
            Resolution of the saved PNG files.
        """

        self.plots_dir = Path(plots_dir)
        self.figure_size = figure_size
        self.dpi = int(dpi)

        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Public orchestration methods
    # ============================================================

    def build_all_individual_plots(
        self,
        individual_comparisons: dict,
    ) -> list[Path]:
        """
        Builds all individual company plots.

        Parameters
        ----------
        individual_comparisons:
            Dictionary mapping company code to comparison DataFrame.

            Expected format:
            {
                "PETR": petr_comparison,
                "ITUB": itub_comparison,
                ...
            }

        Returns
        -------
        list[Path]
            List of saved plot paths.
        """

        saved_paths = []

        for company, comparison in individual_comparisons.items():
            if comparison is None or comparison.empty:
                print(f"Skipping plots for {company}: empty comparison.")
                continue

            print(f"Building plots for {company}...")

            saved_paths.append(
                self.plot_individual_cumulative_returns(
                    company=company,
                    comparison=comparison,
                )
            )

            saved_paths.append(
                self.plot_individual_equity_values(
                    company=company,
                    comparison=comparison,
                )
            )

            saved_paths.append(
                self.plot_individual_excess_returns(
                    company=company,
                    comparison=comparison,
                )
            )

            saved_paths.append(
                self.plot_on_weight(
                    company=company,
                    comparison=comparison,
                )
            )

            saved_paths.append(
                self.plot_z_score_and_signals(
                    company=company,
                    comparison=comparison,
                )
            )

        return saved_paths

    def build_portfolio_plots(
        self,
        portfolio_comparison: pd.DataFrame,
    ) -> list[Path]:
        """
        Builds portfolio-level plots.

        Parameters
        ----------
        portfolio_comparison:
            Output from PortfolioEngine.build_portfolio_comparison.

        Returns
        -------
        list[Path]
            List of saved plot paths.
        """

        if portfolio_comparison is None or portfolio_comparison.empty:
            print("Skipping portfolio plots: empty portfolio comparison.")
            return []

        saved_paths = []

        saved_paths.append(
            self.plot_portfolio_cumulative_returns(
                portfolio_comparison=portfolio_comparison,
            )
        )

        saved_paths.append(
            self.plot_portfolio_excess_returns(
                portfolio_comparison=portfolio_comparison,
            )
        )

        return saved_paths

    # ============================================================
    # Individual company plots
    # ============================================================

    def plot_individual_cumulative_returns(
        self,
        company: str,
        comparison: pd.DataFrame,
    ) -> Path:
        """
        Plots cumulative returns for:
        - ON/PN rotation strategy;
        - passive 50/50 benchmark;
        - Ibovespa benchmark.
        """

        required_columns = [
            "strategy_cumulative_return",
            "benchmark_50_50_cumulative_return",
            "ibovespa_cumulative_return",
        ]

        self._validate_columns(comparison, required_columns)

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(
            comparison.index,
            comparison["strategy_cumulative_return"] * 100,
            label="Strategy",
            linewidth=2,
        )

        ax.plot(
            comparison.index,
            comparison["benchmark_50_50_cumulative_return"] * 100,
            label="50/50 ON-PN",
            linewidth=2,
        )

        ax.plot(
            comparison.index,
            comparison["ibovespa_cumulative_return"] * 100,
            label="Ibovespa",
            linewidth=2,
        )

        ax.set_title(f"{company} - Cumulative Return Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(
            fig=fig,
            filename=f"{company}_cumulative_returns.png",
        )

    def plot_individual_equity_values(
        self,
        company: str,
        comparison: pd.DataFrame,
    ) -> Path:
        """
        Plots normalized equity/value curves.

        This is useful because all curves start from the same initial capital.
        """

        required_columns = [
            "strategy_value",
            "benchmark_50_50_value",
            "ibovespa_value",
        ]

        self._validate_columns(comparison, required_columns)

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(
            comparison.index,
            comparison["strategy_value"],
            label="Strategy",
            linewidth=2,
        )

        ax.plot(
            comparison.index,
            comparison["benchmark_50_50_value"],
            label="50/50 ON-PN",
            linewidth=2,
        )

        ax.plot(
            comparison.index,
            comparison["ibovespa_value"],
            label="Ibovespa",
            linewidth=2,
        )

        ax.set_title(f"{company} - Equity Value Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(
            fig=fig,
            filename=f"{company}_equity_values.png",
        )

    def plot_individual_excess_returns(
        self,
        company: str,
        comparison: pd.DataFrame,
    ) -> Path:
        """
        Plots strategy excess return versus:
        - passive 50/50 benchmark;
        - Ibovespa benchmark.
        """

        required_columns = [
            "strategy_minus_50_50",
            "strategy_minus_ibovespa",
        ]

        self._validate_columns(comparison, required_columns)

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(
            comparison.index,
            comparison["strategy_minus_50_50"] * 100,
            label="Strategy - 50/50",
            linewidth=2,
        )

        ax.plot(
            comparison.index,
            comparison["strategy_minus_ibovespa"] * 100,
            label="Strategy - Ibovespa",
            linewidth=2,
        )

        ax.axhline(0, linewidth=1)

        ax.set_title(f"{company} - Strategy Excess Return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Excess return (percentage points)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(
            fig=fig,
            filename=f"{company}_excess_returns.png",
        )

    def plot_on_weight(
        self,
        company: str,
        comparison: pd.DataFrame,
    ) -> Path:
        """
        Plots the realized ON weight and target ON weight.

        This shows how aggressively the strategy rotates between ON and PN.
        """

        possible_realized_columns = [
            "weight_on",
            "current_weight_on",
            "actual_weight_on",
        ]

        realized_column = self._find_first_existing_column(
            comparison,
            possible_realized_columns,
        )

        required_columns = ["target_weight_on"]

        self._validate_columns(comparison, required_columns)

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(
            comparison.index,
            comparison["target_weight_on"] * 100,
            label="Target ON weight",
            linewidth=2,
        )

        if realized_column is not None:
            ax.plot(
                comparison.index,
                comparison[realized_column] * 100,
                label="Realized ON weight",
                linewidth=2,
                alpha=0.8,
            )

        ax.axhline(50, linewidth=1, linestyle="--")

        ax.set_title(f"{company} - ON Allocation Through Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("ON weight (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(
            fig=fig,
            filename=f"{company}_on_weight.png",
        )

    def plot_z_score_and_signals(
        self,
        company: str,
        comparison: pd.DataFrame,
    ) -> Path:
        """
        Plots the spread z-score.

        If trade information exists, the plot marks trade days.
        """

        required_columns = ["z_score"]

        if "z_score" not in comparison.columns:
            print(f"Skipping z-score plot for {company}: missing z_score.")
            return self._empty_plot_path(company, "z_score_signals")

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(
            comparison.index,
            comparison["z_score"],
            label="Spread z-score",
            linewidth=2,
        )

        ax.axhline(0, linewidth=1)
        ax.axhline(1, linewidth=1, linestyle="--")
        ax.axhline(-1, linewidth=1, linestyle="--")
        ax.axhline(2, linewidth=1, linestyle=":")
        ax.axhline(-2, linewidth=1, linestyle=":")

        trade_column = self._find_first_existing_column(
            comparison,
            ["traded", "trade_executed"],
        )

        if trade_column is not None:
            trade_days = comparison[comparison[trade_column] == True]

            if not trade_days.empty:
                ax.scatter(
                    trade_days.index,
                    trade_days["z_score"],
                    label="Trade days",
                    s=25,
                    zorder=3,
                )

        ax.set_title(f"{company} - Spread Z-Score and Trade Signals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Z-score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(
            fig=fig,
            filename=f"{company}_z_score_signals.png",
        )

    # ============================================================
    # Portfolio plots
    # ============================================================

    def plot_portfolio_cumulative_returns(
        self,
        portfolio_comparison: pd.DataFrame,
    ) -> Path:
        """
        Plots portfolio-level cumulative returns.
        """

        required_columns = [
            "strategy_portfolio_cumulative_return",
            "benchmark_50_50_portfolio_cumulative_return",
            "ibovespa_portfolio_cumulative_return",
        ]

        self._validate_columns(portfolio_comparison, required_columns)

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(
            portfolio_comparison.index,
            portfolio_comparison["strategy_portfolio_cumulative_return"] * 100,
            label="Strategy portfolio",
            linewidth=2,
        )

        ax.plot(
            portfolio_comparison.index,
            portfolio_comparison["benchmark_50_50_portfolio_cumulative_return"] * 100,
            label="50/50 portfolio",
            linewidth=2,
        )

        ax.plot(
            portfolio_comparison.index,
            portfolio_comparison["ibovespa_portfolio_cumulative_return"] * 100,
            label="Ibovespa portfolio",
            linewidth=2,
        )

        ax.set_title("Portfolio - Cumulative Return Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(
            fig=fig,
            filename="portfolio_cumulative_returns.png",
        )

    def plot_portfolio_excess_returns(
        self,
        portfolio_comparison: pd.DataFrame,
    ) -> Path:
        """
        Plots portfolio-level excess returns.
        """

        required_columns = [
            "strategy_minus_50_50",
            "strategy_minus_ibovespa",
        ]

        self._validate_columns(portfolio_comparison, required_columns)

        fig, ax = plt.subplots(figsize=self.figure_size)

        ax.plot(
            portfolio_comparison.index,
            portfolio_comparison["strategy_minus_50_50"] * 100,
            label="Strategy portfolio - 50/50 portfolio",
            linewidth=2,
        )

        ax.plot(
            portfolio_comparison.index,
            portfolio_comparison["strategy_minus_ibovespa"] * 100,
            label="Strategy portfolio - Ibovespa",
            linewidth=2,
        )

        ax.axhline(0, linewidth=1)

        ax.set_title("Portfolio - Strategy Excess Return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Excess return (percentage points)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self._save_figure(
            fig=fig,
            filename="portfolio_excess_returns.png",
        )

    # ============================================================
    # Helper methods
    # ============================================================

    def _save_figure(
        self,
        fig,
        filename: str,
    ) -> Path:
        """
        Saves and closes a matplotlib figure.
        """

        output_path = self.plots_dir / filename

        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot to: {output_path}")

        return output_path

    def _validate_columns(
        self,
        data: pd.DataFrame,
        required_columns: list[str],
    ):
        """
        Validates whether all required columns exist.
        """

        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Missing column for plot: {column}")

    def _find_first_existing_column(
        self,
        data: pd.DataFrame,
        candidates: list[str],
    ) -> str | None:
        """
        Returns the first candidate column found in the DataFrame.
        """

        for column in candidates:
            if column in data.columns:
                return column

        return None

    def _empty_plot_path(
        self,
        company: str,
        plot_name: str,
    ) -> Path:
        """
        Returns the path that would have been used for a skipped plot.

        This avoids breaking the pipeline when an optional plot column is missing.
        """

        return self.plots_dir / f"{company}_{plot_name}.png"