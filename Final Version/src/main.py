import pandas as pd

from project_config import ProjectConfig
from data_loader import MarketDataLoader
from pair_data import PairData
from data_splitter import TimeSeriesSplitter
from universe_filter import UniverseFilter
from company_policy_engine import CompanyPolicyEngine
from rotation_signal_engine import RotationSignalEngine
from share_class_rotation_backtester import ShareClassRotationBacktester
from benchmarks import BenchmarkBuilder
from individual_comparison import IndividualComparisonBuilder
from plot_builder import PlotBuilder


def build_pair_objects(config: ProjectConfig) -> list:
    """
    Loads market data and builds PairData objects for all companies.

    Each PairData object contains:
    - ON prices;
    - PN prices;
    - Ibovespa prices;
    - ON returns;
    - PN returns;
    - Ibovespa returns.
    """

    loader = MarketDataLoader(
        raw_data_dir=config.paths.raw_data_dir,
        download=config.backtest.download_data,
    )

    if config.backtest.download_data:
        loader.download_project_universe(
            company_pairs=config.universe.company_pairs,
            ibovespa_ticker=config.universe.ibovespa_ticker,
            start_date=config.backtest.start_date,
            end_date=config.backtest.end_date,
        )

    pair_objects = []

    for company, tickers in config.universe.company_pairs.items():
        on_ticker, pn_ticker = tickers

        try:
            price_data = loader.load_pair_prices(
                company=company,
                on_ticker=on_ticker,
                pn_ticker=pn_ticker,
                ibovespa_ticker=config.universe.ibovespa_ticker,
            )

            volume_data = loader.load_pair_volumes(
                on_ticker=on_ticker,
                pn_ticker=pn_ticker,
            )

            pair_data = PairData(
                company=company,
                on_ticker=on_ticker,
                pn_ticker=pn_ticker,
                price_data=price_data,
                volume_data=volume_data,
            )

            pair_objects.append(pair_data)

            print(
                f"Loaded {company}: "
                f"{pair_data.data.index.min().date()} -> "
                f"{pair_data.data.index.max().date()} "
                f"({len(pair_data.data)} observations)"
            )

        except Exception as error:
            print(f"Skipping {company}: {error}")

    return pair_objects


def build_train_test_split(
    config: ProjectConfig,
    pair_objects: list,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Splits all pairs into train and test samples.

    The split is chronological:
    - train = oldest observations;
    - test = newest observations.
    """

    splitter = TimeSeriesSplitter(
        train_ratio=config.backtest.train_ratio,
    )

    split_data = splitter.split_pair_objects(pair_objects)

    train_data_by_company = {}
    test_data_by_company = {}

    for company, content in split_data.items():
        train_data_by_company[company] = content["train"]
        test_data_by_company[company] = content["test"]

    split_summary = splitter.build_split_summary(split_data)

    return train_data_by_company, test_data_by_company, split_summary


def run_universe_filter(
    config: ProjectConfig,
    pair_objects: list,
    train_data_by_company: dict,
) -> tuple[list, pd.DataFrame]:
    """
    Applies the universe filter using only training data.

    This prevents future test-period information from affecting
    universe selection.
    """

    universe_filter = UniverseFilter(
        min_observations=config.universe_filter.min_observations,
        max_missing_ratio=config.universe_filter.max_missing_ratio,
        min_avg_volume=config.universe_filter.min_avg_volume,
        min_basic_correlation=config.universe_filter.min_basic_correlation,
        use_cointegration=config.universe_filter.use_cointegration,
        use_adf=config.universe_filter.use_adf,
        require_volume_data=config.universe_filter.require_volume_data,
    )

    selected_pairs, filter_report = universe_filter.filter_pairs(
        pair_objects=pair_objects,
        train_data_by_company=train_data_by_company,
        top_n=config.universe_filter.top_n_selected_companies,
    )

    filter_report.to_csv(
        config.paths.universe_filter_report_path,
        index=False,
    )

    print("\nUniverse filter completed.")
    print(f"Selected pairs: {len(selected_pairs)}")
    print(f"Saved filter report to: {config.paths.universe_filter_report_path}")

    return selected_pairs, filter_report


def build_company_policies(
    config: ProjectConfig,
    filter_report: pd.DataFrame,
) -> dict:
    """
    Builds company behavior policies from training-sample statistics.

    These policies define whether each company should use:
    - conservative preservation;
    - active reversion;
    - moderate rotation;
    - extreme-only rotation;
    - defensive rotation.
    """

    policy_engine = CompanyPolicyEngine(
        policy_settings=config.policies,
    )

    policy_map = policy_engine.build_policy_map(filter_report)

    policy_table = policy_engine.build_policy_table(policy_map)

    policy_table.to_csv(
        config.paths.policy_map_path,
        index=False,
    )

    print("\nCompany policies created.")
    print(f"Saved policy map to: {config.paths.policy_map_path}")

    return policy_map


def run_individual_backtests(
    config: ProjectConfig,
    selected_pairs: list,
    test_data_by_company: dict,
    policy_map: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Runs the individual strategy and benchmarks for each selected company.

    First-stage comparison:
    - active ON/PN rotation strategy;
    - passive 50/50 ON/PN buy-and-hold;
    - Ibovespa buy-and-hold.
    """

    signal_engine = RotationSignalEngine(
        initial_weight_on=config.signals.initial_weight_on,
        initial_weight_pn=config.signals.initial_weight_pn,
        minimum_signal_observations=config.signals.minimum_signal_observations,
    )

    backtester = ShareClassRotationBacktester(
        initial_capital=config.backtest.initial_capital_per_pair,
        transaction_cost_rate=config.backtest.transaction_cost_rate,
        tax_rate=config.backtest.income_tax_rate,
        minimum_rebalance_difference=config.backtest.minimum_rebalance_difference,
        include_transaction_costs_in_tax_basis=(
            config.backtest.include_transaction_costs_in_tax_basis
        ),
        use_loss_carryforward=config.backtest.use_loss_carryforward,
    )

    benchmark_builder = BenchmarkBuilder(
        initial_capital=config.backtest.initial_capital_per_pair,
    )

    comparison_builder = IndividualComparisonBuilder(
        trading_days_per_year=config.backtest.trading_days_per_year,
    )

    individual_comparisons = {}
    metrics_by_company = []

    for pair in selected_pairs:
        company = pair.company

        if company not in test_data_by_company:
            print(f"Skipping {company}: no test data available.")
            continue

        if company not in policy_map:
            print(f"Skipping {company}: no policy available.")
            continue

        try:
            print(f"\nRunning individual backtest for {company}...")

            test_data = test_data_by_company[company].copy()
            policy = policy_map[company]

            # --------------------------------------------------------
            # 1. Generate target ON/PN weights from the spread signal.
            # --------------------------------------------------------

            signal_data = signal_engine.add_signals(
                data=test_data,
                policy=policy,
            )

            # --------------------------------------------------------
            # 2. Run active ON/PN rotation strategy.
            # --------------------------------------------------------

            strategy_result = backtester.backtest_pair(
                data=signal_data,
                pair_name=company,
            )

            # --------------------------------------------------------
            # 3. Build passive benchmarks.
            # --------------------------------------------------------

            benchmark_result = benchmark_builder.build_all_benchmarks(
                data=test_data,
            )

            # --------------------------------------------------------
            # 4. Merge strategy and benchmarks.
            # --------------------------------------------------------

            comparison, metrics = comparison_builder.build_comparison(
                company=company,
                strategy_result=strategy_result,
                benchmark_result=benchmark_result,
            )

            # --------------------------------------------------------
            # 5. Save individual daily comparison.
            # --------------------------------------------------------

            output_path = (
                config.paths.individual_results_dir
                / f"{company}_individual_comparison.csv"
            )

            comparison_builder.save_individual_comparison(
                comparison=comparison,
                output_path=output_path,
            )

            individual_comparisons[company] = comparison
            metrics_by_company.append(metrics)

            print(f"Saved {company} comparison to: {output_path}")
            print(
                f"{company} | "
                f"Strategy: {metrics['strategy_total_return']:.2%} | "
                f"50/50: {metrics['benchmark_50_50_total_return']:.2%} | "
                f"Ibovespa: {metrics['ibovespa_total_return']:.2%} | "
                f"Trades: {metrics['number_of_trade_days']}"
            )

        except Exception as error:
            print(f"Error while running {company}: {error}")

    metrics_table = comparison_builder.build_metrics_table(
        metrics_by_company=metrics_by_company,
    )

    metrics_table.to_csv(
        config.paths.individual_metrics_path,
        index=False,
    )

    print("\nIndividual backtests completed.")
    print(f"Saved metrics table to: {config.paths.individual_metrics_path}")

    return individual_comparisons, metrics_table


def build_individual_plots(
    config: ProjectConfig,
    individual_comparisons: dict,
) -> list:
    """
    Builds and saves all individual company plots.

    The plots are saved in:
    final_results/plots

    Main plots:
    - cumulative returns;
    - equity values;
    - excess returns;
    - ON weight through time;
    - spread z-score and signals.
    """

    if not individual_comparisons:
        print("\nNo individual comparisons available for plotting.")
        return []

    plot_builder = PlotBuilder(
        plots_dir=config.paths.plots_dir,
    )

    saved_plot_paths = plot_builder.build_all_individual_plots(
        individual_comparisons=individual_comparisons,
    )

    print("\nIndividual plots completed.")
    print(f"Saved plots: {len(saved_plot_paths)}")
    print(f"Plots folder: {config.paths.plots_dir}")

    return saved_plot_paths


def print_final_summary(metrics_table: pd.DataFrame):
    """
    Prints a readable final summary in the terminal.
    """

    if metrics_table.empty:
        print("\nNo metrics available.")
        return

    columns_to_show = [
        "company",
        "policy_group",

        "strategy_total_return",
        "benchmark_50_50_total_return",
        "ibovespa_total_return",

        "strategy_excess_return_vs_50_50",
        "strategy_excess_return_vs_ibovespa",

        "strategy_sharpe_ratio",
        "benchmark_50_50_sharpe_ratio",
        "ibovespa_sharpe_ratio",

        "strategy_max_drawdown",
        "benchmark_50_50_max_drawdown",
        "ibovespa_max_drawdown",

        "total_tax_paid",
        "total_transaction_cost",
        "total_realized_pnl",

        "number_of_trade_days",
        "final_weight_on",
        "final_weight_pn",
        "final_accumulated_loss",
    ]

    existing_columns = [
        column
        for column in columns_to_show
        if column in metrics_table.columns
    ]

    summary = metrics_table[existing_columns].copy()

    percentage_columns = [
        "strategy_total_return",
        "benchmark_50_50_total_return",
        "ibovespa_total_return",
        "strategy_excess_return_vs_50_50",
        "strategy_excess_return_vs_ibovespa",
        "strategy_max_drawdown",
        "benchmark_50_50_max_drawdown",
        "ibovespa_max_drawdown",
        "final_weight_on",
        "final_weight_pn",
    ]

    for column in percentage_columns:
        if column in summary.columns:
            summary[column] = summary[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )

    numeric_columns = [
        "strategy_sharpe_ratio",
        "benchmark_50_50_sharpe_ratio",
        "ibovespa_sharpe_ratio",
        "total_tax_paid",
        "total_transaction_cost",
        "total_realized_pnl",
        "final_accumulated_loss",
    ]

    for column in numeric_columns:
        if column in summary.columns:
            summary[column] = summary[column].map(
                lambda value: f"{value:.4f}" if pd.notna(value) else ""
            )

    print("\nFinal individual comparison summary")
    print("=" * 120)
    print(summary.to_string(index=False))


def main():
    """
    Runs the final ON/PN rotation project pipeline.

    First-stage outputs:
    - universe filter report;
    - company policy map;
    - individual strategy vs 50/50 vs Ibovespa CSVs;
    - individual company plots;
    - final company-level metrics table.

    The aggregate portfolio is intentionally not used yet.
    """

    config = ProjectConfig()
    config.initialize_project()

    print("\nStarting final ON/PN rotation project")
    print("=" * 120)

    # ------------------------------------------------------------
    # 1. Load and prepare all company pairs.
    # ------------------------------------------------------------

    pair_objects = build_pair_objects(config)

    if not pair_objects:
        raise ValueError("No valid pair objects were created.")

    # ------------------------------------------------------------
    # 2. Split all company data into train and test samples.
    # ------------------------------------------------------------

    train_data_by_company, test_data_by_company, split_summary = (
        build_train_test_split(
            config=config,
            pair_objects=pair_objects,
        )
    )

    split_summary_path = (
        config.paths.tables_dir
        / "train_test_split_summary.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    print(f"\nSaved train-test split summary to: {split_summary_path}")

    # ------------------------------------------------------------
    # 3. Apply universe filter using only training data.
    # ------------------------------------------------------------

    selected_pairs, filter_report = run_universe_filter(
        config=config,
        pair_objects=pair_objects,
        train_data_by_company=train_data_by_company,
    )

    if not selected_pairs:
        raise ValueError("No pairs passed the universe filter.")

    # ------------------------------------------------------------
    # 4. Build company policies from training statistics.
    # ------------------------------------------------------------

    policy_map = build_company_policies(
        config=config,
        filter_report=filter_report,
    )

    if not policy_map:
        raise ValueError("No company policies were created.")

    # ------------------------------------------------------------
    # 5. Run individual test-period backtests.
    # ------------------------------------------------------------

    individual_comparisons, metrics_table = run_individual_backtests(
        config=config,
        selected_pairs=selected_pairs,
        test_data_by_company=test_data_by_company,
        policy_map=policy_map,
    )

    # ------------------------------------------------------------
    # 6. Build individual company plots.
    # ------------------------------------------------------------

    build_individual_plots(
        config=config,
        individual_comparisons=individual_comparisons,
    )

    # ------------------------------------------------------------
    # 7. Print final terminal summary.
    # ------------------------------------------------------------

    print_final_summary(metrics_table)

    print("\nProject completed successfully.")
    print(f"Results folder: {config.paths.results_dir}")
    print(f"Plots folder: {config.paths.plots_dir}")


if __name__ == "__main__":
    main()