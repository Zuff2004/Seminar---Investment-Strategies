import pandas as pd

from project_config import ProjectConfig
from data_loader import MarketDataLoader
from pair_data import PairData
from data_splitter import TimeSeriesSplitter
from universe_filter import UniverseFilter
from company_policy_engine_test import CompanyPolicyEngine
from rotation_signal_engine import RotationSignalEngine
from share_class_rotation_backtester import ShareClassRotationBacktester
from benchmarks import BenchmarkBuilder
from individual_comparison import IndividualComparisonBuilder
from plot_builder import PlotBuilder


def build_manual_company_pairs() -> dict:
    """
    Defines the manual ON/PN company universe from the final long-only table.

    Important:
    - The portfolio weights from the table are ignored in this script.
    - Each company is tested individually.
    - The preferred ticker shown in the table is used only to identify
      the company, but the strategy requires both ON and PN/share-class tickers.
    """

    return {
        "ITUB": ("ITUB3.SA", "ITUB4.SA"),
        "ISAE": ("ISAE3.SA", "ISAE4.SA"),
        "ALUP": ("ALUP3.SA", "ALUP4.SA"),
        "PETR": ("PETR3.SA", "PETR4.SA"),
        "SAPR": ("SAPR3.SA", "SAPR4.SA"),
        "BBDC": ("BBDC3.SA", "BBDC4.SA"),
        "GGBR": ("GGBR3.SA", "GGBR4.SA"),
        "UNIP": ("UNIP3.SA", "UNIP6.SA"),
        "TAEE": ("TAEE3.SA", "TAEE4.SA"),
        "RAPT": ("RAPT3.SA", "RAPT4.SA"),

        # Banco BTG Pactual
        # Table implementation ticker: BPAC5
        # Strategy requires both share classes:
        # ON = BPAC3.SA
        # PN/share-class ticker = BPAC5.SA
        "BTG": ("BPAC3.SA", "BPAC5.SA"),
    }


def build_pair_objects(config: ProjectConfig) -> list:
    """
    Loads market data and builds PairData objects for all companies.

    Each PairData object contains:
    - ON prices;
    - PN/share-class prices;
    - Ibovespa prices;
    - ON returns;
    - PN/share-class returns;
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
    Splits each company into fixed chronological train and test samples.

    Correct logic:
    - train = all observations before 2020-01-01;
    - test  = all observations on or after 2020-01-01.

    This replaces the old train_ratio logic.
    """

    train_data_by_company = {}
    test_data_by_company = {}
    split_records = []

    test_start_date = pd.Timestamp(config.backtest.test_start_date)
    end_date = pd.Timestamp(config.backtest.end_date)

    for pair in pair_objects:
        data = pair.data.copy()

        if data.empty:
            print(f"Skipping {pair.company}: empty data.")
            continue

        data = data.sort_index()

        # Keep only data up to configured end date.
        data = data[data.index <= end_date]

        train_data = data[data.index < test_start_date].copy()
        test_data = data[data.index >= test_start_date].copy()

        if train_data.empty:
            print(f"Skipping {pair.company}: no training data before {test_start_date.date()}.")
            continue

        if test_data.empty:
            print(f"Skipping {pair.company}: no test data from {test_start_date.date()} onward.")
            continue

        train_data_by_company[pair.company] = train_data
        test_data_by_company[pair.company] = test_data

        split_records.append({
            "company": pair.company,
            "train_start": train_data.index.min().date(),
            "train_end": train_data.index.max().date(),
            "train_observations": len(train_data),
            "test_start": test_data.index.min().date(),
            "test_end": test_data.index.max().date(),
            "test_observations": len(test_data),
        })

    split_summary = pd.DataFrame(split_records)

    return train_data_by_company, test_data_by_company, split_summary


def run_universe_filter(
    config: ProjectConfig,
    pair_objects: list,
    train_data_by_company: dict,
) -> tuple[list, pd.DataFrame]:
    """
    Applies the universe filter using only training data.

    In this test file, the filter is used mainly to calculate the training
    statistics needed by the policy engine.

    The final tested companies are later forced manually, so the filter does
    not decide the final manual test universe.
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
        top_n=None,
    )

    output_path = (
        config.paths.tables_dir
        / "manual_company_universe_filter_report.csv"
    )

    filter_report.to_csv(
        output_path,
        index=False,
    )

    print("\nUniverse filter completed.")
    print(f"Pairs passing hard filters: {len(selected_pairs)}")
    print(f"Saved manual filter report to: {output_path}")

    return selected_pairs, filter_report


def build_company_policies(
    config: ProjectConfig,
    filter_report: pd.DataFrame,
    forced_companies: list | set | tuple | None = None,
) -> dict:
    """
    Builds company behavior policies from training-sample statistics.

    For this manual test, forced_companies allows the policy engine to create
    policies even for companies that did not pass the universe hard filters.

    This is intentional because the goal of main_test.py is exploratory:
    test the selected companies individually, not build the final portfolio.
    """

    policy_engine = CompanyPolicyEngine(
        policy_settings=config.policies,
    )

    policy_map = policy_engine.build_policy_map(
        filter_report=filter_report,
        forced_companies=forced_companies,
    )

    policy_table = policy_engine.build_policy_table(policy_map)

    output_path = (
        config.paths.tables_dir
        / "manual_company_policy_map.csv"
    )

    policy_table.to_csv(
        output_path,
        index=False,
    )

    print("\nCompany policies created.")
    print(f"Saved manual policy map to: {output_path}")

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

            signal_data = signal_engine.add_signals(
                data=test_data,
                policy=policy,
            )

            strategy_result = backtester.backtest_pair(
                data=signal_data,
                pair_name=company,
            )

            benchmark_result = benchmark_builder.build_all_benchmarks(
                data=test_data,
            )

            comparison, metrics = comparison_builder.build_comparison(
                company=company,
                strategy_result=strategy_result,
                benchmark_result=benchmark_result,
            )

            metrics["policy_group"] = policy.policy_group

            output_path = (
                config.paths.individual_results_dir
                / f"{company}_manual_individual_comparison.csv"
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
                f"Policy: {policy.policy_group} | "
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

    output_path = (
        config.paths.tables_dir
        / "manual_company_individual_strategy_vs_benchmarks.csv"
    )

    metrics_table.to_csv(
        output_path,
        index=False,
    )

    print("\nIndividual manual backtests completed.")
    print(f"Saved manual metrics table to: {output_path}")

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

    print("\nFinal manual individual comparison summary")
    print("=" * 120)
    print(summary.to_string(index=False))


def main():
    """
    Runs a separate manual ON/PN rotation test for the companies from the
    final long-only table.

    This file intentionally does not run the full project portfolio logic.

    Outputs:
    - manual company universe filter report;
    - manual company policy map;
    - individual strategy vs 50/50 vs Ibovespa CSVs;
    - individual company plots;
    - manual company-level metrics table.

    Portfolio weights are ignored in this script.
    """

    config = ProjectConfig()
    config.initialize_project()

    # ------------------------------------------------------------
    # Important:
    # This allows missing files such as BPAC3.csv and ISAE3.csv
    # to be downloaded automatically instead of requiring local CSVs.
    # ------------------------------------------------------------

    config.backtest.download_data = True

    print("\nStarting manual company-level ON/PN rotation test")
    print("=" * 120)

    # ------------------------------------------------------------
    # 0. Manual company universe.
    # ------------------------------------------------------------

    manual_company_pairs = build_manual_company_pairs()

    config.universe.company_pairs = manual_company_pairs

    # Do not restrict the manual test by top_n.
    config.universe_filter.top_n_selected_companies = None

    print("\nManual test universe:")
    for company, tickers in manual_company_pairs.items():
        print(f"{company}: ON={tickers[0]} | PN={tickers[1]}")

    # ------------------------------------------------------------
    # 1. Load and prepare all manual company pairs.
    # ------------------------------------------------------------

    pair_objects = build_pair_objects(config)

    if not pair_objects:
        raise ValueError("No valid pair objects were created.")

    loaded_companies = [pair.company for pair in pair_objects]

    print("\nLoaded companies:")
    for company in loaded_companies:
        print(f"- {company}")

    missing_companies = [
        company
        for company in manual_company_pairs.keys()
        if company not in loaded_companies
    ]

    if missing_companies:
        print("\nWarning: some manual companies were not loaded:")
        for company in missing_companies:
            print(f"- {company}")

    # ------------------------------------------------------------
    # 2. Split data into train and test samples.
    # ------------------------------------------------------------

    train_data_by_company, test_data_by_company, split_summary = (
        build_train_test_split(
            config=config,
            pair_objects=pair_objects,
        )
    )

    split_summary_path = (
        config.paths.tables_dir
        / "manual_company_train_test_split_summary.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False,
    )

    print(f"\nSaved manual train-test split summary to: {split_summary_path}")

    # ------------------------------------------------------------
    # 3. Apply universe filter using only training data.
    # ------------------------------------------------------------
    # The filter is used to calculate statistics.
    # It does not decide the final manual test universe.
    # ------------------------------------------------------------

    selected_pairs_from_filter, filter_report = run_universe_filter(
        config=config,
        pair_objects=pair_objects,
        train_data_by_company=train_data_by_company,
    )

    print("\nPairs passing hard filters:")
    if selected_pairs_from_filter:
        for pair in selected_pairs_from_filter:
            print(f"- {pair.company}")
    else:
        print("- None")

    # ------------------------------------------------------------
    # 4. Force the manual companies into the individual test.
    # ------------------------------------------------------------

    selected_pairs = [
        pair
        for pair in pair_objects
        if pair.company in manual_company_pairs.keys()
    ]

    if not selected_pairs:
        raise ValueError("No manual company pairs are available for testing.")

    print("\nCompanies forced into individual backtest:")
    for pair in selected_pairs:
        print(f"- {pair.company}: {pair.on_ticker}/{pair.pn_ticker}")

    # ------------------------------------------------------------
    # 5. Build company policies.
    # ------------------------------------------------------------
    # forced_companies activates the exception in CompanyPolicyEngine:
    # even if a company failed hard filters, it still receives a defensive
    # exploratory policy, so that the manual test can run.
    # ------------------------------------------------------------

    policy_map = build_company_policies(
        config=config,
        filter_report=filter_report,
        forced_companies=manual_company_pairs.keys(),
    )

    if not policy_map:
        raise ValueError("No company policies were created.")

    missing_policy_companies = [
        pair.company
        for pair in selected_pairs
        if pair.company not in policy_map
    ]

    if missing_policy_companies:
        print("\nWarning: some companies have no policy and will be skipped:")
        for company in missing_policy_companies:
            print(f"- {company}")

    selected_pairs = [
        pair
        for pair in selected_pairs
        if pair.company in policy_map
    ]

    if not selected_pairs:
        raise ValueError("No selected manual pairs have available policies.")

    # ------------------------------------------------------------
    # 6. Run individual test-period backtests.
    # ------------------------------------------------------------

    individual_comparisons, metrics_table = run_individual_backtests(
        config=config,
        selected_pairs=selected_pairs,
        test_data_by_company=test_data_by_company,
        policy_map=policy_map,
    )

    # ------------------------------------------------------------
    # 7. Build individual company plots.
    # ------------------------------------------------------------

    build_individual_plots(
        config=config,
        individual_comparisons=individual_comparisons,
    )

    # ------------------------------------------------------------
    # 8. Print final terminal summary.
    # ------------------------------------------------------------

    print_final_summary(metrics_table)

    print("\nManual company-level test completed successfully.")
    print(f"Results folder: {config.paths.results_dir}")
    print(f"Plots folder: {config.paths.plots_dir}")


if __name__ == "__main__":
    main()