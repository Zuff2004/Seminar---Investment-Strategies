import pandas as pd
import copy
from dataclasses import replace

from project_config import ProjectConfig
from data_loader import MarketDataLoader
from pair_data import PairData
from universe_filter import UniverseFilter
from company_policy_engine import CompanyPolicyEngine
from rotation_signal_engine import RotationSignalEngine
from share_class_rotation_backtester import ShareClassRotationBacktester
from benchmarks import BenchmarkBuilder
from individual_comparison import IndividualComparisonBuilder
from performance_metrics import PerformanceMetrics
from plot_builder import PlotBuilder


# ============================================================
# Robustness settings
# ============================================================
# These settings only control the optional robustness section at the end
# of main(). They do not change the baseline pipeline above.

RUN_ROBUSTNESS_ANALYSIS = True

ROBUSTNESS_ENTRY_THRESHOLDS = [1.0, 1.25, 1.5, 2.0]
ROBUSTNESS_ROLLING_WINDOWS = [63, 126, 252]
ROBUSTNESS_TRANSACTION_COSTS = [0.0, 0.0005, 0.001, 0.0025, 0.005]
ROBUSTNESS_TAX_RATES = [0.0, 0.15]


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
    output_filename: str = "company_policy_map.csv",
) -> dict:
    """
    Builds company behavior policies using one single policy engine.

    The same CompanyPolicyEngine is used for:
    - statistical-filtering companies;
    - manually selected / forced portfolio companies.

    If forced_companies is provided, the policy engine may assign policies
    to companies that did not pass the hard universe filters, as long as they
    are present in the filter report.
    """

    policy_engine = CompanyPolicyEngine(
        policy_settings=config.policies,
    )

    policy_map = policy_engine.build_policy_map(
        filter_report=filter_report,
        forced_companies=forced_companies,
    )

    policy_table = policy_engine.build_policy_table(policy_map)

    output_path = config.paths.tables_dir / output_filename

    policy_table.to_csv(
        output_path,
        index=False,
    )

    print("\nCompany policies created.")
    print(f"Saved policy map to: {output_path}")

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



# ============================================================
# Statistical-filtering stage helpers
# ============================================================

def run_statistical_universe_filter(
    config: ProjectConfig,
    pair_objects: list,
    train_data_by_company: dict,
) -> tuple[list, pd.DataFrame]:
    """
    Applies the normal statistical universe filter using training data only.

    This is the same first-stage logic from the original main.py:
    it decides which companies pass the statistical filters and should be
    reported as the statistical-filtering individual results.
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

    output_path = config.paths.tables_dir / "statistical_filtering_universe_filter_report.csv"
    filter_report.to_csv(output_path, index=False)

    print("\n[1] Statistical-filtering universe completed.")
    print(f"Selected pairs: {len(selected_pairs)}")
    print(f"Saved report to: {output_path}")

    return selected_pairs, filter_report


def fix_excess_return_metrics(metrics: dict) -> dict:
    """
    Guarantees that excess return is calculated as a difference in percentage
    points, not as a relative return or malformed ratio.
    """

    metrics = dict(metrics)

    if (
        "strategy_total_return" in metrics
        and "benchmark_50_50_total_return" in metrics
    ):
        metrics["strategy_excess_return_vs_50_50"] = (
            metrics["strategy_total_return"]
            - metrics["benchmark_50_50_total_return"]
        )

    if (
        "strategy_total_return" in metrics
        and "ibovespa_total_return" in metrics
    ):
        metrics["strategy_excess_return_vs_ibovespa"] = (
            metrics["strategy_total_return"]
            - metrics["ibovespa_total_return"]
        )

    return metrics


def run_individual_backtests_generic(
    config: ProjectConfig,
    selected_pairs: list,
    signal_data_by_company: dict,
    execution_data_by_company: dict,
    policy_map: dict,
    file_prefix: str,
    metrics_filename: str,
    policy_required: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """
    Runs individual company backtests in a reusable and consistent way.

    Critical correction:
    - signal_data_by_company may contain train + test data, so rolling
      z-scores in early 2020 can use pre-2020 history;
    - execution_data_by_company must contain only the out-of-sample test
      window;
    - strategy execution/performance is measured only on the test window;
    - passive benchmarks are always built only from the same test window.

    Consequence:
    If the same company is run once with tax and once without tax, and the
    same policy is used, then the 50/50 benchmark, signals and trade dates
    are comparable. Only the strategy value after tax should differ.
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
        execution_start_date=config.backtest.test_start_date,
        execution_end_date=config.backtest.end_date,
        signal_execution_lag=1,
    )

    benchmark_builder = BenchmarkBuilder(
        initial_capital=config.backtest.initial_capital_per_pair,
    )

    comparison_builder = IndividualComparisonBuilder(
        trading_days_per_year=config.backtest.trading_days_per_year,
    )

    individual_comparisons = {}
    metrics_by_company = []

    test_start_date = pd.Timestamp(config.backtest.test_start_date)
    end_date = pd.Timestamp(config.backtest.end_date)

    for pair in selected_pairs:
        company = pair.company

        if company not in signal_data_by_company:
            print(f"Skipping {company}: no signal data available.")
            continue

        if company not in execution_data_by_company:
            print(f"Skipping {company}: no execution/test data available.")
            continue

        if policy_required and company not in policy_map:
            print(f"Skipping {company}: no policy available.")
            continue

        try:
            print(f"\nRunning {file_prefix} individual backtest for {company}...")

            signal_input = signal_data_by_company[company].copy().sort_index()
            execution_data = execution_data_by_company[company].copy().sort_index()
            policy = policy_map[company]

            signal_input = signal_input[signal_input.index <= end_date].copy()
            execution_data = execution_data[
                (execution_data.index >= test_start_date)
                & (execution_data.index <= end_date)
            ].copy()

            if signal_input.empty:
                print(f"Skipping {company}: empty signal input data.")
                continue

            if execution_data.empty:
                print(f"Skipping {company}: empty execution/test data.")
                continue

            # --------------------------------------------------------
            # 1. Generate signals using the full signal history.
            # --------------------------------------------------------
            signal_data_full = signal_engine.add_signals(
                data=signal_input,
                policy=policy,
            )

            # --------------------------------------------------------
            # 2. Confirm that the out-of-sample execution dates exist.
            # --------------------------------------------------------
            signal_data_execution = signal_data_full.loc[
                signal_data_full.index.isin(execution_data.index)
            ].copy()
            signal_data_execution = signal_data_execution.sort_index()

            if signal_data_execution.empty:
                print(f"Skipping {company}: no signal rows in execution window.")
                continue

            # --------------------------------------------------------
            # 3. Run strategy with full signal history.
            # --------------------------------------------------------
            # Important for t+1 execution:
            # The backtester shifts target weights internally before cutting
            # the execution window. Therefore, we pass signal_data_full here
            # instead of only signal_data_execution. This allows the first
            # 2020 trading day to execute the signal observed on the last
            # available pre-2020 trading day.
            strategy_result = backtester.backtest_pair(
                data=signal_data_full,
                pair_name=company,
            )

            # --------------------------------------------------------
            # 4. Build passive benchmarks only from the same test period.
            # --------------------------------------------------------
            benchmark_result = benchmark_builder.build_all_benchmarks(
                data=execution_data,
            )

            # --------------------------------------------------------
            # 5. Merge strategy and benchmarks.
            # --------------------------------------------------------
            comparison, metrics = comparison_builder.build_comparison(
                company=company,
                strategy_result=strategy_result,
                benchmark_result=benchmark_result,
            )

            metrics = fix_excess_return_metrics(metrics)
            metrics["policy_group"] = policy.policy_group

            output_path = (
                config.paths.individual_results_dir
                / f"{company}_{file_prefix}_individual_comparison.csv"
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
                f"Excess vs 50/50: {metrics['strategy_excess_return_vs_50_50']:.2%} | "
                f"Trades: {metrics['number_of_trade_days']}"
            )

        except Exception as error:
            print(f"Error while running {company}: {error}")

    metrics_table = comparison_builder.build_metrics_table(
        metrics_by_company=metrics_by_company,
    )

    # Safety correction after table construction as well.
    if not metrics_table.empty:
        if {
            "strategy_total_return",
            "benchmark_50_50_total_return",
        }.issubset(metrics_table.columns):
            metrics_table["strategy_excess_return_vs_50_50"] = (
                metrics_table["strategy_total_return"]
                - metrics_table["benchmark_50_50_total_return"]
            )

        if {
            "strategy_total_return",
            "ibovespa_total_return",
        }.issubset(metrics_table.columns):
            metrics_table["strategy_excess_return_vs_ibovespa"] = (
                metrics_table["strategy_total_return"]
                - metrics_table["ibovespa_total_return"]
            )

    metrics_output_path = config.paths.tables_dir / metrics_filename
    metrics_table.to_csv(metrics_output_path, index=False)

    print(f"\nSaved {file_prefix} metrics table to: {metrics_output_path}")

    return individual_comparisons, metrics_table

def select_pairs_by_company(pair_objects: list, companies: set | list | tuple) -> list:
    """Keeps PairData objects whose company is present in companies."""

    companies = set(companies)
    return [pair for pair in pair_objects if pair.company in companies]

# ============================================================
# Final portfolio weights
# ============================================================

def build_final_portfolio_weights() -> dict:
    """
    Defines the final company-level portfolio weights.

    Important:
    - These weights are used only as the initial company allocation.
    - The portfolio is not periodically rebalanced back to these weights.
    - Inside each company allocation, the ON/PN rotation strategy still works.
    """

    return {
        "ITUB": 0.16,  # Itaú Unibanco
        "BBDC": 0.13,  # Banco Bradesco
        "PETR": 0.11,  # Petrobras
        "GGBR": 0.10,  # Gerdau

        "ISAE": 0.11,  # ISA Energia
        "ALUP": 0.10,  # Alupar
        "SAPR": 0.08,  # Sanepar
        "UNIP": 0.06,  # Unipar
        "TAEE": 0.05,  # Taesa
        "RAPT": 0.05,  # Randon
        "BTG": 0.05,   # Banco BTG Pactual
    }


# ============================================================
# Weight handling
# ============================================================

def normalize_weights_for_available_companies(
    weights: dict,
    individual_comparisons: dict,
) -> dict:
    """
    Keeps only companies that actually produced backtest results.

    If some weighted company is missing because of unavailable data or failed
    backtest, the remaining weights are normalized to sum to 1.
    """

    available_companies = set(individual_comparisons.keys())

    used_weights = {
        company: float(weight)
        for company, weight in weights.items()
        if company in available_companies
    }

    missing_companies = sorted(set(weights.keys()) - available_companies)

    if missing_companies:
        print("\nWarning: these companies are missing from the final portfolio:")
        for company in missing_companies:
            print(f"- {company}")

    if not used_weights:
        raise ValueError("No weighted companies are available for the portfolio.")

    total_weight = sum(used_weights.values())

    if total_weight <= 0:
        raise ValueError("Total portfolio weight must be positive.")

    normalized_weights = {
        company: weight / total_weight
        for company, weight in used_weights.items()
    }

    return normalized_weights


# ============================================================
# Signal-history data for individual backtests
# ============================================================

def build_signal_history_data_by_company(
    train_data_by_company: dict,
    test_data_by_company: dict,
) -> dict:
    """
    Builds the input data used for signal generation in the individual
    backtests.

    The strategy must measure performance only in the out-of-sample period
    starting in 2020, but rolling signals in early 2020 need historical data
    from the training period.

    Therefore, each company receives train + test data for signal calculation.
    The ShareClassRotationBacktester then starts the actual portfolio execution
    from 2020-01-01 onward.
    """

    signal_history_by_company = {}

    for company, test_df in test_data_by_company.items():
        train_df = train_data_by_company.get(company)

        frames = []

        if train_df is not None and not train_df.empty:
            frames.append(train_df)

        if test_df is not None and not test_df.empty:
            frames.append(test_df)

        if not frames:
            continue

        signal_df = pd.concat(frames, axis=0)
        signal_df = signal_df[~signal_df.index.duplicated(keep="last")]
        signal_df = signal_df.sort_index()

        signal_history_by_company[company] = signal_df

    return signal_history_by_company


def build_fixed_portfolio_test_index(
    individual_comparisons: dict,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> pd.Index:
    """
    Builds a fixed final-portfolio test index.

    The index is the union of all individual comparison dates inside the
    intended out-of-sample period. This avoids reducing the portfolio period
    to the intersection of all companies.
    """

    all_indices = []

    for comparison in individual_comparisons.values():
        if comparison is not None and not comparison.empty:
            all_indices.append(comparison.index)

    if not all_indices:
        raise ValueError("No valid individual comparison indices available.")

    common_index = all_indices[0]

    for index in all_indices[1:]:
        common_index = common_index.union(index)

    common_index = common_index.sort_values()

    common_index = common_index[
        (common_index >= pd.Timestamp(start_date))
        & (common_index <= pd.Timestamp(end_date))
    ]

    if common_index.empty:
        raise ValueError("Fixed portfolio test index is empty.")

    return common_index


# ============================================================
# Buy-and-hold company-level portfolio construction
# ============================================================

def build_initial_weight_buy_and_hold_curve(
    individual_comparisons: dict,
    weights: dict,
    value_column: str,
    output_name: str,
    common_index: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Builds a portfolio curve using only initial company weights.

    This does NOT rebalance between companies.

    Critical date correction:
    The final portfolio must not start late because one company curve has
    missing early rows. Therefore, when common_index is provided, every
    company curve is reindexed to the fixed portfolio test index.

    For missing early values, the company allocation is kept flat at 1.0
    until its first valid curve value. This prevents an accidental inner join
    from moving the portfolio start date from 2020-01-02 to a later date.
    """

    company_curves = []

    if common_index is not None:
        common_index = pd.Index(common_index).sort_values()

    for company, weight in weights.items():
        if company not in individual_comparisons:
            continue

        comparison = individual_comparisons[company]

        if comparison is None or comparison.empty:
            continue

        if value_column not in comparison.columns:
            print(f"Skipping {company}: missing column {value_column}")
            continue

        curve = comparison[value_column].copy()
        curve = pd.to_numeric(curve, errors="coerce")
        curve = curve.replace([float("inf"), float("-inf")], pd.NA)

        if common_index is not None:
            curve = curve.reindex(common_index)

        curve = curve.ffill()

        # Keep the allocation flat before the first valid observation instead
        # of dropping the beginning of the test period.
        curve = curve.fillna(1.0)

        if curve.empty:
            print(f"Skipping {company}: empty curve for {value_column}")
            continue

        first_value = float(curve.iloc[0])

        if first_value == 0:
            raise ValueError(
                f"{company} has initial value zero for {value_column}."
            )

        # Normalize company curve to start at 1.0.
        # Then multiply only by the initial portfolio weight.
        curve = curve / first_value
        curve = curve * float(weight)
        curve.name = company

        company_curves.append(curve)

    if not company_curves:
        raise ValueError(f"No valid curves found for column: {value_column}")

    portfolio = pd.concat(company_curves, axis=1).sort_index()

    if common_index is not None:
        portfolio = portfolio.reindex(common_index)

    portfolio = portfolio.ffill().fillna(0.0)

    if portfolio.empty:
        raise ValueError(
            f"Portfolio curve is empty after date alignment for {value_column}."
        )

    portfolio[f"{output_name}_value"] = portfolio.sum(axis=1)

    portfolio[f"{output_name}_return"] = (
        portfolio[f"{output_name}_value"]
        .pct_change()
        .fillna(0.0)
    )

    portfolio[f"{output_name}_cumulative_return"] = (
        portfolio[f"{output_name}_value"]
        / portfolio[f"{output_name}_value"].iloc[0]
        - 1.0
    )

    return portfolio[
        [
            f"{output_name}_value",
            f"{output_name}_return",
            f"{output_name}_cumulative_return",
        ]
    ]


# ============================================================
# Portfolio-level tax inputs
# ============================================================

def build_weighted_daily_tax_inputs(
    individual_comparisons: dict,
    weights: dict,
    common_index: pd.Index,
) -> pd.DataFrame:
    """
    Aggregates daily realized PnL, sales value, buy value and transaction costs
    across all companies using the initial company weights.

    Important:
    The weights are initial capital multipliers, not rebalancing weights.
    """

    columns_to_aggregate = [
        "realized_pnl",
        "gross_sale_value",
        "gross_buy_value",
        "transaction_cost",
    ]

    company_inputs = []

    for company, weight in weights.items():
        if company not in individual_comparisons:
            continue

        comparison = individual_comparisons[company]

        if comparison is None or comparison.empty:
            continue

        df = pd.DataFrame(index=comparison.index)

        for column in columns_to_aggregate:
            if column in comparison.columns:
                df[column] = (
                    pd.to_numeric(comparison[column], errors="coerce")
                    .fillna(0.0)
                    * float(weight)
                )
            else:
                df[column] = 0.0

        df = df.reindex(common_index).fillna(0.0)
        df["company"] = company

        company_inputs.append(df)

    if not company_inputs:
        raise ValueError("No company tax inputs available.")

    combined = pd.concat(company_inputs, axis=0).sort_index()

    daily_tax_inputs = (
        combined
        .groupby(combined.index)[columns_to_aggregate]
        .sum()
        .sort_index()
    )

    return daily_tax_inputs


# ============================================================
# Portfolio-level monthly tax with payment lag
# ============================================================

def calculate_portfolio_level_monthly_tax_with_payment_lag(
    daily_tax_inputs: pd.DataFrame,
    tax_rate: float,
    use_loss_carryforward: bool = True,
) -> pd.DataFrame:
    """
    Calculates taxes once at the full portfolio level.

    Logic:
    - realized PnL from all companies is aggregated monthly;
    - transaction costs are deducted from the monthly realized PnL;
    - monthly losses increase accumulated loss;
    - monthly gains are offset by accumulated losses;
    - tax is due only on positive taxable profit;
    - tax is paid on the last available trading day of the following month.

    Important:
    The monthly tax base is:

        monthly_tax_base = monthly_realized_pnl - monthly_transaction_cost

    This avoids taxing gross realized gains before implementation costs.
    """

    if daily_tax_inputs is None or daily_tax_inputs.empty:
        raise ValueError("Daily tax input table is empty.")

    required_columns = [
        "realized_pnl",
        "gross_sale_value",
        "transaction_cost",
    ]

    for column in required_columns:
        if column not in daily_tax_inputs.columns:
            raise ValueError(f"Missing tax input column: {column}")

    accumulated_loss = 0.0
    tax_records = []

    all_trading_dates = daily_tax_inputs.index.sort_values().unique()

    monthly_groups = daily_tax_inputs.groupby(
        daily_tax_inputs.index.to_period("M")
    )

    for month, monthly_data in monthly_groups:
        calculation_date = monthly_data.index.max()

        monthly_realized_pnl = float(monthly_data["realized_pnl"].sum())
        monthly_sales_value = float(monthly_data["gross_sale_value"].sum())
        monthly_transaction_cost = float(monthly_data["transaction_cost"].sum())

        # Correct monthly tax base:
        # gains/losses after transaction costs.
        monthly_tax_base = monthly_realized_pnl - monthly_transaction_cost

        loss_used = 0.0
        taxable_profit = 0.0
        tax_due = 0.0

        if monthly_tax_base < 0:
            if use_loss_carryforward:
                accumulated_loss += abs(monthly_tax_base)

        elif monthly_tax_base > 0:
            if use_loss_carryforward:
                loss_used = min(monthly_tax_base, accumulated_loss)
                taxable_profit = monthly_tax_base - loss_used
                accumulated_loss -= loss_used
            else:
                taxable_profit = monthly_tax_base

            tax_due = taxable_profit * float(tax_rate)

        next_month = month + 1

        possible_payment_dates = [
            date
            for date in all_trading_dates
            if date.to_period("M") == next_month
        ]

        if possible_payment_dates:
            tax_payment_date = max(possible_payment_dates)
        else:
            tax_payment_date = pd.NaT

        tax_records.append({
            "calculation_month": str(month),
            "calculation_date": calculation_date,
            "tax_payment_date": tax_payment_date,

            "monthly_realized_pnl": monthly_realized_pnl,
            "monthly_sales_value": monthly_sales_value,
            "monthly_transaction_cost": monthly_transaction_cost,
            "monthly_tax_base": monthly_tax_base,

            "loss_used": loss_used,
            "taxable_profit": taxable_profit,
            "tax_due": tax_due,
            "accumulated_loss_after": accumulated_loss,
        })

    tax_table = pd.DataFrame(tax_records)

    if tax_table.empty:
        return tax_table

    tax_table = tax_table.sort_values("calculation_date").reset_index(drop=True)

    return tax_table


# ============================================================
# Correct recursive after-tax curve
# ============================================================

def apply_portfolio_tax_recursively(
    portfolio_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applies taxes as actual cash outflows from the portfolio.

    This is the correct after-tax implementation.

    Wrong approximation:
        after_tax_value = pre_tax_value - cumulative_tax_paid

    Correct recursive logic:
        after_tax_value[t] =
            after_tax_value[t-1] * (1 + pre_tax_return[t]) - tax_paid[t]

    This means that taxes paid earlier no longer compound in later periods.
    """

    required_columns = [
        "strategy_portfolio_pre_tax_value",
        "strategy_portfolio_pre_tax_return",
        "portfolio_tax_paid",
    ]

    for column in required_columns:
        if column not in portfolio_comparison.columns:
            raise ValueError(f"Missing required column for tax application: {column}")

    portfolio_comparison = portfolio_comparison.copy()

    portfolio_comparison["portfolio_tax_paid"] = (
        pd.to_numeric(
            portfolio_comparison["portfolio_tax_paid"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    portfolio_comparison["strategy_portfolio_pre_tax_return"] = (
        pd.to_numeric(
            portfolio_comparison["strategy_portfolio_pre_tax_return"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    after_tax_values = []

    for i, (_, row) in enumerate(portfolio_comparison.iterrows()):
        if i == 0:
            # Start with the same initial normalized capital.
            after_tax_value = float(
                portfolio_comparison["strategy_portfolio_pre_tax_value"].iloc[0]
            )
        else:
            previous_after_tax_value = after_tax_values[-1]

            daily_pre_tax_return = float(
                row["strategy_portfolio_pre_tax_return"]
            )

            tax_paid_today = float(
                row["portfolio_tax_paid"]
            )

            after_tax_value = (
                previous_after_tax_value
                * (1.0 + daily_pre_tax_return)
                - tax_paid_today
            )

        if after_tax_value <= 0:
            raise ValueError(
                "After-tax strategy portfolio value became non-positive. "
                "Check tax calculation, tax scaling, and portfolio inputs."
            )

        after_tax_values.append(after_tax_value)

    portfolio_comparison["strategy_portfolio_value"] = after_tax_values

    portfolio_comparison["strategy_portfolio_return"] = (
        portfolio_comparison["strategy_portfolio_value"]
        .pct_change()
        .fillna(0.0)
    )

    portfolio_comparison["strategy_portfolio_cumulative_return"] = (
        portfolio_comparison["strategy_portfolio_value"]
        / portfolio_comparison["strategy_portfolio_value"].iloc[0]
        - 1.0
    )

    return portfolio_comparison


# ============================================================
# Final portfolio construction with portfolio-level tax
# ============================================================

def build_final_portfolio_with_portfolio_level_tax(
    config: ProjectConfig,
    individual_comparisons: dict,
    weights: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds the final weighted portfolio.

    Key features:
    - company weights are applied only at initial portfolio formation;
    - there is no rebalancing back to initial company weights;
    - individual tax is disabled before individual backtests;
    - tax is calculated once at the aggregated portfolio level;
    - tax is paid on the last available trading day of the following month;
    - tax is applied recursively as a cash outflow from portfolio value.
    """

    normalized_weights = normalize_weights_for_available_companies(
        weights=weights,
        individual_comparisons=individual_comparisons,
    )

    weights_table = pd.DataFrame(
        [
            {
                "company": company,
                "initial_portfolio_weight": weight,
            }
            for company, weight in normalized_weights.items()
        ]
    )

    weights_output_path = (
        config.paths.tables_dir
        / "final_portfolio_initial_weights.csv"
    )

    weights_table.to_csv(weights_output_path, index=False)

    print(f"\nSaved final portfolio initial weights to: {weights_output_path}")

    # ------------------------------------------------------------
    # 1. Build fixed final portfolio test index.
    # ------------------------------------------------------------
    # The final portfolio must cover the intended out-of-sample window,
    # not the date intersection created by missing early rows.
    # ------------------------------------------------------------

    common_index = build_fixed_portfolio_test_index(
        individual_comparisons=individual_comparisons,
        start_date=config.backtest.test_start_date,
        end_date=config.backtest.end_date,
    )

    # ------------------------------------------------------------
    # 2. Build pre-tax strategy portfolio.
    # ------------------------------------------------------------

    strategy_pre_tax = build_initial_weight_buy_and_hold_curve(
        individual_comparisons=individual_comparisons,
        weights=normalized_weights,
        value_column="strategy_value",
        output_name="strategy_portfolio_pre_tax",
        common_index=common_index,
    )

    # ------------------------------------------------------------
    # 3. Build benchmark portfolios on the same dates.
    # ------------------------------------------------------------

    benchmark_50_50 = build_initial_weight_buy_and_hold_curve(
        individual_comparisons=individual_comparisons,
        weights=normalized_weights,
        value_column="benchmark_50_50_value",
        output_name="benchmark_50_50_portfolio",
        common_index=common_index,
    )

    ibovespa = build_initial_weight_buy_and_hold_curve(
        individual_comparisons=individual_comparisons,
        weights=normalized_weights,
        value_column="ibovespa_value",
        output_name="ibovespa_portfolio",
        common_index=common_index,
    )

    # ------------------------------------------------------------
    # 3. Build weighted tax inputs on the same dates.
    # ------------------------------------------------------------

    daily_tax_inputs = build_weighted_daily_tax_inputs(
        individual_comparisons=individual_comparisons,
        weights=normalized_weights,
        common_index=common_index,
    )

    daily_tax_inputs_output_path = (
        config.paths.tables_dir
        / "final_portfolio_daily_tax_inputs.csv"
    )

    daily_tax_inputs.to_csv(daily_tax_inputs_output_path, index=True)

    print(f"Saved final portfolio daily tax inputs to: {daily_tax_inputs_output_path}")

    # ------------------------------------------------------------
    # 4. Calculate monthly tax at portfolio level.
    # ------------------------------------------------------------

    tax_table = calculate_portfolio_level_monthly_tax_with_payment_lag(
        daily_tax_inputs=daily_tax_inputs,
        tax_rate=config.backtest.original_income_tax_rate,
        use_loss_carryforward=config.backtest.use_loss_carryforward,
    )

    tax_output_path = (
        config.paths.tables_dir
        / "final_portfolio_monthly_tax_records.csv"
    )

    tax_table.to_csv(tax_output_path, index=False)

    print(f"Saved final portfolio monthly tax records to: {tax_output_path}")

    # ------------------------------------------------------------
    # 5. Combine portfolio curves.
    # ------------------------------------------------------------

    portfolio_comparison = pd.concat(
        [
            strategy_pre_tax,
            benchmark_50_50,
            ibovespa,
            daily_tax_inputs.add_prefix("portfolio_daily_"),
        ],
        axis=1,
        join="inner",
    ).sort_index()

    if portfolio_comparison.empty:
        raise ValueError("Final portfolio comparison is empty.")

    # ------------------------------------------------------------
    # 6. Pay tax on the last available trading day of the following month.
    # ------------------------------------------------------------

    portfolio_comparison["portfolio_tax_paid"] = 0.0

    if tax_table is not None and not tax_table.empty:
        for _, tax_row in tax_table.iterrows():
            payment_date = tax_row["tax_payment_date"]

            if pd.isna(payment_date):
                continue

            payment_date = pd.Timestamp(payment_date)

            if payment_date in portfolio_comparison.index:
                portfolio_comparison.loc[payment_date, "portfolio_tax_paid"] += float(
                    tax_row["tax_due"]
                )

    portfolio_comparison["portfolio_cumulative_tax_paid"] = (
        portfolio_comparison["portfolio_tax_paid"]
        .cumsum()
    )

    # ------------------------------------------------------------
    # 7. Build after-tax strategy curve recursively.
    # ------------------------------------------------------------
    # Important:
    # Do NOT use:
    #
    #     strategy_pre_tax_value - cumulative_tax_paid
    #
    # because that ignores the fact that taxes paid earlier stop compounding.
    # ------------------------------------------------------------

    portfolio_comparison = apply_portfolio_tax_recursively(
        portfolio_comparison=portfolio_comparison,
    )

    # ------------------------------------------------------------
    # 8. Excess return columns.
    # ------------------------------------------------------------

    portfolio_comparison["strategy_minus_50_50"] = (
        portfolio_comparison["strategy_portfolio_cumulative_return"]
        - portfolio_comparison["benchmark_50_50_portfolio_cumulative_return"]
    )

    portfolio_comparison["strategy_minus_ibovespa"] = (
        portfolio_comparison["strategy_portfolio_cumulative_return"]
        - portfolio_comparison["ibovespa_portfolio_cumulative_return"]
    )

    portfolio_comparison["strategy_pre_tax_minus_50_50"] = (
        portfolio_comparison["strategy_portfolio_pre_tax_cumulative_return"]
        - portfolio_comparison["benchmark_50_50_portfolio_cumulative_return"]
    )

    portfolio_comparison["strategy_pre_tax_minus_ibovespa"] = (
        portfolio_comparison["strategy_portfolio_pre_tax_cumulative_return"]
        - portfolio_comparison["ibovespa_portfolio_cumulative_return"]
    )

    comparison_output_path = (
        config.paths.tables_dir
        / "final_portfolio_comparison_portfolio_level_tax.csv"
    )

    portfolio_comparison.to_csv(comparison_output_path, index=True)

    print(f"Saved final portfolio comparison to: {comparison_output_path}")

    return portfolio_comparison, tax_table, weights_table



# ============================================================
# Statistical-only equal-weighted portfolio with portfolio-level tax
# ============================================================

def build_statistical_equal_weight_portfolio_with_portfolio_level_tax(
    config: ProjectConfig,
    statistical_individual_comparisons: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds a statistical-only equal-weighted portfolio with portfolio-level tax.

    This function is intentionally separate from the final portfolio builder.

    It does NOT change:
    - the final strategy portfolio;
    - the final 50/50 benchmark;
    - the final Ibovespa benchmark;
    - any existing final portfolio output file.

    It creates an additional diagnostic benchmark:
    - statistical-only companies;
    - equal company weights at inception;
    - no rebalancing back to equal weights;
    - individual tax disabled before this function is called;
    - tax calculated once at the aggregated statistical portfolio level;
    - tax paid on the last available trading day of the following month;
    - after-tax curve applied recursively as a cash outflow.
    """

    if not statistical_individual_comparisons:
        raise ValueError("No statistical individual comparisons available.")

    available_companies = [
        company
        for company, comparison in statistical_individual_comparisons.items()
        if comparison is not None
        and not comparison.empty
        and "strategy_value" in comparison.columns
    ]

    if not available_companies:
        raise ValueError(
            "No statistical individual comparisons with strategy_value available."
        )

    equal_weight = 1.0 / len(available_companies)

    statistical_weights = {
        company: equal_weight
        for company in available_companies
    }

    weights_table = pd.DataFrame(
        [
            {
                "company": company,
                "initial_portfolio_weight": weight,
            }
            for company, weight in statistical_weights.items()
        ]
    )

    weights_output_path = (
        config.paths.tables_dir
        / "statistical_equal_weight_initial_weights.csv"
    )

    weights_table.to_csv(weights_output_path, index=False)

    print(
        "\nSaved statistical equal-weight initial weights to: "
        f"{weights_output_path}"
    )

    common_index = build_fixed_portfolio_test_index(
        individual_comparisons=statistical_individual_comparisons,
        start_date=config.backtest.test_start_date,
        end_date=config.backtest.end_date,
    )

    # ------------------------------------------------------------
    # 1. Statistical-only strategy portfolio before portfolio tax.
    # ------------------------------------------------------------

    statistical_pre_tax = build_initial_weight_buy_and_hold_curve(
        individual_comparisons=statistical_individual_comparisons,
        weights=statistical_weights,
        value_column="strategy_value",
        output_name="statistical_equal_weight_pre_tax",
        common_index=common_index,
    )

    # ------------------------------------------------------------
    # 2. Statistical-only passive 50/50 benchmark.
    # ------------------------------------------------------------

    statistical_50_50 = build_initial_weight_buy_and_hold_curve(
        individual_comparisons=statistical_individual_comparisons,
        weights=statistical_weights,
        value_column="benchmark_50_50_value",
        output_name="statistical_equal_weight_50_50",
        common_index=common_index,
    )

    # ------------------------------------------------------------
    # 3. Ibovespa curve for the same statistical-only dates.
    # ------------------------------------------------------------

    statistical_ibovespa = build_initial_weight_buy_and_hold_curve(
        individual_comparisons=statistical_individual_comparisons,
        weights=statistical_weights,
        value_column="ibovespa_value",
        output_name="statistical_equal_weight_ibovespa",
        common_index=common_index,
    )

    # ------------------------------------------------------------
    # 4. Weighted daily tax inputs for statistical-only portfolio.
    # ------------------------------------------------------------

    daily_tax_inputs = build_weighted_daily_tax_inputs(
        individual_comparisons=statistical_individual_comparisons,
        weights=statistical_weights,
        common_index=common_index,
    )

    daily_tax_inputs_output_path = (
        config.paths.tables_dir
        / "statistical_equal_weight_daily_tax_inputs.csv"
    )

    daily_tax_inputs.to_csv(daily_tax_inputs_output_path, index=True)

    print(
        "Saved statistical equal-weight daily tax inputs to: "
        f"{daily_tax_inputs_output_path}"
    )

    # ------------------------------------------------------------
    # 5. Monthly tax at portfolio level.
    # ------------------------------------------------------------

    tax_table = calculate_portfolio_level_monthly_tax_with_payment_lag(
        daily_tax_inputs=daily_tax_inputs,
        tax_rate=config.backtest.original_income_tax_rate,
        use_loss_carryforward=config.backtest.use_loss_carryforward,
    )

    tax_output_path = (
        config.paths.tables_dir
        / "statistical_equal_weight_monthly_tax_records.csv"
    )

    tax_table.to_csv(tax_output_path, index=False)

    print(
        "Saved statistical equal-weight monthly tax records to: "
        f"{tax_output_path}"
    )

    # ------------------------------------------------------------
    # 6. Combine curves and apply tax recursively.
    # ------------------------------------------------------------

    statistical_comparison = pd.concat(
        [
            statistical_pre_tax,
            statistical_50_50,
            statistical_ibovespa,
            daily_tax_inputs.add_prefix("statistical_daily_"),
        ],
        axis=1,
        join="inner",
    ).sort_index()

    if statistical_comparison.empty:
        raise ValueError("Statistical equal-weight comparison is empty.")

    statistical_comparison["portfolio_tax_paid"] = 0.0

    if tax_table is not None and not tax_table.empty:
        for _, tax_row in tax_table.iterrows():
            payment_date = tax_row["tax_payment_date"]

            if pd.isna(payment_date):
                continue

            payment_date = pd.Timestamp(payment_date)

            if payment_date in statistical_comparison.index:
                statistical_comparison.loc[
                    payment_date,
                    "portfolio_tax_paid",
                ] += float(tax_row["tax_due"])

    statistical_comparison["portfolio_cumulative_tax_paid"] = (
        statistical_comparison["portfolio_tax_paid"].cumsum()
    )

    # Reuse the same recursive after-tax logic by temporarily renaming columns.
    tax_application_input = statistical_comparison.rename(
        columns={
            "statistical_equal_weight_pre_tax_value": (
                "strategy_portfolio_pre_tax_value"
            ),
            "statistical_equal_weight_pre_tax_return": (
                "strategy_portfolio_pre_tax_return"
            ),
        }
    ).copy()

    tax_application_output = apply_portfolio_tax_recursively(
        portfolio_comparison=tax_application_input,
    )

    statistical_comparison["statistical_equal_weight_value"] = (
        tax_application_output["strategy_portfolio_value"]
    )

    statistical_comparison["statistical_equal_weight_return"] = (
        tax_application_output["strategy_portfolio_return"]
    )

    statistical_comparison["statistical_equal_weight_cumulative_return"] = (
        tax_application_output["strategy_portfolio_cumulative_return"]
    )

    statistical_comparison["statistical_equal_weight_minus_50_50"] = (
        statistical_comparison["statistical_equal_weight_cumulative_return"]
        - statistical_comparison[
            "statistical_equal_weight_50_50_cumulative_return"
        ]
    )

    statistical_comparison["statistical_equal_weight_minus_ibovespa"] = (
        statistical_comparison["statistical_equal_weight_cumulative_return"]
        - statistical_comparison[
            "statistical_equal_weight_ibovespa_cumulative_return"
        ]
    )

    comparison_output_path = (
        config.paths.tables_dir
        / "statistical_equal_weight_comparison_portfolio_level_tax.csv"
    )

    statistical_comparison.to_csv(comparison_output_path, index=True)

    print(
        "Saved statistical equal-weight comparison with portfolio-level tax to: "
        f"{comparison_output_path}"
    )

    return statistical_comparison, tax_table, weights_table


def add_statistical_equal_weight_benchmark_to_final_portfolio_plots(
    config: ProjectConfig,
    portfolio_comparison: pd.DataFrame,
    statistical_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds the statistical-only equal-weighted benchmark to the final portfolio
    comparison used by the plot builder.

    This only adds extra columns for plotting and comparison.
    It does not alter the existing strategy, 50/50 benchmark, Ibovespa curve,
    tax records, or metrics.
    """

    if portfolio_comparison is None or portfolio_comparison.empty:
        raise ValueError("portfolio_comparison is empty.")

    if statistical_comparison is None or statistical_comparison.empty:
        print(
            "\nSkipping statistical-only benchmark in final plots: "
            "statistical comparison is empty."
        )
        return portfolio_comparison

    required_columns = [
        "statistical_equal_weight_value",
        "statistical_equal_weight_return",
        "statistical_equal_weight_cumulative_return",
    ]

    for column in required_columns:
        if column not in statistical_comparison.columns:
            raise ValueError(
                f"Missing column in statistical comparison: {column}"
            )

    aligned = statistical_comparison[required_columns].copy()
    aligned = aligned.reindex(portfolio_comparison.index)
    aligned = aligned.ffill()

    if aligned["statistical_equal_weight_value"].isna().any():
        aligned["statistical_equal_weight_value"] = (
            aligned["statistical_equal_weight_value"].fillna(1.0)
        )

    if aligned["statistical_equal_weight_cumulative_return"].isna().any():
        aligned["statistical_equal_weight_cumulative_return"] = (
            aligned["statistical_equal_weight_cumulative_return"].fillna(0.0)
        )

    if aligned["statistical_equal_weight_return"].isna().any():
        aligned["statistical_equal_weight_return"] = (
            aligned["statistical_equal_weight_return"].fillna(0.0)
        )

    portfolio_comparison = portfolio_comparison.copy()

    portfolio_comparison["statistical_equal_weight_portfolio_value"] = (
        aligned["statistical_equal_weight_value"]
    )

    portfolio_comparison["statistical_equal_weight_portfolio_return"] = (
        aligned["statistical_equal_weight_return"]
    )

    portfolio_comparison["statistical_equal_weight_portfolio_cumulative_return"] = (
        aligned["statistical_equal_weight_cumulative_return"]
    )

    portfolio_comparison["strategy_minus_statistical_equal_weight"] = (
        portfolio_comparison["strategy_portfolio_cumulative_return"]
        - portfolio_comparison[
            "statistical_equal_weight_portfolio_cumulative_return"
        ]
    )

    extended_output_path = (
        config.paths.tables_dir
        / "final_portfolio_comparison_extended_with_statistical_benchmark.csv"
    )

    portfolio_comparison.to_csv(extended_output_path, index=True)

    print(
        "\nAdded statistical-only equal-weighted after-tax benchmark "
        "to final portfolio comparison."
    )
    print(f"Saved extended final comparison to: {extended_output_path}")

    return portfolio_comparison


# ============================================================
# Final portfolio metrics
# ============================================================

def calculate_final_portfolio_metrics(
    config: ProjectConfig,
    portfolio_comparison: pd.DataFrame,
    tax_table: pd.DataFrame,
    weights_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates final portfolio-level metrics.
    """

    metrics_calculator = PerformanceMetrics(
        trading_days_per_year=config.backtest.trading_days_per_year,
    )

    strategy_metrics = metrics_calculator.calculate_from_equity_curve(
        equity_curve=portfolio_comparison["strategy_portfolio_value"],
        label="strategy_portfolio",
    )

    strategy_pre_tax_metrics = metrics_calculator.calculate_from_equity_curve(
        equity_curve=portfolio_comparison["strategy_portfolio_pre_tax_value"],
        label="strategy_portfolio_pre_tax",
    )

    benchmark_50_50_metrics = metrics_calculator.calculate_from_equity_curve(
        equity_curve=portfolio_comparison["benchmark_50_50_portfolio_value"],
        label="benchmark_50_50_portfolio",
    )

    ibovespa_metrics = metrics_calculator.calculate_from_equity_curve(
        equity_curve=portfolio_comparison["ibovespa_portfolio_value"],
        label="ibovespa_portfolio",
    )

    final_row = portfolio_comparison.iloc[-1]

    metrics = {
        **strategy_metrics,
        **strategy_pre_tax_metrics,
        **benchmark_50_50_metrics,
        **ibovespa_metrics,

        "strategy_portfolio_final_value": final_row[
            "strategy_portfolio_value"
        ],
        "strategy_portfolio_pre_tax_final_value": final_row[
            "strategy_portfolio_pre_tax_value"
        ],
        "benchmark_50_50_portfolio_final_value": final_row[
            "benchmark_50_50_portfolio_value"
        ],
        "ibovespa_portfolio_final_value": final_row[
            "ibovespa_portfolio_value"
        ],

        "strategy_portfolio_final_return": final_row[
            "strategy_portfolio_cumulative_return"
        ],
        "strategy_portfolio_pre_tax_final_return": final_row[
            "strategy_portfolio_pre_tax_cumulative_return"
        ],
        "benchmark_50_50_portfolio_final_return": final_row[
            "benchmark_50_50_portfolio_cumulative_return"
        ],
        "ibovespa_portfolio_final_return": final_row[
            "ibovespa_portfolio_cumulative_return"
        ],

        "strategy_excess_return_vs_50_50": final_row[
            "strategy_minus_50_50"
        ],
        "strategy_excess_return_vs_ibovespa": final_row[
            "strategy_minus_ibovespa"
        ],
        "strategy_pre_tax_excess_return_vs_50_50": final_row[
            "strategy_pre_tax_minus_50_50"
        ],
        "strategy_pre_tax_excess_return_vs_ibovespa": final_row[
            "strategy_pre_tax_minus_ibovespa"
        ],

        "total_portfolio_tax_paid": float(
            portfolio_comparison["portfolio_tax_paid"].sum()
        ),
        "total_portfolio_transaction_cost": float(
            portfolio_comparison["portfolio_daily_transaction_cost"].sum()
        ),
        "total_portfolio_realized_pnl": float(
            portfolio_comparison["portfolio_daily_realized_pnl"].sum()
        ),
        "total_portfolio_sales_value": float(
            portfolio_comparison["portfolio_daily_gross_sale_value"].sum()
        ),

        "portfolio_start_date": portfolio_comparison.index.min(),
        "portfolio_end_date": portfolio_comparison.index.max(),
        "portfolio_observations": len(portfolio_comparison),
        "number_of_companies": len(weights_table),
    }

    if tax_table is not None and not tax_table.empty:
        metrics["total_tax_due"] = float(tax_table["tax_due"].sum())
        metrics["total_taxable_profit"] = float(tax_table["taxable_profit"].sum())
        metrics["total_loss_used"] = float(tax_table["loss_used"].sum())
        metrics["final_accumulated_portfolio_loss"] = float(
            tax_table["accumulated_loss_after"].iloc[-1]
        )
    else:
        metrics["total_tax_due"] = 0.0
        metrics["total_taxable_profit"] = 0.0
        metrics["total_loss_used"] = 0.0
        metrics["final_accumulated_portfolio_loss"] = 0.0

    metrics_table = pd.DataFrame([metrics])

    metrics_output_path = (
        config.paths.tables_dir
        / "final_portfolio_metrics_portfolio_level_tax.csv"
    )

    metrics_table.to_csv(metrics_output_path, index=False)

    print(f"Saved final portfolio metrics to: {metrics_output_path}")

    return metrics_table


# ============================================================
# Plots and summary
# ============================================================

def build_final_portfolio_plots(
    config: ProjectConfig,
    portfolio_comparison: pd.DataFrame,
):
    """
    Builds final portfolio plots.

    Final plot:
    - final strategy portfolio after portfolio-level tax;
    - final 50/50 benchmark;
    - Ibovespa benchmark;
    - statistical equal-weight benchmark.

    Important:
    This function only creates the final comparison plot.
    It does not change the backtest, tax logic, metrics,
    portfolio weights, or benchmark calculations.
    """

    import matplotlib.pyplot as plt

    if portfolio_comparison is None or portfolio_comparison.empty:
        print("\nNo final portfolio comparison available for plotting.")
        return []

    config.paths.plots_dir.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "strategy_portfolio_cumulative_return": "Final Portfolio Strategy",
        "benchmark_50_50_portfolio_cumulative_return": "Final 50/50 Benchmark",
        "ibovespa_portfolio_cumulative_return": "Ibovespa Benchmark",
        "statistical_equal_weight_portfolio_cumulative_return": "Statistical Equal-Weight Benchmark",
    }

    for column in required_columns:
        if column not in portfolio_comparison.columns:
            raise ValueError(
                f"Missing required column for final portfolio plot: {column}"
            )

    plot_data = portfolio_comparison[list(required_columns.keys())].copy()
    plot_data = plot_data.apply(pd.to_numeric, errors="coerce")
    plot_data = plot_data.replace([float("inf"), float("-inf")], pd.NA)
    plot_data = plot_data.ffill().dropna(how="all")

    if plot_data.empty:
        raise ValueError("Final portfolio plot data is empty after cleaning.")

    # Convert cumulative returns from decimal format to percentage points.
    plot_data = plot_data * 100.0

    plt.figure(figsize=(14, 8))

    for column, label in required_columns.items():
        plt.plot(
            plot_data.index,
            plot_data[column],
            linewidth=2.0,
            label=label,
        )

    plt.axhline(
        y=0.0,
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.title(
        "Final Portfolio vs. Benchmarks",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative return (%)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    png_output_path = (
        config.paths.plots_dir
        / "final_portfolio_vs_benchmarks.png"
    )

    pdf_output_path = (
        config.paths.plots_dir
        / "final_portfolio_vs_benchmarks.pdf"
    )

    plt.savefig(png_output_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_output_path, bbox_inches="tight")
    plt.close()

    saved_paths = [
        png_output_path,
        pdf_output_path,
    ]

    print("\nFinal portfolio comparison plot completed.")
    print(f"Saved final comparison PNG to: {png_output_path}")
    print(f"Saved final comparison PDF to: {pdf_output_path}")
    print(f"Plots folder: {config.paths.plots_dir}")

    return saved_paths


def print_final_portfolio_summary(metrics_table: pd.DataFrame):
    """
    Prints final portfolio summary.
    """

    if metrics_table is None or metrics_table.empty:
        print("\nNo final portfolio metrics available.")
        return

    row = metrics_table.iloc[0]

    print("\nFinal weighted portfolio summary")
    print("=" * 120)

    print(f"Number of companies: {int(row['number_of_companies'])}")

    print(
        f"Portfolio start date: "
        f"{pd.Timestamp(row['portfolio_start_date']).date()}"
    )

    print(
        f"Portfolio end date: "
        f"{pd.Timestamp(row['portfolio_end_date']).date()}"
    )

    print(
        f"Portfolio observations: "
        f"{int(row['portfolio_observations'])}"
    )

    print(
        f"Strategy return after portfolio-level tax: "
        f"{row['strategy_portfolio_total_return']:.2%}"
    )

    print(
        f"Strategy return before tax: "
        f"{row['strategy_portfolio_pre_tax_total_return']:.2%}"
    )

    print(
        f"Fundamental-weighted 50/50 benchmark return: "
        f"{row['benchmark_50_50_portfolio_total_return']:.2%}"
    )

    print(
        f"Ibovespa return: "
        f"{row['ibovespa_portfolio_total_return']:.2%}"
    )

    print(
        f"Excess return vs fundamental-weighted 50/50 after tax: "
        f"{row['strategy_excess_return_vs_50_50']:.2%}"
    )

    print(
        f"Excess return vs Ibovespa after tax: "
        f"{row['strategy_excess_return_vs_ibovespa']:.2%}"
    )

    print(
        f"Strategy Sharpe after tax: "
        f"{row['strategy_portfolio_sharpe_ratio']:.4f}"
    )

    print(
        f"Strategy Sharpe before tax: "
        f"{row['strategy_portfolio_pre_tax_sharpe_ratio']:.4f}"
    )

    print(
        f"Fundamental-weighted 50/50 Sharpe: "
        f"{row['benchmark_50_50_portfolio_sharpe_ratio']:.4f}"
    )

    print(
        f"Ibovespa Sharpe: "
        f"{row['ibovespa_portfolio_sharpe_ratio']:.4f}"
    )

    print(
        f"Strategy max drawdown after tax: "
        f"{row['strategy_portfolio_max_drawdown']:.2%}"
    )

    print(
        f"Total portfolio tax paid: "
        f"{row['total_portfolio_tax_paid']:.6f}"
    )

    print(
        f"Total portfolio transaction cost: "
        f"{row['total_portfolio_transaction_cost']:.6f}"
    )

    print(
        f"Total taxable profit: "
        f"{row['total_taxable_profit']:.6f}"
    )

    print(
        f"Total loss used: "
        f"{row['total_loss_used']:.6f}"
    )

    print(
        f"Final accumulated portfolio loss: "
        f"{row['final_accumulated_portfolio_loss']:.6f}"
    )




# ============================================================
# Robustness analysis helpers
# ============================================================

def make_robustness_scenario_name(
    scenario_family: str,
    scenario_value,
) -> str:
    """
    Builds safe folder/file names for robustness scenarios.
    """

    if isinstance(scenario_value, float):
        value_text = f"{scenario_value:.4f}".rstrip("0").rstrip(".")
    else:
        value_text = str(scenario_value)

    value_text = value_text.replace(".", "_").replace("%", "pct")

    return f"{scenario_family}_{value_text}"


def configure_robustness_output_paths(
    config: ProjectConfig,
    scenario_name: str,
) -> None:
    """
    Sends robustness outputs to a scenario-specific folder.

    This does not alter the main pipeline outputs. The function is called only
    inside the optional robustness section, using a copied configuration.
    """

    robustness_root = config.paths.project_dir / "final_results" / "robustness"
    scenario_dir = robustness_root / scenario_name

    config.paths.results_dir = scenario_dir
    config.paths.individual_results_dir = scenario_dir / "individual"
    config.paths.tables_dir = scenario_dir / "tables"
    config.paths.plots_dir = scenario_dir / "plots"

    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    config.paths.individual_results_dir.mkdir(parents=True, exist_ok=True)
    config.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    config.paths.plots_dir.mkdir(parents=True, exist_ok=True)


def override_policy_variables_for_robustness(
    policy_map: dict,
    entry_threshold: float | None = None,
    signal_window: int | None = None,
) -> dict:
    """
    Creates a copied policy map for one robustness scenario.

    The company selection, policy group, min/max ON bands, exit threshold and
    tax-loss-harvesting setting remain unchanged. Only the requested variable
    is overwritten for the robustness test.
    """

    adjusted_policy_map = {}

    for company, policy in policy_map.items():
        overrides = {}

        if entry_threshold is not None:
            overrides["entry_threshold"] = float(entry_threshold)

        if signal_window is not None:
            overrides["signal_window"] = int(signal_window)

        if overrides:
            adjusted_policy_map[company] = replace(policy, **overrides)
        else:
            adjusted_policy_map[company] = policy

    return adjusted_policy_map


def safe_metric_value(
    metrics_row: pd.Series,
    column: str,
    default: float = 0.0,
) -> float:
    """
    Reads one metric from the final metrics row without failing if a column is
    unavailable.
    """

    if column not in metrics_row.index:
        return default

    value = metrics_row[column]

    if pd.isna(value):
        return default

    return float(value)


def build_robustness_summary_row(
    scenario_family: str,
    scenario_name: str,
    scenario_value,
    metrics_table: pd.DataFrame,
    individual_metrics_table: pd.DataFrame,
) -> dict:
    """
    Extracts the main report-ready robustness metrics from one scenario.
    """

    if metrics_table is None or metrics_table.empty:
        raise ValueError(f"No final metrics available for {scenario_name}.")

    row = metrics_table.iloc[0]

    total_trade_days = 0
    if (
        individual_metrics_table is not None
        and not individual_metrics_table.empty
        and "number_of_trade_days" in individual_metrics_table.columns
    ):
        total_trade_days = int(
            pd.to_numeric(
                individual_metrics_table["number_of_trade_days"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    return {
        "scenario_family": scenario_family,
        "scenario": scenario_name,
        "scenario_value": scenario_value,

        "strategy_total_return_after_tax": safe_metric_value(
            row,
            "strategy_portfolio_total_return",
        ),
        "strategy_total_return_pre_tax": safe_metric_value(
            row,
            "strategy_portfolio_pre_tax_total_return",
        ),
        "benchmark_50_50_total_return": safe_metric_value(
            row,
            "benchmark_50_50_portfolio_total_return",
        ),
        "ibovespa_total_return": safe_metric_value(
            row,
            "ibovespa_portfolio_total_return",
        ),

        "strategy_annualized_return_after_tax": safe_metric_value(
            row,
            "strategy_portfolio_annualized_return",
        ),
        "strategy_volatility_after_tax": safe_metric_value(
            row,
            "strategy_portfolio_volatility",
        ),
        "strategy_sharpe_after_tax": safe_metric_value(
            row,
            "strategy_portfolio_sharpe_ratio",
        ),
        "strategy_max_drawdown_after_tax": safe_metric_value(
            row,
            "strategy_portfolio_max_drawdown",
        ),

        "excess_return_vs_50_50_after_tax": safe_metric_value(
            row,
            "strategy_excess_return_vs_50_50",
        ),
        "excess_return_vs_ibovespa_after_tax": safe_metric_value(
            row,
            "strategy_excess_return_vs_ibovespa",
        ),

        "total_tax_paid": safe_metric_value(
            row,
            "total_portfolio_tax_paid",
        ),
        "total_transaction_cost": safe_metric_value(
            row,
            "total_portfolio_transaction_cost",
        ),
        "total_sales_value_turnover_proxy": safe_metric_value(
            row,
            "total_portfolio_sales_value",
        ),
        "total_trade_days": total_trade_days,

        "portfolio_start_date": row.get("portfolio_start_date"),
        "portfolio_end_date": row.get("portfolio_end_date"),
        "number_of_companies": int(
            safe_metric_value(row, "number_of_companies")
        ),
    }


def run_single_robustness_scenario(
    base_config: ProjectConfig,
    scenario_family: str,
    scenario_name: str,
    scenario_value,
    selected_pairs: list,
    signal_data_by_company: dict,
    execution_data_by_company: dict,
    base_policy_map: dict,
    final_portfolio_weights: dict,
    base_transaction_cost_rate: float,
    base_tax_rate: float,
    entry_threshold: float | None = None,
    signal_window: int | None = None,
    transaction_cost_rate: float | None = None,
    tax_rate: float | None = None,
) -> dict:
    """
    Runs one robustness scenario by reusing the same final-pipeline functions.

    Only the scenario variables are changed. The original baseline main logic
    and all strategy/backtest/tax functions remain unchanged.
    """

    scenario_config = copy.deepcopy(base_config)

    configure_robustness_output_paths(
        config=scenario_config,
        scenario_name=scenario_name,
    )

    if transaction_cost_rate is None:
        transaction_cost_rate = base_transaction_cost_rate

    if tax_rate is None:
        tax_rate = base_tax_rate

    # Individual tax stays disabled, exactly as in the final portfolio input.
    # Portfolio-level tax is applied later by build_final_portfolio_with_portfolio_level_tax.
    scenario_config.backtest.income_tax_rate = 0.0
    scenario_config.backtest.original_income_tax_rate = float(tax_rate)
    scenario_config.backtest.transaction_cost_rate = float(transaction_cost_rate)

    # Short rolling-window robustness needs a compatible minimum observation
    # count. This only affects the scenario copy.
    if signal_window is not None:
        scenario_config.signals.minimum_signal_observations = min(
            int(scenario_config.signals.minimum_signal_observations),
            int(signal_window),
        )

    scenario_policy_map = override_policy_variables_for_robustness(
        policy_map=base_policy_map,
        entry_threshold=entry_threshold,
        signal_window=signal_window,
    )

    print("\n" + "=" * 120)
    print(f"ROBUSTNESS — {scenario_name}")
    print(f"Family: {scenario_family} | Value: {scenario_value}")
    print("=" * 120)

    individual_comparisons, individual_metrics_table = run_individual_backtests_generic(
        config=scenario_config,
        selected_pairs=selected_pairs,
        signal_data_by_company=signal_data_by_company,
        execution_data_by_company=execution_data_by_company,
        policy_map=scenario_policy_map,
        file_prefix=f"robustness_{scenario_name}",
        metrics_filename=f"robustness_{scenario_name}_individual_metrics.csv",
    )

    if not individual_comparisons:
        raise ValueError(
            f"No individual comparisons were created for {scenario_name}."
        )

    portfolio_comparison, tax_table, weights_table = (
        build_final_portfolio_with_portfolio_level_tax(
            config=scenario_config,
            individual_comparisons=individual_comparisons,
            weights=final_portfolio_weights,
        )
    )

    metrics_table = calculate_final_portfolio_metrics(
        config=scenario_config,
        portfolio_comparison=portfolio_comparison,
        tax_table=tax_table,
        weights_table=weights_table,
    )

    summary_row = build_robustness_summary_row(
        scenario_family=scenario_family,
        scenario_name=scenario_name,
        scenario_value=scenario_value,
        metrics_table=metrics_table,
        individual_metrics_table=individual_metrics_table,
    )

    return summary_row


def run_robustness_analysis_from_final_environment(
    config: ProjectConfig,
    final_selected_pairs: list,
    final_signal_history_data_by_company: dict,
    final_test_data_by_company: dict,
    final_policy_map: dict,
    final_portfolio_weights: dict,
    original_income_tax_rate: float,
) -> pd.DataFrame:
    """
    Optional robustness section appended to the final main pipeline.

    It reuses the already prepared final environment:
    - same manual/fundamental final company universe;
    - same train/test split;
    - same policy map;
    - same final portfolio weights;
    - same portfolio-level tax function.

    Only one scenario variable is changed at a time.
    """

    robustness_root = config.paths.project_dir / "final_results" / "robustness"
    robustness_root.mkdir(parents=True, exist_ok=True)

    base_transaction_cost_rate = float(config.backtest.transaction_cost_rate)
    base_tax_rate = float(original_income_tax_rate)

    summary_rows = []

    # ------------------------------------------------------------
    # Baseline reference row.
    # ------------------------------------------------------------

    summary_rows.append(
        run_single_robustness_scenario(
            base_config=config,
            scenario_family="baseline",
            scenario_name="baseline",
            scenario_value="baseline",
            selected_pairs=final_selected_pairs,
            signal_data_by_company=final_signal_history_data_by_company,
            execution_data_by_company=final_test_data_by_company,
            base_policy_map=final_policy_map,
            final_portfolio_weights=final_portfolio_weights,
            base_transaction_cost_rate=base_transaction_cost_rate,
            base_tax_rate=base_tax_rate,
        )
    )

    # ------------------------------------------------------------
    # Entry-threshold robustness.
    # ------------------------------------------------------------

    for threshold in ROBUSTNESS_ENTRY_THRESHOLDS:
        scenario_name = make_robustness_scenario_name(
            "entry_threshold",
            threshold,
        )

        summary_rows.append(
            run_single_robustness_scenario(
                base_config=config,
                scenario_family="entry_threshold",
                scenario_name=scenario_name,
                scenario_value=threshold,
                selected_pairs=final_selected_pairs,
                signal_data_by_company=final_signal_history_data_by_company,
                execution_data_by_company=final_test_data_by_company,
                base_policy_map=final_policy_map,
                final_portfolio_weights=final_portfolio_weights,
                base_transaction_cost_rate=base_transaction_cost_rate,
                base_tax_rate=base_tax_rate,
                entry_threshold=threshold,
            )
        )

    # ------------------------------------------------------------
    # Rolling-window robustness.
    # ------------------------------------------------------------

    for window in ROBUSTNESS_ROLLING_WINDOWS:
        scenario_name = make_robustness_scenario_name(
            "rolling_window",
            window,
        )

        summary_rows.append(
            run_single_robustness_scenario(
                base_config=config,
                scenario_family="rolling_window",
                scenario_name=scenario_name,
                scenario_value=window,
                selected_pairs=final_selected_pairs,
                signal_data_by_company=final_signal_history_data_by_company,
                execution_data_by_company=final_test_data_by_company,
                base_policy_map=final_policy_map,
                final_portfolio_weights=final_portfolio_weights,
                base_transaction_cost_rate=base_transaction_cost_rate,
                base_tax_rate=base_tax_rate,
                signal_window=window,
            )
        )

    # ------------------------------------------------------------
    # Transaction-cost robustness.
    # ------------------------------------------------------------

    for cost in ROBUSTNESS_TRANSACTION_COSTS:
        scenario_name = make_robustness_scenario_name(
            "transaction_cost",
            cost,
        )

        summary_rows.append(
            run_single_robustness_scenario(
                base_config=config,
                scenario_family="transaction_cost",
                scenario_name=scenario_name,
                scenario_value=cost,
                selected_pairs=final_selected_pairs,
                signal_data_by_company=final_signal_history_data_by_company,
                execution_data_by_company=final_test_data_by_company,
                base_policy_map=final_policy_map,
                final_portfolio_weights=final_portfolio_weights,
                base_transaction_cost_rate=base_transaction_cost_rate,
                base_tax_rate=base_tax_rate,
                transaction_cost_rate=cost,
            )
        )

    # ------------------------------------------------------------
    # Tax-treatment robustness.
    # ------------------------------------------------------------

    for tax_rate in ROBUSTNESS_TAX_RATES:
        if float(tax_rate) == 0.0:
            scenario_name = "tax_pre_tax"
        else:
            scenario_name = "tax_after_tax_baseline"

        summary_rows.append(
            run_single_robustness_scenario(
                base_config=config,
                scenario_family="tax_treatment",
                scenario_name=scenario_name,
                scenario_value=tax_rate,
                selected_pairs=final_selected_pairs,
                signal_data_by_company=final_signal_history_data_by_company,
                execution_data_by_company=final_test_data_by_company,
                base_policy_map=final_policy_map,
                final_portfolio_weights=final_portfolio_weights,
                base_transaction_cost_rate=base_transaction_cost_rate,
                base_tax_rate=base_tax_rate,
                tax_rate=tax_rate,
            )
        )

    summary_table = pd.DataFrame(summary_rows)
    summary_table = summary_table.sort_values(
        ["scenario_family", "scenario"]
    ).reset_index(drop=True)

    full_output_path = robustness_root / "robustness_summary_all_scenarios.csv"
    summary_table.to_csv(full_output_path, index=False)

    report_columns = [
        "scenario_family",
        "scenario",
        "scenario_value",
        "strategy_total_return_after_tax",
        "strategy_total_return_pre_tax",
        "benchmark_50_50_total_return",
        "ibovespa_total_return",
        "strategy_sharpe_after_tax",
        "strategy_max_drawdown_after_tax",
        "excess_return_vs_50_50_after_tax",
        "total_tax_paid",
        "total_transaction_cost",
        "total_sales_value_turnover_proxy",
        "total_trade_days",
    ]

    compact_table = summary_table[report_columns].copy()

    compact_output_path = robustness_root / "robustness_report_table.csv"
    compact_table.to_csv(compact_output_path, index=False)

    for scenario_family, family_table in compact_table.groupby("scenario_family"):
        family_output_path = robustness_root / f"robustness_{scenario_family}.csv"
        family_table.to_csv(family_output_path, index=False)

    print("\nRobustness analysis completed.")
    print(f"Saved full robustness table to: {full_output_path}")
    print(f"Saved report robustness table to: {compact_output_path}")
    print(f"Robustness folder: {robustness_root}")

    return summary_table



# ============================================================
# Unified final main
# ============================================================

def main():
    """
    Unified final pipeline.

    Produces, in one run:
    1. statistical-filtering individual results;
    2. final manual/fundamental-weighted individual results with individual
       taxes disabled;
    3. final weighted portfolio with global portfolio-level monthly taxes.

    This replaces the need to run main.py, main_test.py and main_final.py
    separately.
    """

    config = ProjectConfig()
    config.initialize_project()

    print("\nStarting unified final ON/PN project")
    print("=" * 120)

    # ------------------------------------------------------------
    # Global configuration.
    # ------------------------------------------------------------

    config.backtest.download_data = True
    original_income_tax_rate = config.backtest.income_tax_rate
    config.backtest.original_income_tax_rate = original_income_tax_rate
    original_company_pairs = dict(config.universe.company_pairs)

    print("\nBacktest configuration")
    print("-" * 120)
    print(f"Data start date: {config.backtest.start_date}")
    print(f"Data end date:   {config.backtest.end_date}")
    print(f"Test start date: {config.backtest.test_start_date}")

    # ============================================================
    # PART 1 — STATISTICAL-FILTERING INDIVIDUAL RESULTS
    # ============================================================

    print("\nPART 1 — Statistical-filtering individual results")
    print("=" * 120)

    config.universe.company_pairs = original_company_pairs
    config.backtest.income_tax_rate = original_income_tax_rate
    config.universe_filter.top_n_selected_companies = (
        config.universe_filter.top_n_selected_companies
    )

    statistical_pair_objects = build_pair_objects(config)

    if not statistical_pair_objects:
        raise ValueError("No valid pair objects were created for statistical filtering.")

    (
        statistical_train_data_by_company,
        statistical_test_data_by_company,
        statistical_split_summary,
    ) = build_train_test_split(
        config=config,
        pair_objects=statistical_pair_objects,
    )

    statistical_split_path = (
        config.paths.tables_dir
        / "statistical_filtering_train_test_split_summary.csv"
    )
    statistical_split_summary.to_csv(statistical_split_path, index=False)
    print(f"Saved statistical split summary to: {statistical_split_path}")

    statistical_selected_pairs, statistical_filter_report = (
        run_statistical_universe_filter(
            config=config,
            pair_objects=statistical_pair_objects,
            train_data_by_company=statistical_train_data_by_company,
        )
    )

    if not statistical_selected_pairs:
        raise ValueError("No pairs passed the statistical universe filter.")

    statistical_policy_map = build_company_policies(
        config=config,
        filter_report=statistical_filter_report,
        forced_companies=None,
        output_filename="statistical_filtering_company_policy_map.csv",
    )

    statistical_signal_history_data_by_company = build_signal_history_data_by_company(
        train_data_by_company=statistical_train_data_by_company,
        test_data_by_company=statistical_test_data_by_company,
    )

    statistical_individual_comparisons, statistical_metrics_table = (
        run_individual_backtests_generic(
            config=config,
            selected_pairs=statistical_selected_pairs,
            signal_data_by_company=statistical_signal_history_data_by_company,
            execution_data_by_company=statistical_test_data_by_company,
            policy_map=statistical_policy_map,
            file_prefix="statistical_filtering",
            metrics_filename="statistical_filtering_individual_strategy_vs_benchmarks.csv",
        )
    )

    print("\nStatistical-filtering individual summary")
    print_final_summary(statistical_metrics_table)

    # ============================================================
    # PART 1B — STATISTICAL-FILTERING PORTFOLIO-TAX INPUTS
    # ============================================================
    # This block does not replace PART 1.
    # It creates a separate statistical-only benchmark with:
    # - equal weights at inception;
    # - individual tax disabled;
    # - portfolio-level monthly tax applied later.
    # ============================================================

    print("\nPART 1B — Statistical-only equal-weighted benchmark inputs")
    print("=" * 120)

    config.backtest.income_tax_rate = 0.0

    statistical_portfolio_tax_input_comparisons, statistical_portfolio_tax_input_metrics = (
        run_individual_backtests_generic(
            config=config,
            selected_pairs=statistical_selected_pairs,
            signal_data_by_company=statistical_signal_history_data_by_company,
            execution_data_by_company=statistical_test_data_by_company,
            policy_map=statistical_policy_map,
            file_prefix="statistical_equal_weight_portfolio_tax_input",
            metrics_filename=(
                "statistical_equal_weight_portfolio_tax_input_"
                "individual_strategy_vs_benchmarks.csv"
            ),
        )
    )

    print("\nStatistical-only portfolio-tax input summary")
    print_final_summary(statistical_portfolio_tax_input_metrics)

    config.backtest.income_tax_rate = original_income_tax_rate

    # ============================================================
    # PART 2 — FINAL MANUAL/WEIGHTED PORTFOLIO INPUTS
    # ============================================================

    print("\nPART 2 — Final manual/fundamental-weighted company results")
    print("=" * 120)

    manual_company_pairs = build_manual_company_pairs()
    final_portfolio_weights = build_final_portfolio_weights()

    # Disable individual tax here. Taxes are calculated globally in PART 3.
    config.backtest.original_income_tax_rate = original_income_tax_rate
    config.backtest.income_tax_rate = 0.0
    config.universe.company_pairs = manual_company_pairs
    config.universe_filter.top_n_selected_companies = None

    print("\nFinal portfolio universe:")
    for company, tickers in manual_company_pairs.items():
        weight = final_portfolio_weights.get(company, 0.0)
        print(
            f"{company}: ON={tickers[0]} | PN={tickers[1]} | "
            f"initial portfolio weight={weight:.2%}"
        )

    final_pair_objects = build_pair_objects(config)

    if not final_pair_objects:
        raise ValueError("No valid final portfolio pair objects were created.")

    (
        final_train_data_by_company,
        final_test_data_by_company,
        final_split_summary,
    ) = build_train_test_split(
        config=config,
        pair_objects=final_pair_objects,
    )

    final_split_path = (
        config.paths.tables_dir
        / "final_portfolio_train_test_split_summary.csv"
    )
    final_split_summary.to_csv(final_split_path, index=False)
    print(f"Saved final portfolio split summary to: {final_split_path}")

    selected_pairs_from_final_filter, final_filter_report = run_universe_filter(
        config=config,
        pair_objects=final_pair_objects,
        train_data_by_company=final_train_data_by_company,
    )

    print("\nFinal manual universe — pairs passing hard filters:")
    if selected_pairs_from_final_filter:
        for pair in selected_pairs_from_final_filter:
            print(f"- {pair.company}")
    else:
        print("- None")

    final_selected_pairs = select_pairs_by_company(
        pair_objects=final_pair_objects,
        companies=final_portfolio_weights.keys(),
    )

    if not final_selected_pairs:
        raise ValueError("No final weighted pairs are available for testing.")

    print("\nCompanies forced into final portfolio backtest:")
    for pair in final_selected_pairs:
        print(f"- {pair.company}: {pair.on_ticker}/{pair.pn_ticker}")

    final_policy_map = build_company_policies(
        config=config,
        filter_report=final_filter_report,
        forced_companies=final_portfolio_weights.keys(),
        output_filename="final_portfolio_company_policy_map.csv",
    )

    final_selected_pairs = [
        pair
        for pair in final_selected_pairs
        if pair.company in final_policy_map
    ]

    if not final_selected_pairs:
        raise ValueError("No final selected pairs have available policies.")

    final_signal_history_data_by_company = build_signal_history_data_by_company(
        train_data_by_company=final_train_data_by_company,
        test_data_by_company=final_test_data_by_company,
    )

    print("\nFinal individual backtests use train + test for signal calculation.")
    print("Execution and performance are measured by the backtester from the test period onward.")
    print("Individual tax is disabled here to avoid double-counting.")

    final_individual_comparisons, final_individual_metrics_table = (
        run_individual_backtests_generic(
            config=config,
            selected_pairs=final_selected_pairs,
            signal_data_by_company=final_signal_history_data_by_company,
            execution_data_by_company=final_test_data_by_company,
            policy_map=final_policy_map,
            file_prefix="final_portfolio_input",
            metrics_filename="final_portfolio_input_individual_strategy_vs_benchmarks.csv",
        )
    )

    if not final_individual_comparisons:
        raise ValueError("No final individual comparison results were created.")

    build_individual_plots(
        config=config,
        individual_comparisons=final_individual_comparisons,
    )

    print("\nFinal portfolio input individual summary")
    print_final_summary(final_individual_metrics_table)

    # ============================================================
    # PART 3 — FINAL WEIGHTED PORTFOLIO WITH GLOBAL TAX
    # ============================================================

    print("\nPART 3 — Final weighted portfolio with portfolio-level tax")
    print("=" * 120)

    portfolio_comparison, tax_table, weights_table = (
        build_final_portfolio_with_portfolio_level_tax(
            config=config,
            individual_comparisons=final_individual_comparisons,
            weights=final_portfolio_weights,
        )
    )

    statistical_equal_weight_comparison, statistical_equal_weight_tax_table, statistical_equal_weight_weights_table = (
        build_statistical_equal_weight_portfolio_with_portfolio_level_tax(
            config=config,
            statistical_individual_comparisons=(
                statistical_portfolio_tax_input_comparisons
            ),
        )
    )

    portfolio_comparison = (
        add_statistical_equal_weight_benchmark_to_final_portfolio_plots(
            config=config,
            portfolio_comparison=portfolio_comparison,
            statistical_comparison=statistical_equal_weight_comparison,
        )
    )

    print("\nFinal portfolio date range:")
    print(f"Start: {portfolio_comparison.index.min().date()}")
    print(f"End:   {portfolio_comparison.index.max().date()}")
    print(f"Observations: {len(portfolio_comparison)}")

    final_portfolio_metrics_table = calculate_final_portfolio_metrics(
        config=config,
        portfolio_comparison=portfolio_comparison,
        tax_table=tax_table,
        weights_table=weights_table,
    )

    build_final_portfolio_plots(
        config=config,
        portfolio_comparison=portfolio_comparison,
    )

    print_final_portfolio_summary(final_portfolio_metrics_table)

    # ============================================================
    # PART 4 — ROBUSTNESS AND SENSITIVITY ANALYSIS
    # ============================================================
    # Optional extra section. The baseline pipeline above remains unchanged.
    # This block only reruns the already prepared final environment with
    # alternative scenario variables.

    if RUN_ROBUSTNESS_ANALYSIS:
        print("\nPART 4 — Robustness and sensitivity analysis")
        print("=" * 120)

        run_robustness_analysis_from_final_environment(
            config=config,
            final_selected_pairs=final_selected_pairs,
            final_signal_history_data_by_company=final_signal_history_data_by_company,
            final_test_data_by_company=final_test_data_by_company,
            final_policy_map=final_policy_map,
            final_portfolio_weights=final_portfolio_weights,
            original_income_tax_rate=original_income_tax_rate,
        )

    print("\nUnified final pipeline completed successfully.")
    print(f"Results folder: {config.paths.results_dir}")
    print(f"Tables folder: {config.paths.tables_dir}")
    print(f"Plots folder: {config.paths.plots_dir}")


if __name__ == "__main__":
    main()
