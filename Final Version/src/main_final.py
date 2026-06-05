import pandas as pd

from project_config import ProjectConfig

from main_test import (
    build_manual_company_pairs,
    build_pair_objects,
    build_train_test_split,
    run_universe_filter,
    build_company_policies,
    run_individual_backtests,
    build_individual_plots,
    print_final_summary,
)

from performance_metrics import PerformanceMetrics
from plot_builder import PlotBuilder


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
        start_date="2020-01-01",
        end_date="2025-12-31",
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

    PlotBuilder expects:
    - strategy_portfolio_cumulative_return
    - benchmark_50_50_portfolio_cumulative_return
    - ibovespa_portfolio_cumulative_return
    """

    plot_builder = PlotBuilder(
        plots_dir=config.paths.plots_dir,
    )

    saved_paths = plot_builder.build_portfolio_plots(
        portfolio_comparison=portfolio_comparison,
    )

    print("\nFinal portfolio plots completed.")
    print(f"Saved portfolio plots: {len(saved_paths)}")
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
# Main pipeline
# ============================================================

def main():
    """
    Runs the final ON/PN portfolio pipeline.

    Main differences from main_test.py:
    - the portfolio uses the final company-level weights;
    - weights are applied only initially;
    - there is no periodic rebalancing between companies;
    - individual tax is disabled;
    - tax is calculated once at portfolio level;
    - tax base is monthly realized PnL minus transaction costs;
    - tax is paid on the last available trading day of the following month;
    - tax is applied recursively as a real portfolio cash outflow;
    - there is no additional final tax deduction.
    """

    config = ProjectConfig()
    config.initialize_project()

    print("\nStarting final weighted ON/PN portfolio")
    print("=" * 120)

    # ------------------------------------------------------------
    # Important:
    # Disable individual tax to avoid double-counting.
    # The original tax rate is stored and used later at portfolio level.
    # ------------------------------------------------------------

    config.backtest.original_income_tax_rate = config.backtest.income_tax_rate
    config.backtest.income_tax_rate = 0.0

    # If you already have all CSV files in data/raw, you can set this to False.
    # If some tickers such as BPAC3, BPAC5 or ISAE are missing, keep True.
    config.backtest.download_data = True

    # ------------------------------------------------------------
    # 0. Manual final universe and final portfolio weights.
    # ------------------------------------------------------------

    manual_company_pairs = build_manual_company_pairs()
    final_portfolio_weights = build_final_portfolio_weights()

    config.universe.company_pairs = manual_company_pairs
    config.universe_filter.top_n_selected_companies = None

    print("\nFinal portfolio universe:")
    for company, tickers in manual_company_pairs.items():
        weight = final_portfolio_weights.get(company, 0.0)

        print(
            f"{company}: ON={tickers[0]} | PN={tickers[1]} | "
            f"initial portfolio weight={weight:.2%}"
        )

    # ------------------------------------------------------------
    # 1. Load data and build pair objects.
    # ------------------------------------------------------------

    pair_objects = build_pair_objects(config)

    if not pair_objects:
        raise ValueError("No valid pair objects were created.")

    print("\nLoaded companies:")
    for pair in pair_objects:
        print(f"- {pair.company}: {pair.on_ticker}/{pair.pn_ticker}")

    # ------------------------------------------------------------
    # 2. Train-test split.
    # ------------------------------------------------------------

    train_data_by_company, test_data_by_company, split_summary = (
        build_train_test_split(
            config=config,
            pair_objects=pair_objects,
        )
    )

    split_summary_path = (
        config.paths.tables_dir
        / "final_portfolio_train_test_split_summary.csv"
    )

    split_summary.to_csv(split_summary_path, index=False)

    print(f"\nSaved final portfolio train-test split summary to: {split_summary_path}")

    print("\nFixed train-test split:")
    print("Train: from available start date, usually 2010-01-06, to 2019-12-31")
    print("Test:  from first 2020 trading day, usually 2020-01-02, to 2025-12-31")
    print("Signals in the test period can use rolling history from the training period.")

    # ------------------------------------------------------------
    # 3. Universe filter.
    # ------------------------------------------------------------
    # In the final manual portfolio, the filter is mainly used to calculate
    # training statistics and assign policies. Companies from the final
    # allocation table are forced into the portfolio afterwards.
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
    # 4. Force final weighted companies into the portfolio test.
    # ------------------------------------------------------------

    selected_pairs = [
        pair
        for pair in pair_objects
        if pair.company in final_portfolio_weights
    ]

    if not selected_pairs:
        raise ValueError("No final portfolio pairs are available for testing.")

    print("\nCompanies forced into final portfolio backtest:")
    for pair in selected_pairs:
        print(f"- {pair.company}: {pair.on_ticker}/{pair.pn_ticker}")

    # ------------------------------------------------------------
    # 5. Build company policies.
    # ------------------------------------------------------------

    policy_map = build_company_policies(
        config=config,
        filter_report=filter_report,
        forced_companies=final_portfolio_weights.keys(),
    )

    if not policy_map:
        raise ValueError("No company policies were created.")

    selected_pairs = [
        pair
        for pair in selected_pairs
        if pair.company in policy_map
    ]

    if not selected_pairs:
        raise ValueError("No selected pairs have available policies.")

    # ------------------------------------------------------------
    # 6. Run individual backtests with individual tax disabled.
    # ------------------------------------------------------------

    signal_history_data_by_company = build_signal_history_data_by_company(
        train_data_by_company=train_data_by_company,
        test_data_by_company=test_data_by_company,
    )

    print("\nIndividual backtests use train + test data for signal calculation.")
    print("Execution and performance are measured only from 2020-01-01 onward.")

    individual_comparisons, individual_metrics_table = run_individual_backtests(
        config=config,
        selected_pairs=selected_pairs,
        test_data_by_company=signal_history_data_by_company,
        policy_map=policy_map,
    )

    if not individual_comparisons:
        raise ValueError("No individual comparison results were created.")

    # ------------------------------------------------------------
    # 7. Individual plots and individual summary.
    # ------------------------------------------------------------

    build_individual_plots(
        config=config,
        individual_comparisons=individual_comparisons,
    )

    print_final_summary(individual_metrics_table)

    # ------------------------------------------------------------
    # 8. Build final portfolio with portfolio-level tax.
    # ------------------------------------------------------------

    portfolio_comparison, tax_table, weights_table = (
        build_final_portfolio_with_portfolio_level_tax(
            config=config,
            individual_comparisons=individual_comparisons,
            weights=final_portfolio_weights,
        )
    )

    # ------------------------------------------------------------
    # 9. Calculate final portfolio metrics.
    # ------------------------------------------------------------

    print("\nFinal portfolio date range:")
    print(f"Start: {portfolio_comparison.index.min().date()}")
    print(f"End:   {portfolio_comparison.index.max().date()}")
    print(f"Observations: {len(portfolio_comparison)}")

    portfolio_metrics_table = calculate_final_portfolio_metrics(
        config=config,
        portfolio_comparison=portfolio_comparison,
        tax_table=tax_table,
        weights_table=weights_table,
    )

    # ------------------------------------------------------------
    # 10. Portfolio plots.
    # ------------------------------------------------------------

    build_final_portfolio_plots(
        config=config,
        portfolio_comparison=portfolio_comparison,
    )

    # ------------------------------------------------------------
    # 11. Final portfolio summary.
    # ------------------------------------------------------------

    print_final_portfolio_summary(portfolio_metrics_table)

    print("\nFinal weighted portfolio completed successfully.")
    print(f"Results folder: {config.paths.results_dir}")
    print(f"Tables folder: {config.paths.tables_dir}")
    print(f"Plots folder: {config.paths.plots_dir}")


if __name__ == "__main__":
    main()