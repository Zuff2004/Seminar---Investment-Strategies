import numpy as np
import pandas as pd

from individual_tax_account import IndividualTaxAccount


class ShareClassRotationBacktester:
    """
    Backtests an ON/PN share-class rotation strategy for one company.

    Core principles:
    - The strategy always keeps exposure to the company.
    - It never exits the company-level position.
    - It only reallocates capital between ON and PN share classes.
    - Trades are value-based, not quantity-based.
    - Taxes are calculated monthly at the individual company level.
    - Realized losses are tracked and can offset future realized gains.
    """

    def __init__(
        self,
        initial_capital: float = 1.0,
        transaction_cost_rate: float = 0.001,
        tax_rate: float = 0.15,
        minimum_rebalance_difference: float = 0.025,
        include_transaction_costs_in_tax_basis: bool = True,
        use_loss_carryforward: bool = True,
        execution_start_date: str | pd.Timestamp | None = "2020-01-01",
        execution_end_date: str | pd.Timestamp | None = "2025-12-31",
    ):
        """
        Initializes the backtester.

        Parameters
        ----------
        initial_capital:
            Initial capital allocated to the company pair.

        transaction_cost_rate:
            Proportional transaction cost applied to buys and sells.

        tax_rate:
            Tax rate applied to taxable realized gains.

        minimum_rebalance_difference:
            Minimum difference between current and target ON weight required
            to trigger a trade.

        include_transaction_costs_in_tax_basis:
            If True, transaction costs are included in average cost basis.

        use_loss_carryforward:
            If True, realized losses can offset future realized gains.

        execution_start_date:
            First date on which the portfolio is actually executed and
            performance is measured. Signal columns may be calculated with
            earlier training data, but the backtest starts here.

        execution_end_date:
            Last date included in the measured out-of-sample backtest.
        """

        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.tax_rate = float(tax_rate)
        self.minimum_rebalance_difference = float(minimum_rebalance_difference)
        self.include_transaction_costs_in_tax_basis = bool(
            include_transaction_costs_in_tax_basis
        )
        self.use_loss_carryforward = bool(use_loss_carryforward)
        self.execution_start_date = (
            pd.Timestamp(execution_start_date)
            if execution_start_date is not None
            else None
        )
        self.execution_end_date = (
            pd.Timestamp(execution_end_date)
            if execution_end_date is not None
            else None
        )

    # ============================================================
    # Main backtest
    # ============================================================

    def backtest_pair(
        self,
        data: pd.DataFrame,
        pair_name: str = "",
    ) -> pd.DataFrame:
        """
        Runs the individual ON/PN rotation backtest.

        Parameters
        ----------
        data:
            DataFrame containing:
            - ON
            - PN
            - return_on
            - return_pn
            - target_weight_on
            - target_weight_pn
            - signal

        pair_name:
            Company identifier used for reporting.

        Returns
        -------
        pandas.DataFrame
            Daily strategy results with positions, weights, costs, taxes
            and equity curve.
        """

        required_columns = [
            "ON",
            "PN",
            "return_on",
            "return_pn",
            "target_weight_on",
            "target_weight_pn",
        ]

        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Missing column: {column}")

        df = data.copy().sort_index()

        if df.empty:
            raise ValueError("Backtest data is empty.")

        # ------------------------------------------------------------
        # Critical train/test date correction
        # ------------------------------------------------------------
        # The input data may include the training period so that rolling
        # signal columns in early 2020 can be calculated with pre-2020
        # history. However, the portfolio must be initialized and measured
        # only in the out-of-sample execution period.
        #
        # Therefore, we slice only here, after the external signal pipeline
        # has already created target_weight_on / target_weight_pn using the
        # full train + test history.
        # ------------------------------------------------------------

        if self.execution_start_date is not None:
            df = df.loc[df.index >= self.execution_start_date]

        if self.execution_end_date is not None:
            df = df.loc[df.index <= self.execution_end_date]

        if df.empty:
            raise ValueError(
                "Backtest data is empty after applying execution date window."
            )

        tax_account = IndividualTaxAccount(
            tax_rate=self.tax_rate,
            use_loss_carryforward=self.use_loss_carryforward,
        )

        # ------------------------------------------------------------
        # Initial portfolio:
        # The strategy starts with 50% ON and 50% PN.
        # ------------------------------------------------------------

        first_date = df.index[0]
        first_on_price = float(df["ON"].iloc[0])
        first_pn_price = float(df["PN"].iloc[0])

        if first_on_price <= 0 or first_pn_price <= 0:
            raise ValueError("Initial ON and PN prices must be positive.")

        initial_on_value = self.initial_capital * 0.50
        initial_pn_value = self.initial_capital * 0.50

        on_quantity = initial_on_value / first_on_price
        pn_quantity = initial_pn_value / first_pn_price

        # Average cost starts as the initial purchase price.
        on_average_cost = first_on_price
        pn_average_cost = first_pn_price

        cash = 0.0

        previous_date = None

        rows = []

        # ============================================================
        # Daily loop
        # ============================================================

        for date, row in df.iterrows():
            on_price = float(row["ON"])
            pn_price = float(row["PN"])

            target_weight_on = float(row["target_weight_on"])
            target_weight_pn = float(row["target_weight_pn"])

            signal = row["signal"] if "signal" in df.columns else ""

            # --------------------------------------------------------
            # Close previous month if the month changed.
            # Taxes are paid at month-end and reduce cash.
            # --------------------------------------------------------

            tax_record = tax_account.close_month_if_needed(
                current_date=date,
                previous_date=previous_date,
            )

            tax_paid_today = 0.0

            if tax_record is not None:
                tax_paid_today = float(tax_record["tax_paid"])
                cash -= tax_paid_today

            # --------------------------------------------------------
            # Current portfolio value before today's rebalance.
            # --------------------------------------------------------

            on_value_before = on_quantity * on_price
            pn_value_before = pn_quantity * pn_price

            portfolio_value_before_trade = (
                on_value_before
                + pn_value_before
                + cash
            )

            if portfolio_value_before_trade <= 0:
                raise ValueError(
                    f"Portfolio value became non-positive for {pair_name} on {date}"
                )

            current_weight_on = on_value_before / (
                on_value_before + pn_value_before
            )

            current_weight_pn = 1.0 - current_weight_on

            weight_difference = target_weight_on - current_weight_on

            # Daily values initialized before possible trade.
            traded = False
            trade_direction = "no_trade"

            gross_sale_value = 0.0
            gross_buy_value = 0.0

            total_transaction_cost = 0.0
            realized_pnl = 0.0

            on_quantity_before_trade = on_quantity
            pn_quantity_before_trade = pn_quantity

            on_average_cost_before_trade = on_average_cost
            pn_average_cost_before_trade = pn_average_cost

            # --------------------------------------------------------
            # Trade only if the difference is large enough.
            # --------------------------------------------------------

            if abs(weight_difference) >= self.minimum_rebalance_difference:
                traded = True

                target_on_value = (
                    target_weight_on
                    * (on_value_before + pn_value_before)
                )

                current_on_value = on_value_before

                # Positive value_to_move:
                # Need to increase ON by selling PN.
                #
                # Negative value_to_move:
                # Need to reduce ON by selling ON.
                value_to_move = target_on_value - current_on_value

                if value_to_move > 0:
                    trade_direction = "sell_pn_buy_on"

                    sell_result = self._sell_asset(
                        quantity_before=pn_quantity,
                        average_cost_before=pn_average_cost,
                        gross_sell_value=value_to_move,
                        price=pn_price,
                    )

                    pn_quantity = sell_result["quantity_after"]
                    pn_average_cost = sell_result["average_cost_after"]

                    cash += sell_result["net_cash_from_sale"]

                    gross_sale_value += sell_result["gross_sale_value"]
                    total_transaction_cost += sell_result["transaction_cost"]
                    realized_pnl += sell_result["realized_pnl"]

                    tax_account.register_sale(
                        date=date,
                        gross_sale_value=sell_result["gross_sale_value"],
                        realized_pnl=sell_result["realized_pnl"],
                        transaction_cost=sell_result["transaction_cost"],
                    )

                    buy_result = self._buy_asset(
                        quantity_before=on_quantity,
                        average_cost_before=on_average_cost,
                        available_cash=cash,
                        price=on_price,
                    )

                    on_quantity = buy_result["quantity_after"]
                    on_average_cost = buy_result["average_cost_after"]

                    cash -= buy_result["cash_used"]

                    gross_buy_value += buy_result["gross_buy_value"]
                    total_transaction_cost += buy_result["transaction_cost"]

                elif value_to_move < 0:
                    trade_direction = "sell_on_buy_pn"

                    sell_value = abs(value_to_move)

                    sell_result = self._sell_asset(
                        quantity_before=on_quantity,
                        average_cost_before=on_average_cost,
                        gross_sell_value=sell_value,
                        price=on_price,
                    )

                    on_quantity = sell_result["quantity_after"]
                    on_average_cost = sell_result["average_cost_after"]

                    cash += sell_result["net_cash_from_sale"]

                    gross_sale_value += sell_result["gross_sale_value"]
                    total_transaction_cost += sell_result["transaction_cost"]
                    realized_pnl += sell_result["realized_pnl"]

                    tax_account.register_sale(
                        date=date,
                        gross_sale_value=sell_result["gross_sale_value"],
                        realized_pnl=sell_result["realized_pnl"],
                        transaction_cost=sell_result["transaction_cost"],
                    )

                    buy_result = self._buy_asset(
                        quantity_before=pn_quantity,
                        average_cost_before=pn_average_cost,
                        available_cash=cash,
                        price=pn_price,
                    )

                    pn_quantity = buy_result["quantity_after"]
                    pn_average_cost = buy_result["average_cost_after"]

                    cash -= buy_result["cash_used"]

                    gross_buy_value += buy_result["gross_buy_value"]
                    total_transaction_cost += buy_result["transaction_cost"]

            # --------------------------------------------------------
            # Portfolio value after possible trade and tax payment.
            # --------------------------------------------------------

            on_value_after = on_quantity * on_price
            pn_value_after = pn_quantity * pn_price

            portfolio_value_after_trade = (
                on_value_after
                + pn_value_after
                + cash
            )

            if portfolio_value_after_trade > 0:
                final_weight_on = on_value_after / (
                    on_value_after + pn_value_after
                )
                final_weight_pn = 1.0 - final_weight_on
            else:
                final_weight_on = np.nan
                final_weight_pn = np.nan

            # --------------------------------------------------------
            # Store daily row.
            # --------------------------------------------------------

            rows.append({
                "date": date,
                "pair": pair_name,

                "ON": on_price,
                "PN": pn_price,

                "target_weight_on": target_weight_on,
                "target_weight_pn": target_weight_pn,

                "weight_on_before_trade": current_weight_on,
                "weight_pn_before_trade": current_weight_pn,

                "weight_on": final_weight_on,
                "weight_pn": final_weight_pn,

                "on_quantity": on_quantity,
                "pn_quantity": pn_quantity,

                "on_average_cost": on_average_cost,
                "pn_average_cost": pn_average_cost,

                "on_value": on_value_after,
                "pn_value": pn_value_after,
                "cash": cash,

                "strategy_value": portfolio_value_after_trade,
                "strategy_return": np.nan,

                "signal": signal,
                "traded": traded,
                "trade_direction": trade_direction,

                "gross_sale_value": gross_sale_value,
                "gross_buy_value": gross_buy_value,

                "transaction_cost": total_transaction_cost,
                "realized_pnl": realized_pnl,

                "tax_paid": tax_paid_today,
                "accumulated_loss": tax_account.get_accumulated_loss(),

                "on_unrealized_pnl": (
                    on_quantity * (on_price - on_average_cost)
                ),
                "pn_unrealized_pnl": (
                    pn_quantity * (pn_price - pn_average_cost)
                ),

                "on_quantity_before_trade": on_quantity_before_trade,
                "pn_quantity_before_trade": pn_quantity_before_trade,

                "on_average_cost_before_trade": on_average_cost_before_trade,
                "pn_average_cost_before_trade": pn_average_cost_before_trade,
            })

            previous_date = date

        # ------------------------------------------------------------
        # Close final month.
        # ------------------------------------------------------------

        final_tax_record = tax_account.close_final_month(df.index[-1])

        final_tax_paid = float(final_tax_record["tax_paid"])

        if rows:
            rows[-1]["tax_paid"] += final_tax_paid
            rows[-1]["cash"] -= final_tax_paid
            rows[-1]["strategy_value"] -= final_tax_paid
            rows[-1]["accumulated_loss"] = tax_account.get_accumulated_loss()

        result = pd.DataFrame(rows)
        result = result.set_index("date")
        result = result.sort_index()

        result["strategy_return"] = (
            result["strategy_value"]
            .pct_change()
            .fillna(0.0)
        )

        result["strategy_cumulative_return"] = (
            result["strategy_value"]
            / self.initial_capital
            - 1.0
        )

        # ------------------------------------------------------------
        # Add optional signal columns from the input data.
        # ------------------------------------------------------------

        optional_columns = [
            "spread",
            "spread_mean",
            "spread_std",
            "z_score",
            "policy_group",
            "policy_explanation",
        ]

        for column in optional_columns:
            if column in df.columns:
                result[column] = df[column]

        return result

    # ============================================================
    # Trade helpers
    # ============================================================

    def _sell_asset(
        self,
        quantity_before: float,
        average_cost_before: float,
        gross_sell_value: float,
        price: float,
    ) -> dict:
        """
        Sells part of an asset position and calculates realized PnL.

        The realized PnL is based on average cost accounting.
        """

        if (
            gross_sell_value <= 0
            or price <= 0
            or quantity_before <= 0
        ):
            return {
                "quantity_after": quantity_before,
                "average_cost_after": average_cost_before,
                "transaction_cost": 0.0,
                "realized_pnl": 0.0,
                "gross_sale_value": 0.0,
                "net_cash_from_sale": 0.0,
            }

        maximum_sell_value = quantity_before * price
        gross_sell_value = min(gross_sell_value, maximum_sell_value)

        quantity_sold = gross_sell_value / price

        transaction_cost = gross_sell_value * self.transaction_cost_rate
        net_cash_from_sale = gross_sell_value - transaction_cost

        if self.include_transaction_costs_in_tax_basis:
            sale_value_for_tax = gross_sell_value - transaction_cost
        else:
            sale_value_for_tax = gross_sell_value

        realized_pnl = (
            sale_value_for_tax
            - average_cost_before * quantity_sold
        )

        quantity_after = quantity_before - quantity_sold

        if quantity_after <= 1e-12:
            quantity_after = 0.0
            average_cost_after = 0.0
        else:
            average_cost_after = average_cost_before

        return {
            "quantity_after": quantity_after,
            "average_cost_after": average_cost_after,
            "transaction_cost": transaction_cost,
            "realized_pnl": realized_pnl,
            "gross_sale_value": gross_sell_value,
            "net_cash_from_sale": net_cash_from_sale,
        }

    def _buy_asset(
        self,
        quantity_before: float,
        average_cost_before: float,
        available_cash: float,
        price: float,
    ) -> dict:
        """
        Buys an asset using available cash and updates average cost.
        """

        if available_cash <= 0 or price <= 0:
            return {
                "quantity_after": quantity_before,
                "average_cost_after": average_cost_before,
                "transaction_cost": 0.0,
                "gross_buy_value": 0.0,
                "cash_used": 0.0,
            }

        gross_buy_value = available_cash / (1.0 + self.transaction_cost_rate)
        transaction_cost = gross_buy_value * self.transaction_cost_rate
        cash_used = gross_buy_value + transaction_cost

        quantity_bought = gross_buy_value / price

        previous_total_cost = quantity_before * average_cost_before

        if self.include_transaction_costs_in_tax_basis:
            new_total_cost = (
                previous_total_cost
                + gross_buy_value
                + transaction_cost
            )
        else:
            new_total_cost = previous_total_cost + gross_buy_value

        quantity_after = quantity_before + quantity_bought

        if quantity_after > 0:
            average_cost_after = new_total_cost / quantity_after
        else:
            average_cost_after = 0.0

        return {
            "quantity_after": quantity_after,
            "average_cost_after": average_cost_after,
            "transaction_cost": transaction_cost,
            "gross_buy_value": gross_buy_value,
            "cash_used": cash_used,
        }