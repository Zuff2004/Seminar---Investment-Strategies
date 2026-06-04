import pandas as pd


class IndividualTaxAccount:
    """
    Tracks tax accounting for one individual ON/PN company pair.

    The tax account is calculated separately for each company.

    Main logic:
    - realized gains and losses are accumulated during each month;
    - at month-end, realized losses offset realized gains;
    - unused losses are carried forward;
    - tax is paid only on positive taxable profit after loss offsets.

    This class does not execute trades.
    It only receives realized PnL values from the backtester.
    """

    def __init__(
        self,
        tax_rate: float = 0.15,
        use_loss_carryforward: bool = True,
    ):
        """
        Initializes the individual tax account.

        Parameters
        ----------
        tax_rate:
            Income tax rate applied to taxable realized profits.

        use_loss_carryforward:
            If True, realized losses are carried forward and can offset
            future realized gains for the same company.
        """

        self.tax_rate = float(tax_rate)
        self.use_loss_carryforward = bool(use_loss_carryforward)

        self.accumulated_loss = 0.0

        self.current_month_realized_pnl = 0.0
        self.current_month_sales_value = 0.0
        self.current_month_transaction_cost = 0.0

        self.monthly_records = []

    # ============================================================
    # Daily registration
    # ============================================================

    def register_sale(
        self,
        date,
        gross_sale_value: float,
        realized_pnl: float,
        transaction_cost: float = 0.0,
    ):
        """
        Registers one realized sale during the current month.

        Parameters
        ----------
        date:
            Date of the sale.

        gross_sale_value:
            Gross value sold before transaction costs.

        realized_pnl:
            Realized gain or loss from the sale.

        transaction_cost:
            Transaction cost paid on the sale.
        """

        self.current_month_sales_value += float(gross_sale_value)
        self.current_month_realized_pnl += float(realized_pnl)
        self.current_month_transaction_cost += float(transaction_cost)

    # ============================================================
    # Month-end tax calculation
    # ============================================================

    def close_month(
        self,
        date,
    ) -> dict:
        """
        Closes the current tax month and calculates tax due.

        The account is reset after closing the month.
        """

        monthly_realized_pnl = float(self.current_month_realized_pnl)

        loss_used = 0.0
        taxable_profit = 0.0
        tax_paid = 0.0

        # ------------------------------------------------------------
        # Case 1:
        # The month ended with a realized loss.
        # The loss is added to accumulated losses.
        # ------------------------------------------------------------

        if monthly_realized_pnl < 0:
            if self.use_loss_carryforward:
                self.accumulated_loss += abs(monthly_realized_pnl)

        # ------------------------------------------------------------
        # Case 2:
        # The month ended with a realized gain.
        # Accumulated losses can offset this gain.
        # ------------------------------------------------------------

        elif monthly_realized_pnl > 0:
            if self.use_loss_carryforward:
                loss_used = min(
                    monthly_realized_pnl,
                    self.accumulated_loss,
                )

                taxable_profit = monthly_realized_pnl - loss_used
                self.accumulated_loss -= loss_used

            else:
                taxable_profit = monthly_realized_pnl

            tax_paid = taxable_profit * self.tax_rate

        # ------------------------------------------------------------
        # Case 3:
        # Realized PnL is exactly zero.
        # No tax is paid and no loss is added.
        # ------------------------------------------------------------

        else:
            taxable_profit = 0.0
            tax_paid = 0.0
            loss_used = 0.0

        record = {
            "date": date,
            "monthly_realized_pnl": monthly_realized_pnl,
            "monthly_sales_value": float(self.current_month_sales_value),
            "monthly_transaction_cost": float(self.current_month_transaction_cost),
            "loss_used": float(loss_used),
            "taxable_profit": float(taxable_profit),
            "tax_paid": float(tax_paid),
            "accumulated_loss_after": float(self.accumulated_loss),
        }

        self.monthly_records.append(record)

        self._reset_current_month()

        return record

    def close_month_if_needed(
        self,
        current_date,
        previous_date,
    ) -> dict | None:
        """
        Closes the tax month when the calendar month changes.

        This method should be called by the backtester while iterating
        through daily observations.

        Returns None if the month did not change.
        """

        if previous_date is None:
            return None

        current_period = pd.Timestamp(current_date).to_period("M")
        previous_period = pd.Timestamp(previous_date).to_period("M")

        if current_period != previous_period:
            return self.close_month(previous_date)

        return None

    def close_final_month(
        self,
        final_date,
    ) -> dict:
        """
        Closes the last month of the backtest.

        This must be called after the daily loop ends.
        """

        return self.close_month(final_date)

    # ============================================================
    # Helpers
    # ============================================================

    def _reset_current_month(self):
        """
        Resets monthly realized values after month-end tax calculation.
        """

        self.current_month_realized_pnl = 0.0
        self.current_month_sales_value = 0.0
        self.current_month_transaction_cost = 0.0

    def get_monthly_records(self) -> pd.DataFrame:
        """
        Returns all monthly tax records as a DataFrame.
        """

        return pd.DataFrame(self.monthly_records)

    def get_accumulated_loss(self) -> float:
        """
        Returns the current accumulated loss balance.
        """

        return float(self.accumulated_loss)