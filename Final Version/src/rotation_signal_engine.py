import numpy as np
import pandas as pd


class RotationSignalEngine:
    """
    Generates ON/PN rotation signals from the log-price spread.

    Core idea:
    - The strategy never exits the company-level position.
    - It only reallocates capital between ON and PN share classes.
    - If ON is expensive relative to PN, the strategy increases PN weight.
    - If ON is cheap relative to PN, the strategy increases ON weight.

    The engine only generates target weights.
    It does not execute trades.
    It does not calculate transaction costs.
    It does not calculate taxes.
    """

    def __init__(
        self,
        initial_weight_on: float = 0.50,
        initial_weight_pn: float = 0.50,
        minimum_signal_observations: int = 126,
    ):
        """
        Initializes the signal engine.

        Parameters
        ----------
        initial_weight_on:
            Initial ON allocation weight.

        initial_weight_pn:
            Initial PN allocation weight.

        minimum_signal_observations:
            Minimum rolling observations required before calculating signals.
        """

        self.initial_weight_on = float(initial_weight_on)
        self.initial_weight_pn = float(initial_weight_pn)
        self.minimum_signal_observations = int(minimum_signal_observations)

        if abs(self.initial_weight_on + self.initial_weight_pn - 1.0) > 1e-8:
            raise ValueError("Initial ON and PN weights must sum to 1.")

    def add_signals(
        self,
        data: pd.DataFrame,
        policy,
    ) -> pd.DataFrame:
        """
        Adds spread, z-score, target weights and signal labels.

        Parameters
        ----------
        data:
            DataFrame containing ON, PN, return_on and return_pn.

        policy:
            CompanyPolicy object created by CompanyPolicyEngine.

        Returns
        -------
        pandas.DataFrame
            Original data with additional signal columns.
        """

        required_columns = ["ON", "PN", "return_on", "return_pn"]

        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Missing column: {column}")

        df = data.copy().sort_index()

        # ------------------------------------------------------------
        # 1. Spread calculation
        # ------------------------------------------------------------
        # The spread is defined as:
        # log(ON price) - log(PN price)
        #
        # Positive spread:
        # ON is expensive relative to PN.
        #
        # Negative spread:
        # ON is cheap relative to PN.
        # ------------------------------------------------------------

        df["spread"] = np.log(df["ON"]) - np.log(df["PN"])

        # ------------------------------------------------------------
        # 2. Rolling spread statistics
        # ------------------------------------------------------------
        # Rolling mean and standard deviation are shifted by one day.
        #
        # This is important:
        # The signal at day t must only use information available before
        # the trading decision at day t.
        # ------------------------------------------------------------

        df["spread_mean"] = (
            df["spread"]
            .rolling(
                window=policy.signal_window,
                min_periods=self.minimum_signal_observations,
            )
            .mean()
            .shift(1)
        )

        df["spread_std"] = (
            df["spread"]
            .rolling(
                window=policy.signal_window,
                min_periods=self.minimum_signal_observations,
            )
            .std()
            .shift(1)
        )

        df["z_score"] = np.where(
            df["spread_std"] == 0,
            np.nan,
            (df["spread"] - df["spread_mean"]) / df["spread_std"],
        )

        # ------------------------------------------------------------
        # 3. Initialize target weights
        # ------------------------------------------------------------

        df["target_weight_on"] = self.initial_weight_on
        df["target_weight_pn"] = self.initial_weight_pn
        df["signal"] = "initial_50_50"

        current_weight_on = self.initial_weight_on
        current_weight_pn = self.initial_weight_pn
        current_signal = "initial_50_50"

        # ------------------------------------------------------------
        # 4. Sequential signal generation
        # ------------------------------------------------------------
        # The loop keeps the previous allocation when no new signal appears.
        # This avoids unrealistic daily forced rebalancing.
        # ------------------------------------------------------------

        for i in range(len(df)):
            z_score = df["z_score"].iloc[i]

            if pd.isna(z_score):
                self._set_row_signal(
                    df=df,
                    row_index=i,
                    weight_on=current_weight_on,
                    weight_pn=current_weight_pn,
                    signal=current_signal,
                )
                continue

            # --------------------------------------------------------
            # ON expensive relative to PN
            # --------------------------------------------------------
            # If z-score is high, ON is expensive compared to PN.
            # The strategy reduces ON weight and increases PN weight.
            # --------------------------------------------------------

            if z_score >= policy.entry_threshold:
                current_weight_on = policy.min_weight_on
                current_weight_pn = 1.0 - current_weight_on
                current_signal = "overweight_pn"

            # --------------------------------------------------------
            # ON cheap relative to PN
            # --------------------------------------------------------
            # If z-score is very negative, ON is cheap compared to PN.
            # The strategy increases ON weight and reduces PN weight.
            # --------------------------------------------------------

            elif z_score <= -policy.entry_threshold:
                current_weight_on = policy.max_weight_on
                current_weight_pn = 1.0 - current_weight_on
                current_signal = "overweight_on"

            # --------------------------------------------------------
            # Spread normalized
            # --------------------------------------------------------
            # If the spread returns close to its mean, the strategy can
            # return to 50/50. This only happens when the policy defines
            # an exit threshold.
            # --------------------------------------------------------

            elif (
                policy.exit_threshold is not None
                and abs(z_score) <= policy.exit_threshold
            ):
                current_weight_on = self.initial_weight_on
                current_weight_pn = self.initial_weight_pn
                current_signal = "return_to_50_50"

            # --------------------------------------------------------
            # No new signal
            # --------------------------------------------------------
            # Keep yesterday's target allocation.
            # --------------------------------------------------------

            else:
                current_signal = "hold_previous_allocation"

            self._set_row_signal(
                df=df,
                row_index=i,
                weight_on=current_weight_on,
                weight_pn=current_weight_pn,
                signal=current_signal,
            )

        # ------------------------------------------------------------
        # 5. Add policy metadata
        # ------------------------------------------------------------

        df["policy_group"] = policy.policy_group
        df["policy_explanation"] = policy.explanation

        return df

    def _set_row_signal(
        self,
        df: pd.DataFrame,
        row_index: int,
        weight_on: float,
        weight_pn: float,
        signal: str,
    ):
        """
        Writes target weights and signal label into one row.
        """

        df.iloc[row_index, df.columns.get_loc("target_weight_on")] = weight_on
        df.iloc[row_index, df.columns.get_loc("target_weight_pn")] = weight_pn
        df.iloc[row_index, df.columns.get_loc("signal")] = signal