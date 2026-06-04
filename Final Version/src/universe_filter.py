import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


class UniverseFilter:
    """
    Filters and ranks the ON/PN trading universe.

    The class separates two concepts:

    1. Hard filters:
       Objective conditions that a pair must satisfy to be included.

    2. Quality score:
       A ranking metric used to compare the pairs that passed the filters.

    The filter is applied only on the training sample.
    This prevents future test-period information from influencing the
    universe selection.
    """

    def __init__(
        self,
        min_observations: int = 500,
        max_missing_ratio: float = 0.10,
        min_avg_volume: float = 100_000,
        min_basic_correlation: float = 0.60,
        use_cointegration: bool = True,
        use_adf: bool = True,
        require_volume_data: bool = False,
    ):
        """
        Initializes the universe filter.

        Parameters
        ----------
        min_observations:
            Minimum number of observations required.

        max_missing_ratio:
            Maximum allowed missing-data ratio.

        min_avg_volume:
            Minimum average volume if volume data is required.

        min_basic_correlation:
            Minimum ON/PN return correlation required.

        use_cointegration:
            If True, the filter calculates the Engle-Granger cointegration test.

        use_adf:
            If True, the filter calculates the ADF test on the spread.

        require_volume_data:
            If True, pairs without valid volume data are rejected.
        """

        self.min_observations = int(min_observations)
        self.max_missing_ratio = float(max_missing_ratio)
        self.min_avg_volume = float(min_avg_volume)
        self.min_basic_correlation = float(min_basic_correlation)

        self.use_cointegration = bool(use_cointegration)
        self.use_adf = bool(use_adf)
        self.require_volume_data = bool(require_volume_data)

    # ============================================================
    # Public method
    # ============================================================

    def filter_pairs(
        self,
        pair_objects: list,
        train_data_by_company: dict,
        top_n: int | None = None,
    ) -> tuple[list, pd.DataFrame]:
        """
        Applies hard filters and ranks the valid ON/PN pairs.

        Parameters
        ----------
        pair_objects:
            List of PairData objects.

        train_data_by_company:
            Dictionary containing training data for each company.

            Expected format:
            {
                "PETR": train_dataframe,
                "ITUB": train_dataframe,
            }

        top_n:
            Optional number of top-ranked pairs to select.
            If None, all pairs that passed the hard filters are selected.

        Returns
        -------
        tuple
            selected_pairs, report_df
        """

        report_rows = []
        passed_pairs = []

        for pair in pair_objects:
            company = pair.company

            if company not in train_data_by_company:
                report_rows.append({
                    "pair": company,
                    "passed_hard_filters": False,
                    "rejection_reasons": "No training data available",
                })
                continue

            train_data = train_data_by_company[company]

            passed, reasons, metrics = self._evaluate_pair(
                pair=pair,
                train_data=train_data,
            )

            report_rows.append({
                "pair": company,
                "passed_hard_filters": passed,
                "rejection_reasons": "; ".join(reasons),
                **metrics,
            })

            if passed:
                passed_pairs.append(pair)

        report_df = pd.DataFrame(report_rows)

        if report_df.empty:
            return [], report_df

        report_df = self._add_quality_score(report_df)

        passed_report = report_df[
            report_df["passed_hard_filters"] == True
        ].copy()

        passed_report = passed_report.sort_values(
            by="quality_score",
            ascending=False,
        )

        if top_n is not None:
            passed_report = passed_report.head(top_n)

        selected_names = set(passed_report["pair"])

        report_df["selected_for_analysis"] = report_df["pair"].isin(
            selected_names
        )

        selected_pairs = [
            pair
            for pair in passed_pairs
            if pair.company in selected_names
        ]

        return selected_pairs, report_df

    # ============================================================
    # Pair evaluation
    # ============================================================

    def _evaluate_pair(
        self,
        pair,
        train_data: pd.DataFrame,
    ) -> tuple[bool, list[str], dict]:
        """
        Evaluates one ON/PN pair using training-sample data only.
        """

        reasons = []
        metrics = {}

        if train_data is None or train_data.empty:
            return False, ["Training data is empty"], metrics

        required_columns = ["ON", "PN", "return_on", "return_pn"]

        for column in required_columns:
            if column not in train_data.columns:
                return False, [f"Missing column: {column}"], metrics

        data = train_data.copy().sort_index()

        # ------------------------------------------------------------
        # 1. Observations and missing data
        # ------------------------------------------------------------

        observations = len(data)
        metrics["observations"] = observations

        if observations < self.min_observations:
            reasons.append("Not enough observations")

        missing_ratio = (
            data[required_columns]
            .isna()
            .mean()
            .mean()
        )

        metrics["missing_ratio"] = missing_ratio

        if missing_ratio > self.max_missing_ratio:
            reasons.append("Too many missing values")

        clean_data = data.dropna(subset=required_columns)

        metrics["clean_observations"] = len(clean_data)

        if clean_data.empty:
            return False, ["No clean observations"], metrics

        # ------------------------------------------------------------
        # 2. Return correlation
        # ------------------------------------------------------------

        correlation = clean_data["return_on"].corr(clean_data["return_pn"])
        metrics["correlation"] = correlation

        if pd.isna(correlation):
            reasons.append("Correlation is missing")
        elif correlation < self.min_basic_correlation:
            reasons.append("Correlation below minimum threshold")

        # ------------------------------------------------------------
        # 3. Spread volatility
        # ------------------------------------------------------------

        spread = np.log(clean_data["ON"]) - np.log(clean_data["PN"])
        spread = spread.replace([np.inf, -np.inf], np.nan).dropna()

        spread_volatility = spread.std()
        metrics["spread_volatility"] = spread_volatility

        if pd.isna(spread_volatility) or spread_volatility <= 0:
            reasons.append("Invalid spread volatility")

        # ------------------------------------------------------------
        # 4. Cointegration test
        # ------------------------------------------------------------

        cointegration_pvalue = np.nan

        if self.use_cointegration:
            cointegration_pvalue = self._calculate_cointegration_pvalue(
                clean_data=clean_data,
            )

        metrics["cointegration_pvalue"] = cointegration_pvalue

        # ------------------------------------------------------------
        # 5. ADF test on spread
        # ------------------------------------------------------------

        adf_pvalue = np.nan

        if self.use_adf:
            adf_pvalue = self._calculate_adf_pvalue(spread)

        metrics["adf_pvalue"] = adf_pvalue

        # ------------------------------------------------------------
        # 6. Volume filter
        # ------------------------------------------------------------

        avg_volume_on = np.nan
        avg_volume_pn = np.nan
        min_average_volume = np.nan

        if pair.volumes is not None:
            aligned_volume = pair.volumes.reindex(clean_data.index)
            avg_volume_on = aligned_volume["ON"].mean()
            avg_volume_pn = aligned_volume["PN"].mean()
            min_average_volume = min(avg_volume_on, avg_volume_pn)

        metrics["avg_volume_on"] = avg_volume_on
        metrics["avg_volume_pn"] = avg_volume_pn
        metrics["min_average_volume"] = min_average_volume

        if self.require_volume_data:
            if pd.isna(min_average_volume):
                reasons.append("Volume data required but unavailable")
            elif min_average_volume < self.min_avg_volume:
                reasons.append("Average volume below minimum threshold")

        passed = len(reasons) == 0

        return passed, reasons, metrics

    # ============================================================
    # Statistical tests
    # ============================================================

    def _calculate_cointegration_pvalue(
        self,
        clean_data: pd.DataFrame,
    ) -> float:
        """
        Calculates the Engle-Granger cointegration p-value.
        """

        try:
            on_log = np.log(clean_data["ON"])
            pn_log = np.log(clean_data["PN"])

            on_log = on_log.replace([np.inf, -np.inf], np.nan)
            pn_log = pn_log.replace([np.inf, -np.inf], np.nan)

            test_data = pd.concat(
                [on_log.rename("ON"), pn_log.rename("PN")],
                axis=1,
            ).dropna()

            if len(test_data) < 50:
                return np.nan

            _, pvalue, _ = coint(
                test_data["ON"],
                test_data["PN"],
            )

            return float(pvalue)

        except Exception:
            return np.nan

    def _calculate_adf_pvalue(
        self,
        spread: pd.Series,
    ) -> float:
        """
        Calculates the ADF p-value for the ON/PN spread.
        """

        try:
            clean_spread = (
                spread
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(clean_spread) < 50:
                return np.nan

            result = adfuller(clean_spread)

            return float(result[1])

        except Exception:
            return np.nan

    # ============================================================
    # Quality score
    # ============================================================

    def _add_quality_score(
        self,
        report_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adds a composite quality score to the filter report.

        The score rewards:
        - higher ON/PN correlation;
        - stronger mean-reversion evidence;
        - sufficient spread volatility;
        - more observations.
        """

        report = report_df.copy()

        report["correlation_score"] = report["correlation"].clip(
            lower=0.0,
            upper=1.0,
        )

        report["cointegration_score"] = report["cointegration_pvalue"].apply(
            self._pvalue_to_score
        )

        report["adf_score"] = report["adf_pvalue"].apply(
            self._pvalue_to_score
        )

        report["spread_opportunity_score"] = report["spread_volatility"].apply(
            self._spread_volatility_to_score
        )

        max_observations = report["observations"].max()

        if pd.isna(max_observations) or max_observations <= 0:
            report["observation_score"] = 0.0
        else:
            report["observation_score"] = (
                report["observations"] / max_observations
            ).clip(lower=0.0, upper=1.0)

        report["quality_score"] = (
            0.30 * report["correlation_score"]
            + 0.25 * report["cointegration_score"]
            + 0.20 * report["adf_score"]
            + 0.15 * report["spread_opportunity_score"]
            + 0.10 * report["observation_score"]
        )

        report.loc[
            report["passed_hard_filters"] == False,
            "quality_score",
        ] = np.nan

        return report

    @staticmethod
    def _pvalue_to_score(pvalue) -> float:
        """
        Converts a statistical p-value into a quality score.

        Lower p-values imply stronger statistical evidence and therefore
        a higher score.
        """

        if pd.isna(pvalue):
            return 0.0

        pvalue = float(pvalue)

        if pvalue <= 0.01:
            return 1.0

        if pvalue <= 0.05:
            return 0.85

        if pvalue <= 0.10:
            return 0.65

        if pvalue <= 0.20:
            return 0.40

        return 0.15

    @staticmethod
    def _spread_volatility_to_score(spread_volatility) -> float:
        """
        Converts spread volatility into an opportunity score.

        Very low spread volatility means few economically meaningful trades.
        Extremely high spread volatility may indicate instability.

        The score therefore rewards moderate-to-high but not extreme volatility.
        """

        if pd.isna(spread_volatility):
            return 0.0

        spread_volatility = float(spread_volatility)

        if spread_volatility <= 0:
            return 0.0

        if spread_volatility < 0.04:
            return 0.20

        if spread_volatility < 0.08:
            return 0.45

        if spread_volatility < 0.12:
            return 0.75

        if spread_volatility < 0.20:
            return 1.00

        return 0.65