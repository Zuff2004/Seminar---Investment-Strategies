from dataclasses import dataclass

import pandas as pd


@dataclass
class CompanyPolicy:
    """
    Stores the final ON/PN rotation policy for one company.

    The policy defines:
    - how aggressive the allocation can be;
    - when the strategy should enter a trade;
    - when it should return to 50/50;
    - whether tax-loss harvesting is allowed;
    - the statistical explanation behind the assigned group.
    """

    company: str
    policy_group: str

    min_weight_on: float
    max_weight_on: float

    entry_threshold: float
    exit_threshold: float | None

    signal_window: int

    allow_tax_loss_harvesting: bool

    explanation: str


class CompanyPolicyEngine:
    """
    Builds company-level policies using only training-sample statistics.

    This class is used for both:

    1. Statistical-filtering results:
       - only companies that passed the hard filters receive a policy.

    2. Final manual/fundamental-weighted portfolio:
       - companies that passed the hard filters receive normal statistical
         policies;
       - companies manually forced into the final portfolio but that did not
         pass the hard filters receive forced_default_rotation.

    This avoids mixing statistically selected defensive_rotation companies
    with manually forced companies.
    """

    def __init__(self, policy_settings):
        """
        Initializes the policy engine.

        Parameters
        ----------
        policy_settings:
            PolicySettings object from project_config.py.
        """

        self.settings = policy_settings

    # ============================================================
    # Public methods
    # ============================================================

    def build_policy_map(
        self,
        filter_report: pd.DataFrame,
        forced_companies: list | set | tuple | None = None,
    ) -> dict:
        """
        Builds a policy dictionary.

        Normal statistical behavior:
        - if forced_companies is None, only companies that passed hard filters
          receive a policy.

        Final/manual portfolio behavior:
        - if forced_companies is provided, companies in that list are allowed
          into the policy map even if they did not pass the hard filters;
        - companies that passed hard filters receive a normal statistical policy;
        - companies that did not pass hard filters but were forced receive
          forced_default_rotation.

        Parameters
        ----------
        filter_report:
            DataFrame created by UniverseFilter.

        forced_companies:
            Optional list/set/tuple of companies manually included in the final
            portfolio.

        Returns
        -------
        dict
            Dictionary mapping company code to CompanyPolicy.
        """

        required_columns = [
            "pair",
            "passed_hard_filters",
            "correlation",
            "spread_volatility",
            "cointegration_pvalue",
            "adf_pvalue",
            "quality_score",
        ]

        for column in required_columns:
            if column not in filter_report.columns:
                raise ValueError(f"Missing column in filter report: {column}")

        report = filter_report.copy()
        report["pair"] = report["pair"].astype(str).str.upper()

        forced_companies = set(forced_companies or [])
        forced_companies = {
            str(company).upper()
            for company in forced_companies
        }

        # ------------------------------------------------------------
        # Case 1:
        # Normal statistical pipeline.
        # Only companies that passed hard filters are selected.
        # ------------------------------------------------------------

        if not forced_companies:
            selected_report = report[
                report["passed_hard_filters"] == True
            ].copy()

        # ------------------------------------------------------------
        # Case 2:
        # Final manual/fundamental portfolio.
        # Keep companies that passed hard filters OR were manually forced.
        # ------------------------------------------------------------

        else:
            selected_report = report[
                (report["passed_hard_filters"] == True)
                | (report["pair"].isin(forced_companies))
            ].copy()

        policy_map = {}

        for _, row in selected_report.iterrows():
            company = row["pair"]
            passed_hard_filters = bool(row["passed_hard_filters"])

            # --------------------------------------------------------
            # Companies that passed hard filters:
            # receive normal statistical policy.
            # --------------------------------------------------------

            if passed_hard_filters:
                policy = self.build_single_policy(
                    company=company,
                    correlation=row["correlation"],
                    spread_volatility=row["spread_volatility"],
                    cointegration_pvalue=row["cointegration_pvalue"],
                    adf_pvalue=row["adf_pvalue"],
                    quality_score=row["quality_score"],
                )

            # --------------------------------------------------------
            # Companies manually included but not statistically selected:
            # receive separate forced default policy.
            # --------------------------------------------------------

            elif company in forced_companies:
                policy = self.build_forced_default_policy(
                    company=company,
                    correlation=row["correlation"],
                    spread_volatility=row["spread_volatility"],
                    cointegration_pvalue=row["cointegration_pvalue"],
                    adf_pvalue=row["adf_pvalue"],
                    quality_score=row["quality_score"],
                )

            else:
                continue

            policy_map[company] = policy

        return policy_map

    def build_policy_table(
        self,
        policy_map: dict,
    ) -> pd.DataFrame:
        """
        Converts the policy map into a table that can be saved as CSV.

        This table documents which statistical/forced rule was assigned to
        each company.
        """

        rows = []

        for company, policy in policy_map.items():
            rows.append({
                "company": company,
                "policy_group": policy.policy_group,
                "min_weight_on": policy.min_weight_on,
                "max_weight_on": policy.max_weight_on,
                "entry_threshold": policy.entry_threshold,
                "exit_threshold": policy.exit_threshold,
                "signal_window": policy.signal_window,
                "allow_tax_loss_harvesting": policy.allow_tax_loss_harvesting,
                "explanation": policy.explanation,
            })

        policy_table = pd.DataFrame(rows)

        if not policy_table.empty:
            policy_table = policy_table.sort_values("company").reset_index(drop=True)

        return policy_table

    # ============================================================
    # Normal statistical policy rules
    # ============================================================

    def build_single_policy(
        self,
        company: str,
        correlation: float,
        spread_volatility: float,
        cointegration_pvalue: float,
        adf_pvalue: float,
        quality_score: float,
    ) -> CompanyPolicy:
        """
        Builds one policy from training-sample statistical indicators.

        This method is intended for companies that passed the hard statistical
        filters.
        """

        correlation = self._safe_number(correlation, fallback=0.0)

        spread_volatility = self._safe_number(
            spread_volatility,
            fallback=0.0,
        )

        cointegration_pvalue = self._safe_number(
            cointegration_pvalue,
            fallback=1.0,
        )

        adf_pvalue = self._safe_number(
            adf_pvalue,
            fallback=1.0,
        )

        quality_score = self._safe_number(
            quality_score,
            fallback=0.0,
        )

        strong_relation = (
            correlation >= self.settings.strong_correlation_threshold
        )

        acceptable_relation = (
            correlation >= self.settings.acceptable_correlation_threshold
        )

        strong_reversion = (
            cointegration_pvalue <= self.settings.strong_cointegration_pvalue
            and adf_pvalue <= self.settings.strong_adf_pvalue
        )

        acceptable_reversion = (
            cointegration_pvalue <= self.settings.acceptable_cointegration_pvalue
            or adf_pvalue <= self.settings.acceptable_adf_pvalue
        )

        weak_reversion = (
            cointegration_pvalue > 0.20
            and adf_pvalue > 0.20
        )

        high_spread_opportunity = (
            spread_volatility >= self.settings.high_spread_volatility_threshold
        )

        medium_spread_opportunity = (
            spread_volatility >= self.settings.medium_spread_volatility_threshold
        )

        low_spread_opportunity = (
            spread_volatility < self.settings.low_spread_volatility_threshold
        )

        high_quality = (
            quality_score >= self.settings.high_quality_threshold
        )

        medium_quality = (
            quality_score >= self.settings.medium_quality_threshold
        )

        # ============================================================
        # 1. Conservative preservation
        # ============================================================
        # Strong ON/PN relation, but weak mean-reversion evidence.
        #
        # Economic interpretation:
        # The two share classes move together, but the spread does not show
        # reliable reversion. Therefore, aggressive rotation may damage returns.
        # The strategy stays extremely close to 50/50.
        # ============================================================

        if strong_relation and weak_reversion:
            return CompanyPolicy(
                company=company,
                policy_group="conservative_preservation",
                min_weight_on=0.49,
                max_weight_on=0.51,
                entry_threshold=3.0,
                exit_threshold=1.0,
                signal_window=252,
                allow_tax_loss_harvesting=True,
                explanation=(
                    "Strong ON/PN relation but weak mean-reversion evidence. "
                    "The strategy preserves company-level exposure and remains "
                    "very close to the passive 50/50 allocation."
                ),
            )

        # ============================================================
        # 2. Active reversion
        # ============================================================
        # Strong statistical reversion and high spread opportunity.
        #
        # Economic interpretation:
        # The pair has enough evidence to justify active ON/PN rotation.
        # ============================================================

        if acceptable_relation and strong_reversion and high_spread_opportunity:
            return CompanyPolicy(
                company=company,
                policy_group="active_reversion",
                min_weight_on=0.0,
                max_weight_on=1.0,
                entry_threshold=1.0,
                exit_threshold=0.5,
                signal_window=252,
                allow_tax_loss_harvesting=True,
                explanation=(
                    "Strong reversion evidence and high spread opportunity. "
                    "The strategy is allowed to rotate actively between ON and PN."
                ),
            )

        # ============================================================
        # 3. Extreme-only rotation
        # ============================================================
        # Acceptable relation and reversion, but low spread opportunity.
        #
        # Economic interpretation:
        # Normal deviations may be too small after costs and taxes.
        # The strategy only reacts to extreme spread deviations.
        # ============================================================

        if acceptable_relation and acceptable_reversion and low_spread_opportunity:
            return CompanyPolicy(
                company=company,
                policy_group="extreme_only",
                min_weight_on=0.0,
                max_weight_on=1.0,
                entry_threshold=3.0,
                exit_threshold=0.10,
                signal_window=252,
                allow_tax_loss_harvesting=True,
                explanation=(
                    "Acceptable reversion evidence, but low spread volatility. "
                    "The strategy trades only extreme deviations."
                ),
            )

        # ============================================================
        # 4. Moderate rotation
        # ============================================================
        # Reasonable statistical evidence and medium spread opportunity.
        #
        # Economic interpretation:
        # The pair can be traded, but the strategy should not be too aggressive.
        # ============================================================

        if acceptable_relation and acceptable_reversion and medium_spread_opportunity:
            return CompanyPolicy(
                company=company,
                policy_group="moderate_rotation",
                min_weight_on=0.0,
                max_weight_on=1.0,
                entry_threshold=1.5,
                exit_threshold=0.5,
                signal_window=252,
                allow_tax_loss_harvesting=True,
                explanation=(
                    "Acceptable statistical quality and medium spread opportunity. "
                    "The strategy uses moderate ON/PN rotation."
                ),
            )

        # ============================================================
        # 5. High-quality stable pair
        # ============================================================
        # High-quality and strongly related pair, but not necessarily enough
        # spread opportunity for active trading.
        #
        # Economic interpretation:
        # Stable pairs can still offer opportunities, but the strategy should
        # wait for stronger deviations and stay close to 50/50.
        # ============================================================

        if strong_relation and high_quality:
            return CompanyPolicy(
                company=company,
                policy_group="high_quality_stable_pair",
                min_weight_on=0.49,
                max_weight_on=0.51,
                entry_threshold=4.0,
                exit_threshold=0.1,
                signal_window=252,
                allow_tax_loss_harvesting=True,
                explanation=(
                    "High-quality stable pair. The strategy remains controlled "
                    "and reacts only to stronger deviations."
                ),
            )

        # ============================================================
        # 6. Medium-quality defensive rotation
        # ============================================================
        # The pair is not strong enough for active rotation, but it is not
        # completely unusable.
        # ============================================================

        if acceptable_relation and medium_quality:
            return CompanyPolicy(
                company=company,
                policy_group="medium_quality_defensive_rotation",
                min_weight_on=0.45,
                max_weight_on=0.55,
                entry_threshold=2.0,
                exit_threshold=0.25,
                signal_window=252,
                allow_tax_loss_harvesting=True,
                explanation=(
                    "Medium-quality pair with limited statistical evidence. "
                    "The strategy uses defensive and limited rotation."
                ),
            )

        # ============================================================
        # 7. Defensive fallback
        # ============================================================
        # Weak or unclear evidence, but the company still passed hard filters.
        #
        # Economic interpretation:
        # Stay close to the passive benchmark and avoid excessive turnover.
        # ============================================================

        return CompanyPolicy(
            company=company,
            policy_group="defensive_rotation",
            min_weight_on=0.475,
            max_weight_on=0.525,
            entry_threshold=1.0,
            exit_threshold=0.5,
            signal_window=252,
            allow_tax_loss_harvesting=True,
            explanation=(
                "Company passed the hard statistical filters, but the evidence "
                "was not strong enough for a more aggressive rule. The strategy "
                "uses a defensive rotation band around the passive 50/50 benchmark."
            ),
        )

    # ============================================================
    # Forced/default policy for manual portfolio companies
    # ============================================================

    def build_forced_default_policy(
        self,
        company: str,
        correlation: float,
        spread_volatility: float,
        cointegration_pvalue: float,
        adf_pvalue: float,
        quality_score: float,
    ) -> CompanyPolicy:
        """
        Builds the default policy for companies manually included in the final
        fundamental-weighted portfolio but not selected by the hard statistical
        filters.

        Economic interpretation:
        - the company is included for portfolio construction/fundamental reasons;
        - the ON/PN statistical evidence was not strong enough to pass the hard
          universe filters;
        - therefore, the strategy can rotate, but only inside a conservative
          allocation band and only after stronger deviations.

        This policy is intentionally separate from defensive_rotation.
        """

        correlation = self._safe_number(correlation, fallback=0.0)

        spread_volatility = self._safe_number(
            spread_volatility,
            fallback=0.0,
        )

        cointegration_pvalue = self._safe_number(
            cointegration_pvalue,
            fallback=1.0,
        )

        adf_pvalue = self._safe_number(
            adf_pvalue,
            fallback=1.0,
        )

        quality_score = self._safe_number(
            quality_score,
            fallback=0.0,
        )

        return CompanyPolicy(
            company=company,
            policy_group="forced_default_rotation",

            # Conservative band for manually included companies.
            # These companies did not pass the hard statistical filters.
            min_weight_on=0.0,
            max_weight_on=1.0,

            # Stronger entry requirement than normal defensive rotation.
            entry_threshold=1.0,
            exit_threshold=0.25,

            signal_window=252,
            allow_tax_loss_harvesting=True,

            explanation=(
                "Company was manually included in the final fundamental-weighted "
                "portfolio despite not passing the hard statistical universe "
                "filters. It therefore receives forced_default_rotation: a "
                "separate conservative rule that keeps the ON/PN allocation close "
                "to 50/50 and requires stronger spread deviations before trading."
            ),
        )

    # ============================================================
    # Helper methods
    # ============================================================

    @staticmethod
    def _safe_number(
        value,
        fallback: float,
    ) -> float:
        """
        Converts invalid numeric values to a safe fallback.
        """

        if pd.isna(value):
            return fallback

        return float(value)