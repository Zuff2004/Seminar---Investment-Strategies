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
    Test version of the company-level policy engine.

    This class is intended for main_test.py only.

    Difference from the production CompanyPolicyEngine:
    - the original engine creates policies only for companies that passed
      the universe hard filters;
    - this test engine can force selected companies into the policy map
      through the forced_companies argument.

    This is useful for exploratory testing:
    - the final project logic remains unchanged;
    - main_test.py can test companies individually even when they failed
      the universe filter.
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

        Normal behavior:
        - companies that passed the hard filters receive their normal policy.

        Test behavior:
        - companies listed in forced_companies also receive a policy,
          even if they failed the hard filters.

        Forced companies that failed the hard filters receive a defensive
        exploratory policy.
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

        forced_companies = set(forced_companies or [])
        forced_companies = {
            str(company).upper()
            for company in forced_companies
        }

        report = filter_report.copy()
        report["pair"] = report["pair"].astype(str).str.upper()

        selected_report = report[
            (report["passed_hard_filters"] == True)
            | (report["pair"].isin(forced_companies))
        ].copy()

        policy_map = {}

        for _, row in selected_report.iterrows():
            company = row["pair"]
            passed_hard_filters = bool(row["passed_hard_filters"])

            if not passed_hard_filters and company in forced_companies:
                policy = self.build_forced_manual_test_policy(
                    company=company,
                    correlation=row["correlation"],
                    spread_volatility=row["spread_volatility"],
                    cointegration_pvalue=row["cointegration_pvalue"],
                    adf_pvalue=row["adf_pvalue"],
                    quality_score=row["quality_score"],
                )
            else:
                policy = self.build_single_policy(
                    company=company,
                    correlation=row["correlation"],
                    spread_volatility=row["spread_volatility"],
                    cointegration_pvalue=row["cointegration_pvalue"],
                    adf_pvalue=row["adf_pvalue"],
                    quality_score=row["quality_score"],
                )

            policy_map[company] = policy

        return policy_map

    def build_policy_table(
        self,
        policy_map: dict,
    ) -> pd.DataFrame:
        """
        Converts the policy map into a table that can be saved as CSV.

        This table is useful for documenting which statistical rule was
        assigned to each company.
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

        return pd.DataFrame(rows)

    def build_forced_manual_test_policy(
        self,
        company: str,
        correlation: float,
        spread_volatility: float,
        cointegration_pvalue: float,
        adf_pvalue: float,
        quality_score: float,
    ) -> CompanyPolicy:
        """
        Builds a defensive exploratory policy for companies forced into
        main_test.py even though they failed the hard filters.

        This policy is intentionally conservative:
        - it stays close to 50/50;
        - it only reacts to larger deviations;
        - it avoids giving weak pairs an aggressive rotation rule.
        """

        correlation = self._safe_number(correlation, fallback=0.0)
        spread_volatility = self._safe_number(spread_volatility, fallback=0.0)

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
            policy_group="forced_manual_test_defensive",
            min_weight_on=0.45,
            max_weight_on=0.55,
            entry_threshold=2.50,
            exit_threshold=0.20,
            signal_window=252,
            allow_tax_loss_harvesting=True,
            explanation=(
                "Company was manually forced into the exploratory test even "
                "though it did not pass the hard universe filters. The strategy "
                "uses a defensive rule close to 50/50 to avoid giving weak or "
                "unclear ON/PN evidence an aggressive rotation policy. "
                f"Training statistics: correlation={correlation:.4f}, "
                f"spread_volatility={spread_volatility:.4f}, "
                f"cointegration_pvalue={cointegration_pvalue:.4f}, "
                f"adf_pvalue={adf_pvalue:.4f}, "
                f"quality_score={quality_score:.4f}."
            ),
        )

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
        """

        correlation = self._safe_number(correlation, fallback=0.0)
        spread_volatility = self._safe_number(spread_volatility, fallback=0.0)

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
        # The strategy stays close to 50/50.
        # ============================================================

        if strong_relation and weak_reversion:
            return CompanyPolicy(
                company=company,
                policy_group="conservative_preservation",
                min_weight_on=0.49,
                max_weight_on=0.51,
                entry_threshold=3.5,
                exit_threshold=1.0,
                signal_window=252,
                allow_tax_loss_harvesting=True,
                explanation=(
                    "Strong ON/PN relation but weak mean-reversion evidence. "
                    "The strategy preserves company-level exposure and remains "
                    "close to the passive 50/50 allocation."
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
        # wait for stronger deviations.
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
        # 6. Medium-quality fallback
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
        # Weak or unclear evidence.
        #
        # Economic interpretation:
        # Stay almost passive and avoid unnecessary turnover.
        # ============================================================

        return CompanyPolicy(
            company=company,
            policy_group="defensive_rotation",
            min_weight_on=0.45,
            max_weight_on=0.55,
            entry_threshold=2.50,
            exit_threshold=0.20,
            signal_window=252,
            allow_tax_loss_harvesting=True,
            explanation=(
                "Weak or unclear statistical evidence. The strategy stays close "
                "to the passive 50/50 benchmark."
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