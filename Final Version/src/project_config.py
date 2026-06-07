from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectPaths:
    """
    Centralizes all project paths.

    The goal is to avoid hardcoded paths in the rest of the project.
    All folders and output files should be accessed through this class.
    """

    base_dir: Path = Path(__file__).resolve().parent
    project_dir: Path = field(init=False)

    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)

    results_dir: Path = field(init=False)
    individual_results_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)

    selected_companies_path: Path = field(init=False)
    universe_filter_report_path: Path = field(init=False)
    policy_map_path: Path = field(init=False)
    individual_metrics_path: Path = field(init=False)

    def __post_init__(self):
        """
        Builds all project paths after initialization.
        """

        self.project_dir = self.base_dir.parent

        self.data_dir = self.project_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"

        self.results_dir = self.project_dir / "final_results"
        self.individual_results_dir = self.results_dir / "individual"
        self.tables_dir = self.results_dir / "tables"
        self.plots_dir = self.results_dir / "plots"

        self.selected_companies_path = (
            self.results_dir / "selected_companies.csv"
        )

        self.universe_filter_report_path = (
            self.tables_dir / "universe_filter_report.csv"
        )

        self.policy_map_path = (
            self.tables_dir / "company_policy_map.csv"
        )

        self.individual_metrics_path = (
            self.tables_dir / "individual_strategy_vs_benchmarks.csv"
        )

    def create_directories(self):
        """
        Creates all required folders for the project.
        """

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.individual_results_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestSettings:
    """
    Stores general backtest parameters.

    These parameters should be shared by all companies and should not be
    optimized separately for each pair using test-period performance.
    """

    start_date: str = "2010-01-06"
    end_date: str = "2025-12-31"

    # Fixed out-of-sample period.
    # The training period uses all available observations before this date.
    # The test/backtest period starts at the first available trading day
    # on or after this date.
    test_start_date: str = "2020-01-01"

    download_data: bool = False

    initial_capital_per_pair: float = 1.0

    transaction_cost_rate: float = 0.001
    income_tax_rate: float = 0.15

    use_loss_carryforward: bool = True
    include_transaction_costs_in_tax_basis: bool = False

    minimum_rebalance_difference: float = 0.025

    trading_days_per_year: int = 252


@dataclass
class SignalSettings:
    """
    Stores default signal generation parameters.

    The signal engine may later receive company-specific policies, but these
    are the neutral default values used across the project.
    """

    initial_weight_on: float = 0.50
    initial_weight_pn: float = 0.50

    default_signal_window: int = 252
    minimum_signal_observations: int = 126

    default_entry_threshold: float = 1.50
    default_exit_threshold: float = 0.35


@dataclass
class PolicySettings:
    """
    Stores thresholds used to classify each company into a policy group.

    These thresholds are intentionally general. They are based on training
    statistics, not on test-period strategy performance.
    """

    strong_correlation_threshold: float = 0.75
    acceptable_correlation_threshold: float = 0.60

    strong_cointegration_pvalue: float = 0.05
    acceptable_cointegration_pvalue: float = 0.15

    strong_adf_pvalue: float = 0.05
    acceptable_adf_pvalue: float = 0.10

    high_spread_volatility_threshold: float = 0.12
    medium_spread_volatility_threshold: float = 0.08
    low_spread_volatility_threshold: float = 0.07

    high_quality_threshold: float = 0.70
    medium_quality_threshold: float = 0.55


@dataclass
class UniverseFilterSettings:
    """
    Stores the hard-filter parameters used to define the tradable universe.
    """

    min_observations: int = 500
    max_missing_ratio: float = 0.10
    min_avg_volume: float = 100_000
    min_basic_correlation: float = 0.60

    use_cointegration: bool = True
    use_adf: bool = True

    require_volume_data: bool = False

    top_n_selected_companies: int | None = None


@dataclass
class CompanyUniverse:
    """
    Stores the ON/PN company universe.

    The project can later expand this dictionary, but the final strategy
    should only trade companies that pass the statistical filters.
    """

    company_pairs: dict = field(default_factory=lambda: {
        "PETR": ("PETR3.SA", "PETR4.SA"),
        "BBDC": ("BBDC3.SA", "BBDC4.SA"),
        "ITUB": ("ITUB3.SA", "ITUB4.SA"),
        "CMIG": ("CMIG3.SA", "CMIG4.SA"),
        "USIM": ("USIM3.SA", "USIM5.SA"),
        "ITSA": ("ITSA3.SA", "ITSA4.SA"),
        "GGBR": ("GGBR3.SA", "GGBR4.SA"),
        "BRAP": ("BRAP3.SA", "BRAP4.SA"),
        "GOAU": ("GOAU3.SA", "GOAU4.SA"),
        "TASA": ("TASA3.SA", "TASA4.SA"),
        "OIBR": ("OIBR3.SA", "OIBR4.SA"),
        "POMO": ("POMO3.SA", "POMO4.SA"),
        "SANB": ("SANB3.SA", "SANB4.SA"),
        "KLBN": ("KLBN3.SA", "KLBN4.SA"),
        "SAPR": ("SAPR3.SA", "SAPR4.SA"),
        "UNIP": ("UNIP3.SA", "UNIP6.SA"),
        "BRKM": ("BRKM3.SA", "BRKM5.SA"),
        "TAEE": ("TAEE3.SA", "TAEE4.SA"),
        "ALUP": ("ALUP3.SA", "ALUP4.SA"),
        "RAPT": ("RAPT3.SA", "RAPT4.SA"),
        "BRSR": ("BRSR3.SA", "BRSR6.SA"),
    })

    ibovespa_ticker: str = "^BVSP"


@dataclass
class ProjectConfig:
    """
    Main configuration object for the final project.

    This object should be created once in main.py and then passed to the
    other parts of the pipeline.
    """

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    signals: SignalSettings = field(default_factory=SignalSettings)
    policies: PolicySettings = field(default_factory=PolicySettings)
    universe_filter: UniverseFilterSettings = field(default_factory=UniverseFilterSettings)
    universe: CompanyUniverse = field(default_factory=CompanyUniverse)

    def initialize_project(self):
        """
        Creates all required folders before running the pipeline.
        """

        self.paths.create_directories()