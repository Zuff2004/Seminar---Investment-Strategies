import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Calculates performance metrics for an equity curve or return series.

    The same class is used for:
    - the ON/PN rotation strategy;
    - the passive 50/50 benchmark;
    - the Ibovespa benchmark.

    This keeps all performance calculations consistent across strategies.
    """

    def __init__(self, trading_days_per_year: int = 252):
        """
        Initializes the metric calculator.

        Parameters
        ----------
        trading_days_per_year:
            Number of trading days used for annualization.
        """

        self.trading_days_per_year = int(trading_days_per_year)

    def calculate_from_equity_curve(
        self,
        equity_curve: pd.Series,
        label: str = "",
    ) -> dict:
        """
        Calculates performance metrics from an equity curve.

        Parameters
        ----------
        equity_curve:
            Series containing portfolio/benchmark value through time.

        label:
            Optional label used as a prefix in the returned metric names.

        Returns
        -------
        dict
            Dictionary with total return, annualized return, volatility,
            Sharpe ratio, max drawdown and hit ratio.
        """

        clean_equity = (
            equity_curve
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if clean_equity.empty:
            return self._empty_metrics(label)

        returns = (
            clean_equity
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        return self.calculate_from_returns_and_equity(
            returns=returns,
            equity_curve=clean_equity,
            label=label,
        )

    def calculate_from_returns(
        self,
        returns: pd.Series,
        label: str = "",
    ) -> dict:
        """
        Calculates performance metrics from a return series.

        The equity curve is reconstructed from the return series.
        """

        clean_returns = (
            returns
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if clean_returns.empty:
            return self._empty_metrics(label)

        equity_curve = (1.0 + clean_returns).cumprod()

        return self.calculate_from_returns_and_equity(
            returns=clean_returns,
            equity_curve=equity_curve,
            label=label,
        )

    def calculate_from_returns_and_equity(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        label: str = "",
    ) -> dict:
        """
        Calculates metrics using both returns and equity curve.

        This method is useful when the equity curve already exists and should
        not be reconstructed from returns.
        """

        clean_returns = (
            returns
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        clean_equity = (
            equity_curve
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if clean_equity.empty:
            return self._empty_metrics(label)

        total_return = self._calculate_total_return(clean_equity)
        annualized_return = self._calculate_annualized_return(
            equity_curve=clean_equity,
            total_return=total_return,
        )

        annualized_volatility = self._calculate_annualized_volatility(
            returns=clean_returns,
        )

        sharpe_ratio = self._calculate_sharpe_ratio(
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
        )

        max_drawdown = self._calculate_max_drawdown(clean_equity)

        hit_ratio = self._calculate_hit_ratio(clean_returns)

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "hit_ratio": hit_ratio,
            "observations": len(clean_equity),
        }

        if label:
            metrics = {
                f"{label}_{key}": value
                for key, value in metrics.items()
            }

        return metrics

    def _calculate_total_return(self, equity_curve: pd.Series) -> float:
        """
        Calculates total return from first value to last value.
        """

        if len(equity_curve) < 2:
            return np.nan

        first_value = float(equity_curve.iloc[0])
        last_value = float(equity_curve.iloc[-1])

        if first_value <= 0:
            return np.nan

        return last_value / first_value - 1.0

    def _calculate_annualized_return(
        self,
        equity_curve: pd.Series,
        total_return: float,
    ) -> float:
        """
        Annualizes total return using the number of observations.
        """

        if pd.isna(total_return):
            return np.nan

        number_of_years = len(equity_curve) / self.trading_days_per_year

        if number_of_years <= 0:
            return np.nan

        if 1.0 + total_return <= 0:
            return np.nan

        return (1.0 + total_return) ** (1.0 / number_of_years) - 1.0

    def _calculate_annualized_volatility(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Calculates annualized volatility from daily returns.
        """

        if returns.empty:
            return np.nan

        return returns.std() * np.sqrt(self.trading_days_per_year)

    def _calculate_sharpe_ratio(
        self,
        annualized_return: float,
        annualized_volatility: float,
    ) -> float:
        """
        Calculates a zero-risk-free-rate Sharpe ratio.

        For this project, the Sharpe ratio is used only as a relative
        comparison between strategy and benchmarks.
        """

        if pd.isna(annualized_return):
            return np.nan

        if pd.isna(annualized_volatility) or annualized_volatility == 0:
            return np.nan

        return annualized_return / annualized_volatility

    def _calculate_max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> float:
        """
        Calculates maximum drawdown.

        Drawdown measures the percentage loss from a previous peak.
        """

        if equity_curve.empty:
            return np.nan

        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0

        return drawdown.min()

    def _calculate_hit_ratio(
        self,
        returns: pd.Series,
    ) -> float:
        """
        Calculates the fraction of positive return days.
        """

        if returns.empty:
            return np.nan

        return (returns > 0).mean()

    @staticmethod
    def _empty_metrics(label: str = "") -> dict:
        """
        Returns an empty metric dictionary when input data is invalid.
        """

        metrics = {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "hit_ratio": np.nan,
            "observations": 0,
        }

        if label:
            metrics = {
                f"{label}_{key}": value
                for key, value in metrics.items()
            }

        return metrics