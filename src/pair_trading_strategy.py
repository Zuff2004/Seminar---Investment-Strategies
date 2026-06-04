'''
1.0: THIS IS A SIMPLIFIED TESTING ENVIRONMENT FOR THE MANDATORY MEETING

--> STRATEGY IDEA: The strategy compares the relative price relationship between an ON share and a PN share of the same company

--> Instead of staying in cash or shorting one asset, this version is an "always-long rotation strategy:
    a) If PN looks cheap relative to ON, the strategy holds PN
    b) If PN looks expensive relative to ON, the strategy switches to ON
    c) If there is no new signal, the strategy keeps the previous position
    --> Therefore, the strategy is always invested in either PN or ON (holding the long position)

============================================================
MAIN VARIABLES AND PARAMETERS
============================================================

1) on_ticker: Ticker symbol of the ordinary share, for example "ITUB3.SA"
2) pn_ticker: Ticker symbol of the preferred share, for example "ITUB4.SA"
3) benchmark_ticker: Market benchmark used for comparison, for example "^BVSP"
4) start_date / end_date:
    a) Historical period used for the backtest
    b) The chosen period is very important because the relationship between ON and PN shares may change across market cycles (Pandemics, etc)
5) window: --> TO BE BETTER STUDIED
    a) Rolling window used to calculate the moving average and standard deviation of the spread

    b) Smaller window:
        - More sensitive to recent movements
        - Generates more signals
        - Can increase returns, but also noise and overtrading

    c) Larger window:
    - More sensitive to recent movements
        - More stable
        - Generates fewer signals
        - Can reduce noise, but may react too slowly

6) entry_z: Z-score threshold used to trigger a switch between ON and PN
            --> TO BE BETTER STUDIED
    a) Lower entry_z:
        - More trades
        - More sensitive strategy
        - Higher risk of false signals
    b) Higher entry_z:
        - Fewer trades
        - More selective strategy
        - May miss profitable opportunities

7) exit_z:
    a) In a classic long/short pair trade, exit_z would be used to close the position when the spread normalizes
    b) In this version, the strategy is always invested, so exit_z is not used to close the position
    c) It is kept as a parameter for possible future versions of the model

8) alpha:
    a) Intercept from the OLS regression: PN = alpha + beta * ON
    b) It captures the constant price difference between PN and ON

9) beta:
    a) Hedge ration from the OLS regression
    b) It measures how much PN tends to move when ON moves

10) spread: Difference between actual PN price and predicted PN price => spread = PN - (alpha + beta * ON)

11) z_score:
    a) Standardized spread
    b) It shows how far the current spread is from its recent average

12) position:
    a) Current asset held by the strategy:
        - position = 1 -> holding PN
        - position = -1 -> holding ON

13) strategy_return: Return of the asset that the strategy was holding on the previous day
    a) The previous position is used to avoid look-ahead bias

============================================================
CURRENT LIMITATIONS (TO BE WORKED ON)
============================================================
1) Data leakage:
    - We used already known data (we are not really testing it in present scenarios)
    - Divide training sets, etc --> IDEAS
2) Transaction costs and taxes are not incurred
'''
#==========================================
# PART 0: IMPORTING THE REQUIRED LIBRARIES
#==========================================
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm


#==========================================
# PART 1: DEFINING THE CLASS
#==========================================
class ONPNPairTradingStrategy:

    def __init__(self, on_ticker: str, pn_ticker: str, start_date: str, end_date: str, window: int, entry_z: float, exit_z: float, benchmark_ticker: str):
        self.on_ticker = on_ticker
        self.pn_ticker = pn_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.benchmark_ticker = benchmark_ticker
        self.data = None
        self.results = None
        self.alpha = None
        self.beta = None
        self.ols_model = None

    def download_data(self):
        tickers = [self.on_ticker, self.pn_ticker, self.benchmark_ticker]
        data = yf.download(tickers, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)["Close"]
        data = data.dropna()
        self.data = data
        return self.data

    def estimate_hedge_ratio(self):
        if self.data is None: self.download_data()

        df = self.data.copy()

        df["ON"] = df[self.on_ticker]
        df["PN"] = df[self.pn_ticker]

        # OLS regression instead of simple regression
        # PN = alpha + beta * ON
        X = sm.add_constant(df["ON"])
        Y = df["PN"]

        model = sm.OLS(Y, X).fit()

        self.alpha = model.params["const"]
        self.beta = model.params["ON"]
        self.ols_model = model

        print("\nOLS Hedge Ratio Estimation")
        print(f"Alpha: {self.alpha:.4f}")
        print(f"Beta: {self.beta:.4f}")
        print(f"R²: {model.rsquared:.4f}")

        return self.beta, model

    def calculate_spread(self):
        if self.data is None: self.download_data()

        if self.beta is None: self.estimate_hedge_ratio()

        df = self.data.copy()

        df["ON"] = df[self.on_ticker]
        df["PN"] = df[self.pn_ticker]
        df["Benchmark"] = df[self.benchmark_ticker]

        df["spread"] = df["PN"] - self.beta*df["ON"]

        df["spread_mean"] = (df["spread"].rolling(window=self.window).mean())

        df["spread_std"] = (df["spread"].rolling(window=self.window).std())

        df["z_score"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

        self.results = df.dropna()

        return self.results

    def generate_signals(self):
        if self.results is None:
            self.calculate_spread()

        df = self.results.copy()

        df["signal"] = 0

        # PN barata em relação à ON -> comprar PN
        df.loc[df["z_score"] < -self.entry_z, "signal"] = 1

        # PN cara em relação à ON -> comprar ON
        df.loc[df["z_score"] > self.entry_z, "signal"] = -1

        # UPDATED: começa comprado em PN por padrão
        position = 1
        positions = []

        for z in df["z_score"]:

            # PN barata -> ficar comprado em PN
            if z < -self.entry_z:
                position = 1

            # PN cara -> trocar para ON
            elif z > self.entry_z:
                position = -1

            # Se não há sinal novo, mantém a posição anterior
            positions.append(position)

        df["position"] = positions

        self.results = df
        return self.results

    def calculate_returns(self):
        if self.results is None or "position" not in self.results.columns:
            self.generate_signals()

        df = self.results.copy()

        df["return_on"] = df["ON"].pct_change()
        df["return_pn"] = df["PN"].pct_change()
        df["benchmark_return"] = df["Benchmark"].pct_change()

        df["pn_buy_hold"] = df["return_pn"]
        df["on_buy_hold"] = df["return_on"]

        # UPDATED:
        # position = 1  -> comprado em PN
        # position = -1 -> comprado em ON
        # shift(1) evita look-ahead bias
        df["strategy_return"] = np.where(
            df["position"].shift(1) == 1,
            df["return_pn"],
            df["return_on"]
        )

        df["strategy_return"] = df["strategy_return"].fillna(0)
        df["pn_buy_hold"] = df["pn_buy_hold"].fillna(0)
        df["on_buy_hold"] = df["on_buy_hold"].fillna(0)
        df["benchmark_return"] = df["benchmark_return"].fillna(0)

        df["strategy_cumulative"] = (1 + df["strategy_return"]).cumprod()
        df["pn_cumulative"] = (1 + df["pn_buy_hold"]).cumprod()
        df["on_cumulative"] = (1 + df["on_buy_hold"]).cumprod()
        df["benchmark_cumulative"] = (1 + df["benchmark_return"]).cumprod()

        self.results = df
        return self.results

    def performance_metrics(self):
        if self.results is None or "strategy_return" not in self.results.columns: self.calculate_returns()

        df = self.results.copy()

        def annualized_return(returns):
            return returns.mean() * 252

        def annualized_volatility(returns):
            return returns.std() * np.sqrt(252)

        def sharpe_ratio(returns):
            if returns.std() == 0: return np.nan
            return np.sqrt(252) * returns.mean() / returns.std()

        def max_drawdown(cumulative):
            running_max = cumulative.cummax()
            drawdown = cumulative / running_max - 1
            return drawdown.min()

        metrics = pd.DataFrame({
            "Annualized Return": [
                annualized_return(df["strategy_return"]),
                annualized_return(df["pn_buy_hold"]),
                annualized_return(df["benchmark_return"])
            ],
            "Annualized Volatility" : [
                annualized_volatility(df["strategy_return"]),
                annualized_volatility(df["pn_buy_hold"]),
                annualized_volatility(df["benchmark_return"])
            ],
            "Sharpe Ratio" : [
                sharpe_ratio(df["strategy_return"]),
                sharpe_ratio(df["pn_buy_hold"]),
                sharpe_ratio(df["benchmark_return"])
            ],
            "Max Drawdown" : [
                max_drawdown(df["strategy_cumulative"]),
                max_drawdown(df["pn_cumulative"]),
                max_drawdown(df["benchmark_cumulative"])
            ]
        }, index=["ON/PN Pair Strategy",
                  f"Buy & Hold {self.pn_ticker}",
                  "Ibovespa"
        ])

        return metrics

    def run(self):
        self.download_data()
        self.calculate_spread()
        self.generate_signals()
        self.calculate_returns()
        return self.results, self.performance_metrics()