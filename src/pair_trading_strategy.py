# IMPORTANT: This is only a testing environment and a simplified version for the mandatory meeting

#==========================================
# PART 0: IMPORTING THE REQUIRED LIBRARIES
#==========================================
import yfinance as yf
import pandas as pd
import numpy as np


#==========================================
# PART 1: DEFINING THE CLASS
#==========================================
class ONPNPairTradingStrategy:

    def __init__(self, on_ticker: str, pn_ticker: str, start_date: str, end_date: str, window: str, entry_z: str, exit_z: str, benchmarkt_ticker: str):
        self.on_ticker = on_ticker,
        self.pn_ticker = pn_ticker,
        self.start_date = start_date,
        self.end_date = end_date,
        self.window = window,
        self.entry_z = entry_z,
        self.exit_z = exit_z,
        self.benchmarkt_ticker = benchmarkt_ticker
        self.data = None,
        self.results = None

    def download_data(self):
        tickers = [self.on_ticker, self.pn_ticker, self.start_date, self.end_date, self.window, self.benchmarkt_ticker]
        data = yf.download(tickers, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)["Close"]
        data.dropna()
        self.data = data
        return self.data

    def calculate_spread(self):
        if self.data is None: self.download_data()
        df = self.data.copy()

        df["ON"] = df[self.on_ticker]
        df["PN"] = df[self.pn_ticker]
        df["Benchmark"] = df[self.benchmarkt_ticker]

        df["spread"] = df["PN"] - df["ON"]
        df["spread_mean"] = df["spread"].rolling(window=self.window).mean()
        df["spread_std"] = df["spread"].rolling(window=self.window).std()
        df["z_score"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

        self.results = df.droptna()
        return self.results

    def generate_signals(self):
        if self.results is None: self.calculate_spread()
        df = self.results.copy()

        df["signal"] = 0

        # PN expensive relative to ON: short PN, long ON
        df.loc[df["z_score"] > self.entry_z, "signal"] = -1

        # PN cheap relative to ON: long PN, short ON
        df.loc[df["z_score"] < -self.entry_z, "signal"] = 1

        # Exit zone
        df.loc[df["z_score"].abs() < self.exit_z, "signal"] = 0

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

        self.results = df
        return df

    def calculate_returns(self):
        if self.results is None or "position" not in self.results.columns: self.generate_signals()
        df = self.results.copy()

        df["return_on"] = df["ON"].pct_change()
        df["return_pn"] = df["PN"].pct_change()
        df["return_benchmark"] = df["Benchmark"].pct_change()

        # position = 1 => long PN, short ON
        # position = -1 => short PN, long ON
        df["strategy_return"] = df["position"].shift(1) * (df["return_pn"] - df["return_on"])

        df["pn_buy_hold"] = df["return_pn"]
        df["benchmark_return"] = df["return_benchmark"]

        df["strategy_cumultative"] = (1 + df["strategy_return"]).cumprod()
        df["pn_cumultative"] = (1 + df["pn_buy_hold"]).cumprod()
        df["benchmark_cumultative"] = (1 + df["benchmark_return"]).cumprod()

        self.results = df.dropna()
        return self.results

    