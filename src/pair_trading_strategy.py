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
