import sys
import os
import matplotlib.pyplot as plt
from pair_trading_strategy import ONPNPairTradingStrategy

def main():
    strategy = ONPNPairTradingStrategy(
        on_ticker = input("On ticker: "),
        pn_ticker = input("Pn ticker: "),
        start_date = input("Start date: "),
        end_date = input("End date: "),
        window = int(input("Window: ")),
        entry_z = float(input("Entry Z: ")),
        exit_z = float(input("Exit Z: ")),
        benchmark_ticker = input("Benchmark ticker: ")
    )

    results, metrics = strategy.run()

    print("\nPerformance Metrics:")
    print(metrics)

    # Plot cumulative performance
    results[
        ["strategy_cumulative", "pn_cumulative", "benchmark_cumulative"]
    ].plot(figsize=(12, 6))

    plt.title("ON/PN Pair Strategy vs Benchmark")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()