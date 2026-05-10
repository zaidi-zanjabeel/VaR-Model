# Multi-Method Portfolio Value at Risk (VaR) Calculator

## Project Overview
This project is a comprehensive risk management tool developed in Python to calculate the **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR/Expected Shortfall)** of a multi-asset portfolio. Unlike simple static calculators, this tool fetches real-time market data to simulate potential losses across different statistical methodologies. 

The script is designed to help portfolio managers understand the "worst-case scenario" for their investments within a specific confidence interval.

## Key Features & Methodologies
The calculator implements the three industry-standard approaches to risk modeling:
* **Historical Simulation:** Uses actual historical price changes to determine the distribution of returns.
* **Parametric (Variance-Covariance):** Utilizes the Normal Distribution, portfolio mean, and standard deviation (Z-scores) to estimate risk.
* **Monte Carlo Simulation:** Runs **10,000 random scenarios** based on portfolio volatility to project future profit/loss outcomes.
* **Conditional VaR (CVaR):** Calculates the Expected Shortfall—the average loss that occurs in the extreme tail (beyond the VaR threshold).

## Technical Stack
* **Data Acquisition:** `yfinance` for fetching live stock and bond market data.
* **Analytical Engine:** `NumPy` and `Pandas` for matrix multiplications, covariance calculations, and return series analysis.
* **Statistical Modeling:** `SciPy (stats.norm)` for generating Z-scores and probability density functions.
* **Visualization:** `Matplotlib` to generate a dual-panel risk dashboard, including a distribution histogram and a formatted Risk Report table.

## How to Use
1. Ensure you have the required libraries installed: `pip install pandas numpy yfinance scipy matplotlib`.
2. Define your `tickers` (e.g., SPY for Stocks, BND for Bonds) and their respective `weights` in the script.
3. Set your initial investment amount and the confidence level (default is 99%).
4. Run the `VaR calculator.py` script.
5. The tool will download historical data, compute risk metrics across all methods, and display a professional dark-themed dashboard.
