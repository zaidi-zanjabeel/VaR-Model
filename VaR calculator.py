import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt 

tickers = ['SPY', 'BND']
weights = np.array([0.7, 0.3])
investment = 500000
days = 1
conf_level = 0.99

df = yf.download(tickers, start="2023-01-01", end="2025-12-31")['Close'] 
returns = df.pct_change().dropna()

returns['profit/loss'] = investment*returns[tickers].dot(weights)
historical_VaR = (np.percentile(returns['profit/loss'], (1 - conf_level)*100))*np.sqrt(days)
print(f"Portfolio Historical {days}-Day VaR: {historical_VaR:.2f} ")

exp_returns = returns[tickers].mean()
st_dev = returns[tickers].std()
cov_matrix = returns[tickers].cov()
port_stdev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
port_mean = (np.sum(exp_returns*weights))*investment*days
Z_score = norm.ppf(conf_level)

parametric_VaR = port_mean - (investment*port_stdev*Z_score*np.sqrt(days))
print(f"Portfolio Parametric {days}-Day VaR: {parametric_VaR:.2f}")

simulations = 10000
random_Z = np.random.standard_normal(simulations)
Scenario_VaRs = port_mean - (investment*port_stdev*random_Z*np.sqrt(days))

MonteCarlo_VaR = np.percentile(Scenario_VaRs, (1 - conf_level)*100)
print(f"Portfolio MonteCarlo {days}-Day VaR: {MonteCarlo_VaR:.2f}")

CVaR = np.mean(Scenario_VaRs[Scenario_VaRs <= MonteCarlo_VaR])

#Output Summary
plt.style.use('dark_background')
fig, (ax_hist, ax_table) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

# Left Side: Histogram
ax_hist.hist(Scenario_VaRs, bins=100, color='skyblue', alpha=0.7, edgecolor='navy')
ax_hist.axvline(MonteCarlo_VaR, color='red', linestyle='--', linewidth=2, label=f'VaR: {MonteCarlo_VaR:,.2f}')
ax_hist.axvline(CVaR, color='orange', linestyle='--', linewidth=2, label=f'CVaR: {CVaR:,.2f}')
ax_hist.set_title(f"Monte Carlo Risk Simulation", fontsize=14, pad=15)
ax_hist.set_xlabel("Portfolio Profit/loss ($)")
ax_hist.set_ylabel("Frequency")
ax_hist.legend()

# Right Side: Risk Report Table
ax_table.axis('off') 
report_data = [
    ["Metric", "Value"],
    ["Investment", f"${investment:,}"],
    ["Time Horizon", f"{days}-Day VaR"],
    ["Confidence", f"{conf_level*100}%"],
    ["Parametric VaR", f"-${abs(parametric_VaR):,.2f}"],
    ["Historical VaR", f"-${abs(historical_VaR):,.2f}"],
    ["Monte Carlo VaR", f"-${abs(MonteCarlo_VaR):,.2f}"],
    ["Expected Shortfall (CVaR)", f"-${abs(CVaR):,.2f}"]
]

# Table design
table = ax_table.table(cellText=report_data, loc='center', cellLoc='left', colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 3)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('white') 
    if row == 0: 
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50') 
    else: 
        cell.set_text_props(color='white') 
        cell.set_facecolor('black')

ax_table.set_title("Portfolio Risk Summary", fontsize=14, pad=10, color='white')

plt.tight_layout()
plt.show()