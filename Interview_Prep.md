qa_content = """# Quantitative Backtesting Framework for EMA Angle Crossover Strategies
## Project Summary & Interview Prep Guide

### Core Elevator Pitch (The "Tell me about this project" Answer)
"I architected an algorithmic backtesting engine in Python to evaluate moving average crossover strategies. My research focused on fixing a dimensional inconsistency in how geometric angles are usually calculated by formulating a scale-independent normalized angle metric. I then built a 70/30 Out-of-Sample Walk-Forward Analysis pipeline to eliminate look-ahead bias and tested the strategy across 1,440 parameter combinations. Finally, I integrated realistic transaction cost modeling and used non-parametric statistics (Wilcoxon Signed-Rank Test) to prove that while the math was theoretically superior, the strategy failed to overcome real-world friction. I authored a paper on these negative empirical findings to warn against publication bias and overfitting in algorithmic trading."

---

## Category 1: Project Overview & Resume Walkthrough

**1. What was the primary objective of this project?**
**Answer:** The primary objective was to critically evaluate the "EMA Angle" trading heuristic. Specifically, I wanted to fix the mathematical dimensional flaw in how angles are usually calculated (nominal price vs. time) by creating a scale-independent normalized angle, and then rigorously test if this improved metric actually generated profitable trades after accounting for slippage and commissions.

**2. Why did you choose this specific topic for your research?**
**Answer:** I noticed a significant gap between theoretical technical analysis and empirical reality. Many traders use moving average angles, but calculating an angle on nominal price is dimensionally inconsistent (it changes depending on if the asset is priced at $1 or $1000). I wanted to apply rigorous Data Science and Quantitative Finance principles to test if correcting this math actually mattered in the real world.

**3. What was the most challenging part of this project?**
**Answer:** The most challenging part was architecting the Walk-Forward Analysis (WFA) pipeline. Preventing look-ahead bias and ensuring that the In-Sample optimization didn't accidentally peek at Out-of-Sample data required very careful Pandas index management and execution logic.

**4. You mentioned "negative empirical findings." Can you explain what you found?**
**Answer:** I found that while my normalized angle successfully filtered out bad trades in a zero-friction environment, it failed in realistic conditions. The 10 basis points of round-trip friction (spread and commission) completely destroyed the alpha. The strategy simply didn't capture enough price movement post-crossover to break even against the costs of trading.

**5. Why are you proud of a project that yielded a negative result?**
**Answer:** In data science and quant finance, honesty and rigor are more important than faking profitability. By correctly identifying that the theoretical edge was consumed by transaction costs, I avoided the trap of overfitting. Publishing robust negative findings helps the scientific community avoid dead-ends.

**6. If you had 3 more months to work on this, what would you add?**
**Answer:** I would implement a Machine Learning layer. Instead of a static angle threshold, I would use the normalized angle as a feature input alongside Volume and ATR into an XGBoost classifier to predict true vs. false breakouts. I would also expand the dataset to the entire S&P 500 cross-section.

**7. How did you organize the codebase?**
**Answer:** I used a modular, object-oriented architecture. I separated concerns into `config.yaml` for parameters, `indicators.py` for vectorized pandas math, `strategy.py` for signal generation, `backtest.py` for the execution engine, and `optimizer.py` for the grid search coordination.

**8. What role did your professor (Dev Narayan Yadav) play in this?**
**Answer:** Prof. Yadav acted as the Principal Investigator and Guide. I pitched the architecture and executed the Python engineering, and he reviewed the statistical methodology and guided the academic framing of the paper for eventual IEEE submission.

**9. How do you know your backtester is accurate?**
**Answer:** I implemented a strict T+1 execution model. Signals were generated at the close of the current bar, and trades were executed at the open of the next bar. This mathematically prevents the algorithm from trading on information it shouldn't have yet.

**10. What makes your framework different from just using Backtrader or VectorBT?**
**Answer:** While existing libraries are great, building the engine from scratch using core Pandas allowed me to deeply understand the mechanics of vectorization, trade-level friction accounting, and custom WFA rolling windows. It proved my software engineering capabilities rather than just my ability to call an API.

---

## Category 2: Quantitative Finance & Trading Logic

**11. What is an Exponential Moving Average (EMA) and why use it over SMA?**
**Answer:** An EMA applies exponential weighting, giving more importance to recent prices, whereas an SMA weights all periods equally. I chose EMA because it reduces lag and responds faster to momentum shifts, which is critical for trend-following strategies.

**12. Explain the "dimensional inconsistency" in traditional geometric angles.**
**Answer:** A geometric angle is $\\arctan(\\Delta Y / \\Delta X)$. In trading, $\\Delta Y$ is nominal currency (e.g., $10) and $\\Delta X$ is time (e.g., 1 day). Taking the arctangent of currency over time is mathematically invalid because it's scale-dependent. A 5-degree angle on a $60,000 Bitcoin looks completely different from a 5-degree angle on a $20 Silver asset.

**13. How did you normalize the angle?**
**Answer:** I created a dimensionless rate of change: $S_t = (EMA_t - EMA_{t-1}) / EMA_{t-1}$. This produces a pure percentage shift. I then took the arctangent of $S_t$. This made the angle bounded between -90 and +90 degrees and perfectly comparable across completely different asset classes.

**14. What are Transaction Costs and how did you model them?**
**Answer:** Transaction costs include broker commissions and slippage (the difference between expected price and actual execution price). I modeled a highly realistic 10 basis points (0.10%) round-trip cost (5 bps entry, 5 bps exit).

**15. What is slippage, and why does it occur?**
**Answer:** Slippage happens because of market latency and order book liquidity. By the time your algorithm decides to buy at $100, the best available offer in the limit order book might have shifted to $100.05. 

**16. Why do EMA crossover strategies fail in consolidating markets?**
**Answer:** They are trend-following indicators. In sideways, range-bound markets, the fast and slow EMAs constantly cross each other, triggering multiple false signals ("whipsaws"). This generates massive transaction costs without capturing any directional trend.

**17. What is the Sharpe Ratio?**
**Answer:** It measures risk-adjusted return: (Return of Portfolio - Risk-Free Rate) / Standard Deviation of Portfolio's Excess Return. It tells you how much excess return you are getting for the extra volatility you endure.

**18. Why did you also calculate the Sortino Ratio?**
**Answer:** The Sharpe ratio penalizes both upside and downside volatility. The Sortino ratio only penalizes downside volatility (downside deviation). In algorithmic trading, we actually like upside volatility, so Sortino is often a better metric for evaluating strategy risk.

**19. What is Maximum Drawdown (MDD)?**
**Answer:** MDD is the maximum observed loss from a historical peak to a trough before a new peak is attained. It measures the worst-case scenario for a strategy. My empirical testing showed severe MDDs when friction was applied.

**20. Explain the T+1 Execution Model.**
**Answer:** If my logic analyzes data up to the close of Monday (Time $T$), the trade can only be realistically executed at the open of Tuesday (Time $T+1$). If you execute at Monday's close, you are introducing look-ahead bias because you are acting on the closing price at the exact millisecond it is formed.

---

## Category 3: Data Science & Walk-Forward Analysis

**21. What is Look-Ahead Bias?**
**Answer:** It's the cardinal sin of backtesting. It occurs when a strategy uses information that would not have been available at the time the trade was executed (e.g., using tomorrow's high to set today's target).

**22. What is Overfitting (Curve Fitting) in algorithmic trading?**
**Answer:** It happens when you optimize parameters (like EMA lengths) so heavily on past data that the algorithm learns the exact noise of that historical period rather than a generalized pattern. It leads to excellent backtests but catastrophic live trading.

**23. How did you prevent Overfitting in your project?**
**Answer:** I implemented Walk-Forward Analysis (WFA) with a strict 70% In-Sample (IS) and 30% Out-of-Sample (OOS) chronological split. 

**24. Walk me through your Walk-Forward Analysis pipeline.**
**Answer:** I divided the dataset. I ran 1,440 parameter combinations exclusively on the first 70% of the data. I found the parameters that yielded the highest Net PnL. I then "froze" those parameters and applied them to the remaining unseen 30% of the data. All my reported conclusions are based solely on that 30% unseen data.

**25. Why not use K-Fold Cross Validation like in standard Machine Learning?**
**Answer:** K-Fold randomizes data subsets. In time-series finance, data is serially correlated (autocorrelation). Randomly splitting time-series data leaks future information into the training set (look-ahead bias). WFA preserves the chronological integrity of the data.

**26. What was the purpose of your Grid Search?**
**Answer:** To systematically explore the parameter space. I tested multiple combinations of Fast EMAs, Slow EMAs, and Angle Thresholds (e.g., 0.01 to 0.10 degrees) to find the local optima of the strategy.

**27. What assets did you test and why?**
**Answer:** I tested Bitcoin, Gold, and Silver. I wanted a diverse multi-asset universe to prove that my normalized angle calculation was truly scale-independent across high-volatility (BTC) and legacy macro assets (Gold/Silver).

**28. How did you handle data aggregation?**
**Answer:** I resampled tick/minute-level data into multiple timeframes (1m, 5m, 15m, 1h). This allowed me to test the impact of transaction costs across different trading frequencies.

**29. What happens to a high-frequency strategy (like 1m timeframe) when transaction costs are applied?**
**Answer:** It usually collapses. In my findings, the 1m and 5m timeframes were profitable at zero friction but highly negative at 10 bps friction because the average win size on those short timeframes was smaller than the cost to execute the trade.

**30. Why is Out-of-Sample (OOS) performance the only thing that matters?**
**Answer:** Because In-Sample performance is just memorization. OOS performance proves generalized predictive power. If a strategy fails OOS, it has zero economic value.

---

## Category 4: Statistics & Hypothesis Testing

**31. Why did you use Non-Parametric statistical tests?**
**Answer:** Financial returns (especially in high-frequency algorithmic trading) are notoriously non-normal. They exhibit fat tails and high kurtosis. Parametric tests like the T-Test assume a normal distribution. Non-parametric tests make no such assumptions, making them mathematically valid for financial data.

**32. What is the Wilcoxon Signed-Rank Test?**
**Answer:** It is a non-parametric test used to compare two paired groups. I used it to compare the paired daily returns of the Standard EMA strategy versus my Normalized Angle EMA strategy to see if the angle filter produced a statistically significant difference in median returns.

**33. What is a p-value, and what was yours?**
**Answer:** A p-value indicates the probability of obtaining the observed results if the null hypothesis is true. My p-values were typically around 0.60 to 0.80. Since they were $> 0.05$, I failed to reject the null hypothesis, meaning the angle filter did not provide a statistically significant edge over the baseline after costs.

**34. What is Cliff’s Delta?**
**Answer:** Cliff’s Delta is a non-parametric measure of effect size. While a p-value tells you if a difference exists, Cliff's Delta tells you the *magnitude* of that difference. My Cliff's Delta was near zero, proving the economic impact of the filter was negligible after friction.

**35. What is the Rank-Biserial Correlation?**
**Answer:** It's another measure of effect size for the Wilcoxon test, indicating the proportion of favorable vs. unfavorable paired differences. Again, it showed no meaningful divergence between the baseline and the proposed strategy.

**36. Why is distinguishing statistical significance from economic significance important?**
**Answer:** A strategy might be statistically significant (e.g., it predicts direction 53% of the time with $p < 0.01$), but if the average trade profit is 0.05% and the transaction cost is 0.10%, it is economically unviable. My paper highlighted this exact trap.

**37. What is Publication Bias?**
**Answer:** It's the tendency for researchers and journals to only publish positive, profitable results while burying negative results. This leads to a false perception that everything works. I authored my paper to actively combat this by rigorously reporting a negative result.

**38. What is the Null Hypothesis ($H_0$) in your project?**
**Answer:** The null hypothesis was that there is no difference in the Out-of-Sample Net PnL between the Standard EMA crossover strategy and the Normalized Angle filtered strategy after accounting for transaction costs.

**39. Did you Reject or Fail to Reject the Null Hypothesis?**
**Answer:** I failed to reject the null hypothesis. The empirical data showed no statistically significant outperformance.

**40. If the strategy lost money, why write an academic paper on it?**
**Answer:** Because the *methodological fix* (resolving the dimensional flaw of nominal angles) is highly valuable to the data science community. Proving that an old heuristic fails under rigorous conditions saves future researchers time and capital.

---

## Category 5: Python & Software Engineering

**41. Which Python libraries did you use and why?**
**Answer:** I used Pandas for heavy lifting and vectorized time-series manipulation, NumPy for fast mathematical transformations (arctangent, log returns), and Matplotlib for plotting equity curves.

**42. What is Vectorization in Pandas and why is it important for backtesting?**
**Answer:** Vectorization means performing operations on entire arrays at once rather than using `for` loops. It is crucial because iterating through 100,000 rows in Python is painfully slow, whereas vectorized Pandas operations run in optimized C code instantly.

**43. How did you calculate the EMA in Python?**
**Answer:** I used Pandas' built-in Exponential Moving Window function: `df['close'].ewm(span=period, adjust=False).mean()`.

**44. How did you generate Buy/Sell signals without loops?**
**Answer:** I created boolean Series for conditions (e.g., `fast_ema > slow_ema`). I used `shift(1)` to identify the exact crossover bar, and then multiplied the boolean arrays to generate entry signals instantly across the entire dataset.

**45. How did you handle NaN values caused by moving average lookbacks?**
**Answer:** I used `dropna()` on the indicator columns before passing the dataframe into the execution engine to ensure the algorithm didn't crash or attempt to trade on null data at the start of the series.

**46. What is the difference between `loc` and `iloc` in Pandas?**
**Answer:** `loc` is label-based indexing (e.g., selecting by a specific date index), while `iloc` is integer-position based indexing (e.g., selecting the 5th row). I primarily used integer indexing when tracking specific trade entries and exits in the execution loop.

**47. How did you log your experiments?**
**Answer:** I implemented a structured logging approach, exporting run configurations, optimal parameters, and performance metrics into JSON and CSV files. This ensured reproducibility for my academic paper.

**48. Why is Python preferred for Quant Finance research?**
**Answer:** Python offers an unmatched ecosystem of data manipulation (Pandas), statistical testing (SciPy), and machine learning (Scikit-Learn). It allows for rapid prototyping of algorithmic logic before potentially rewriting execution engines in C++ for live low-latency trading.

**49. How did you calculate Maximum Drawdown in Python?**
**Answer:** I calculated a running cumulative maximum of the equity curve (`cummax()`). I then subtracted the current equity from the cumulative maximum, divided by the cumulative maximum, and found the absolute lowest point (`min()`) of that drawdown series.

**50. What is a Lambda function and did you use it?**
**Answer:** A lambda is a small anonymous function. I used them occasionally within `apply()` operations in Pandas when I needed to quickly transform a specific column without defining a standalone function block.

---

## Category 6: Behavioral & Technical Deep Dives

**51. Tell me about a time you realized your code had a bug. How did you fix it?**
**Answer:** Initially, my backtest showed a 95% win rate. I immediately knew this was impossible. I debugged the execution logic and realized I had a look-ahead bias: I was generating signals at the close and executing them at that exact same closing price. I refactored the engine to strict T+1 logic (execute on the next open), and the results normalized.

**52. How would you deploy this algorithm in a live environment?**
**Answer:** I would need to wrap the strategy logic in an asynchronous event-driven framework (like `asyncio`), connect it to a broker's Websocket API for real-time tick data, and implement rigorous risk-management circuit breakers.

**53. Can you explain your parameter grid size?**
**Answer:** I tested 1,440 combinations. This was derived from stepping through Fast EMAs (e.g., 5-20), Slow EMAs (20-50), and Angle Thresholds (0.01 to 0.15 degrees). 

**54. What is the computational complexity (Big O) of your backtest?**
**Answer:** Thanks to Pandas vectorization, the indicator generation is $O(N)$ where N is the number of rows. The Grid Search wrapper is $O(N \\times P)$ where P is the number of parameter combinations. It runs highly efficiently.

**55. Why didn't you use deep learning for this?**
**Answer:** Deep learning (like LSTMs) requires massive datasets and is prone to extreme overfitting in financial time series due to high noise-to-signal ratios. I wanted to build a transparent, interpretable mathematical model first.

**56. How do you handle changing market regimes (e.g., Bull vs. Bear)?**
**Answer:** My current engine doesn't explicitly filter for regimes. If I extend the project, I would add a Hidden Markov Model (HMM) or use a 200-day SMA macro-filter to only allow long trades during confirmed bull regimes.

**57. What are the limitations of the Sortino Ratio?**
**Answer:** It assumes that the target return (often the risk-free rate) is the only baseline that matters, and it relies on the assumption that historical downside deviation will predict future downside risk, which isn't always true in Black Swan events.

**58. Did you version control this project?**
**Answer:** Yes, I used Git for version control and pushed the final framework to my GitHub repository to maintain a history of the architectural changes, particularly when transitioning from Phase 1 (logic) to Phase 11 (empirical hardening).

**59. What did you learn about academic writing from this project?**
**Answer:** I learned how to format LaTeX documents for IEEE submissions, how to structure abstract/methodology sections, and most importantly, how to frame negative empirical results as valuable scientific critiques.

**60. In one sentence, why should I hire you based on this project?**
**Answer:** Because this project proves I have the software engineering skills to build complex data pipelines, the mathematical rigor to identify flawed logic, and the scientific honesty to prioritize accurate data over fake results.
"""

with open("Interview_Prep.md", "w", encoding="utf-8") as f:
    f.write(qa_content)
