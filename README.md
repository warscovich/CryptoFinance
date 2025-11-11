# CryptoFinance: Bitcoin Market Prediction

## Project Overview
This repository hosts the team project for the Data Sciences Institute (University of Toronto) Certificate & Microcredential program. We created an end-to-end machine learning workflow that classifies the next-day Bitcoin market regime (Bullish or Bearish) so that retail investors, quantitative analysts, and financial content creators can react to rapid price swings with greater confidence. Source: https://github.com/UofT-DSI/team_project/blob/main/README.md

---

## Business Case
* **Business Value** – Early identification of bullish or bearish sentiment supports better risk management, automated trading, and content scheduling by surfacing actionable signals instead of raw price movements.

* **Stakeholders** – Individual investors, algorithmic traders, crypto research desks, and financial media outlets seeking objective guidance for volatile Bitcoin markets.

### Guiding Questions
* **Who is the intended audience for your project?** – Retail investors, algorithmic traders, crypto analysts, and financial media seeking reliable insights.
* **What is the question you will answer with your analysis?** – “Given today’s market state, will Bitcoin close higher or lower tomorrow?”
* **What are the key variables and attributes in your dataset?** – Daily OHLCV fields (open, high, low, close, volume), engineered technical indicators (moving averages, RSI, MACD, Bollinger Band widths, volatility lags), and weekly qualitative sentiment scores aligned to each trading day.
* **Do you need to clean your data, and if so what is the best strategy?** – Yes. We remove duplicate timestamps, forward-fill small sentiment gaps, validate numeric ranges, and checked for extreme outliers before scaling.
* **How can you explore the relationships between different variables?** – Time-series plots, rolling correlation heatmaps, pairwise scatterplots, SHAP-based feature attributions, and confusion matrix diagnostics across temporal slices.
* **What types of patterns or trends are in your data?** – The data reveals several key patterns and trends:
- 1. Sentiment Alignment and Divergence: Most weeks show alignment between sentiment and price change, e.g., bullish sentiment during price increases. However, divergence occurs frequently — some bullish weeks end with price declines, and bearish weeks sometimes see gains. This suggests sentiment alone doesn't always predict direction accurately.
- 2. Momentum and Reversals: Large positive or negative price swings often cluster for several weeks, indicating momentum phases. These periods are often followed by sharp reversals, potentially driven by overbought/oversold conditions or external events.
- 3. Sentiment Volatility: Sentiment switches between bullish and bearish frequently, showing high week-to-week variability, which may reflect changing narratives or market uncertainty.
- 4. Disagreement Insights: The Agreement_Label column shows whether the sentiment matched actual outcomes. Frequent disagreement suggests sentiment may lag market movements or reflect biased interpretations. Tracking these disagreements can help refine the predictive power of sentiment labels.
* **Are there any specific libraries or frameworks that are well-suited to your project requirements?** – pandas for wrangling, NumPy for numerical routines, scikit-learn for classical models and preprocessing, XGBoost for gradient boosting, TensorFlow/Keras for LSTM modeling, MLflow for experiment tracking, and Databricks for scalable execution.

## Formulating the ML Problem
* **Prediction Task** – Binary classification of the following day’s market direction (Bullish vs. Bearish).
* **Success Criteria** – Achieve at least 65% directional accuracy on hold-out data while maintaining interpretable drivers of the prediction.
* **Operational Constraints** – Daily batch inference must complete within minutes and produce MLflow-tracked artefacts deployable from Databricks.

## Gathering Data
1. **Historical Pricing** – Daily OHLCV Bitcoin prices (Nov 2020 – Nov 2025) from `data/raw/dataset.csv`.
2. **Weekly On-chain Sentiment** – Curated dataset (`data/raw/weekly_on_chain_sentiment.csv`) derived from web-scraped on-chain analysis blogs. Each summary was scored as Bullish/Bearish using ChatGPT with advanced/few-shot prompting to standardize sentiment labels.

## Merging and Preparing Data
* Converted raw files into a unified time series indexed by trading day.
* Forward-filled weekly sentiment to align with daily observations, then merged with technical indicators.
* Created a binary target where **Bullish** indicates the closing price exceeds the previous day’s close, otherwise **Bearish**.
* Split the data into training, validation, and test sets using chronological order to prevent look-ahead bias.

## Data Analysis and Visualization
* Conducted exploratory analysis within `notebooks/btc_price_prediction.ipynb` to understand price trends, volatility clusters, and volume regimes.
* Visualized correlations between engineered indicators, sentiment, and returns to identify dominant drivers.
* Plotted class balance and temporal drift to confirm the necessity of regular retraining.

## Expanding the Dataset
* Applied web scraping scripts (documented in the notebooks) to ingest weekly on-chain commentaries from 2020–2025.
* Employed ChatGPT-driven summarization and sentiment scoring prompts to generate consistent Bullish/Bearish labels.
* Validated sentiment coverage against the price timeline to ensure minimal gaps before merging into the master dataset.

## Feature Engineering
* Generated technical analysis metrics including moving averages, RSI, MACD, Bollinger Bands, and volume-derived oscillators.
* Lagged returns and volatility estimates to capture momentum and mean-reversion effects.
* One-hot encoded sentiment labels and constructed interaction terms between sentiment and price momentum.
* Scaled numerical features with standardization where appropriate and persisted preprocessing parameters for reuse.

### Machine Learning Guiding Questions
* **What are the specific objectives and success criteria for your machine learning model?** – The objective is to flag next-day price jumps of at least 1%; success is judged by hold-out accuracy, ROC-AUC, and class-level precision/recall recorded in the notebooks (e.g., XGBoost reached 0.7178 accuracy with ROC-AUC 0.5866 on the reserved test window, while Logistic Regression achieved 0.6963 accuracy).
* **How can you select the most relevant features for training?** – The current workflow relies on domain-driven feature lists of spreads, momentum, and volatility indicators, then inspects model coefficients/feature importances logged through MLflow to decide which engineered metrics warrant retention.
* **Are there any missing values or outliers that need to be addressed through preprocessing?** – Rolling calculations and the shifted target create leading NaNs that are removed with `dropna`, and the OBV seed is filled with zero; no additional outlier clipping is applied in the current experiments.
* **Which machine learning algorithms are suitable for the problem domain?** – Baseline Logistic Regression, Random Forest, and XGBoost classifiers capture tabular signal interactions, while an LSTM sequence model ingests 10-day windows to learn temporal dependencies.
* **What techniques are available to validate and tune the hyperparameters?** – Models are evaluated on a chronological 80/20 split with MLflow autologging so manual hyperparameter adjustments are tracked; the LSTM leverages a 20% validation split with early stopping to prevent overfitting, and Databricks experiments can be rerun with altered settings for comparative analysis.
* **How should the data be split into training, validation, and test sets?** – Use `train_test_split` with `shuffle=False` to hold out the most recent 20% of observations for testing, and rely on the Keras `validation_split=0.2` argument during LSTM training to carve out an in-sample validation fold for early stopping.
* **Are there any ethical implications or biases associated with the machine learning model?** – The model only observes historical market data, so there is no personal information, but it can still propagate optimism bias if traders over-trust signals with low bullish recall; documentation highlights the class imbalance (32% bullish) to discourage overconfident deployment.
* **How can you document the machine learning pipeline and model architecture for future reference?** – Version the notebooks alongside helper utilities, capture each training run with MLflow autologging on Databricks, and store scaler/estimator artifacts so that future contributors can trace parameters, metrics, and model summaries directly from the tracking UI.


---

## Acknowledgements
* Project completed by Juan Bueno, Julian Bueno, Kirti Vardhan and Vincent Van Schaik for the Data Sciences Institute (University of Toronto) Certificate team project requirement.
* We thank the following instructional team for their support and guidance: Phil Van-Lane (he/him) phil.vanlane@mail.utoronto.ca and Aditya Kulkarni (he/him) aditya.kulkarni@mail.utoronto.ca.

## Disclaimer

* This project is presented for educational and research purposes only and does not constitute financial, investment, trading, or any other form of professional advice. The models, datasets, and analyses included in this repository are experimental and should not be relied upon to make financial decisions. You are solely responsible for any actions you take based on the information or tools provided here.

* Historical Bitcoin data was sourced from Perplexity Finance. While efforts were made to ensure data quality, no guarantees are provided regarding accuracy, completeness, or suitability for any purpose. The developers and contributors assume no liability for losses or damages arising from the use, interpretation, or application of this project.

* Use discretion and consult a qualified financial professional before making investment decisions.