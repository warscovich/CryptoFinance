# CryptoFinance: Bitcoin Market Prediction

## Project Overview
This repository hosts the team project for the Data Sciences Institute (University of Toronto) Certificate & Microcredential program. We created an end-to-end machine learning workflow that classifies the next-day Bitcoin market regime (Bullish or Bearish) so that retail investors, quantitative analysts, and financial content creators can react to rapid price swings with greater confidence. Source: https://github.com/UofT-DSI/team_project/blob/main/README.md

---
## Industry Context

Bitcoin and the broader cryptocurrency market are known for their extreme volatility, decentralized structure, and strong retail investor participation. Unlike equities or FX, crypto markets trade 24/7, are heavily influenced by online sentiment, and are more prone to speculative bubbles and sharp drawdowns. This makes traditional financial models less effective and creates opportunities for alternative modeling approaches that incorporate both price action and sentiment.

In practice, machine learning models like this could support:
- Retail investors seeking timing guidance for discretionary trades.
- Content creators aiming to align publication schedules with market sentiment.
- Quant teams prototyping directional signals to supplement automated strategies.
- Research desks trying to frame narratives with evidence-based short-term forecasts.

This model differs from equity forecasting in that it does not rely on fundamentals (e.g., earnings), focuses on next-day classification (not returns), and incorporates crowdsourced sentiment, which plays a disproportionately large role in crypto price moves.

## Business Problem Clarification

### Formulating the ML Problem

**Task Type**  
Binary classification (not regression)

**Target Variable**  
Predict whether the next-day **closing price** of Bitcoin (BTC-USD) will be **higher or lower** than today’s close.  
Target classes: **Bullish** (next close > today’s close) vs. **Bearish** (next close ≤ today’s close)

**Prediction Horizon**  
1 trading day (next-day directional movement)

**Success Metric**  
**Directional accuracy** – the percentage of predictions where the model correctly classifies the next-day price movement  
Minimum threshold: **65% accuracy** on a chronologically held-out test set

**Note**  
We avoid the phrase “stock prediction,” as Bitcoin is not a stock.

**Operational Constraints** 
Daily batch inference must complete within minutes and produce MLflow-tracked artefacts deployable from Databricks.

---
## Required Libraries
## Requirements

[![MLflow](https://img.shields.io/badge/MLflow-F4AA41?logo=MLflow&logoColor=black)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff)](#)
[![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=fff)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ff8f00?logo=tensorflow&logoColor=white)](#)




## Data Sources and Documentation

### 1. Price Data

**Source**: Perplexity Finance (via web exporting)
- **Fields**: Daily Open, High, Low, Close, Volume (OHLCV)
- **Asset**: BTC-USD
- **Frequency**: Daily
- **Time Range**: November 2020 – November 2025
- **File**: `data/raw/dataset.csv`

### 2. Sentiment Data

**Source**: Weekly blog posts from leading on-chain analytics sites (e.g., Glassnode)
- **Extraction**: Web scraping scripts
- **Processing**: GPT-5 prompted summarization + sentiment labeling (Bullish/Bearish)
- **Format**: One sentiment label per week, forward-filled to each corresponding trading day
- **File**: `data/raw/weekly_on_chain_sentiment.csv`


**Source**: Daily Twitter
- **Extraction**: Kaggle
- **Processing**: Bitcoin Daily Twitter information
- **Format**: csv
- **File**: `data/raw/bitcoin_tweets_clean.csv`
---

## Risks and Unknowns

### Modeling Risks
- Overfitting due to small or unstable training sets(Is really challenging to acquire up to date news for bitcoin).
- Temporal leakage from improper splits or look-ahead bias
- Regime changes invalidating past relationships (e.g., macro events, regulations)

### Business Risks
- Over-reliance on model output without considering broader context
- Amplification of noise if used in high-frequency or automated settings
- False confidence in signals due to class imbalance (e.g., bearish class dominance)

### Ethical Risks
- Retail investors misinterpreting probabilities as guarantees
- Biased sentiment inputs due to echo chambers or influencer dominance

**Mitigation Strategies**  
- Clear documentation of limitations
- Regular retraining
- Monitoring for performance degradation
- Human-in-the-loop deployment where applicable

---

## Data Cleaning and Exploration Plan

### Expanding, Merging adn preparing the Dataset
* Applied web scraping scripts (documented in the notebooks) to ingest weekly on-chain commentaries from 2020–2025.
* Employed ChatGPT-driven summarization and sentiment scoring prompts to generate consistent Bullish/Bearish labels.
* Validated sentiment coverage against the price timeline to ensure minimal gaps before merging into the master dataset.
* Converted raw files into a unified time series indexed by trading day.
* Forward-filled weekly sentiment to align with daily observations, then merged with technical indicators.
* Created a binary target where **Bullish** indicates the closing price exceeds the previous day’s close, otherwise **Bearish**.
* Split the data into training, validation, and test sets using chronological order to prevent look-ahead bias.

### Cleaning Steps Taken

- Removed duplicate timestamps and ensured chronological consistency
- Validated numeric ranges (e.g., price > 0)
- Forward-filled weekly sentiment labels
- Dropped rows with missing or invalid values (e.g., during holidays or API gaps)
- Initialized OBV seed with zero to compute consistent cumulative values

### EDA Performed

- Plotted daily closing price, log returns, and rolling volatility
- Visualized extreme events (e.g., 2021 bull run, 2022 crash)
- Examined distribution of returns to detect fat tails and volatility clustering
- Analyzed alignment between sentiment and next-day returns (confusion matrix)
- Assessed temporal drift in sentiment frequency and signal consistency

### Model Development

### Feature Engineering
* Generated technical analysis metrics including moving averages, RSI, MACD, Bollinger Bands, and volume-derived oscillators.
* Lagged returns and volatility estimates to capture momentum and mean-reversion effects.
* One-hot encoded sentiment labels and constructed interaction terms between sentiment and price momentum.
* Scaled numerical features with standardization where appropriate and persisted preprocessing parameters for reuse.
* Ensured that time order was strictly preserved for all models to avoid data leakage.
* Verified that no rows contained null values across the dataset.
* Used cyclical transformations for month and day-of-week using sin/cos encoding to better capture seasonality and periodicity.
* For tree-based models (e.g., Random Forest), created lag features to represent sequential dependencies.


For our technical analysis, we feature-engineered 26 key technical indicators. In addition, we label-encoded sentiment classifications from GlassNode and GPT-5 data sources.
All numerical features were standardized using StandardScaler to bring all values to a similar scale. This prevents bias toward features with larger magnitudes and improves model convergence.
We trained 7 exploratory models to establish a baseline and obtained the following results:

### Model Experimentation Results

## Performance Comparison Table

| Model | Algorithm | Features | Probability Threshold | Accuracy | Recall (Bullish) |
|-------|-----------|----------|----------------------|----------|------------------|
| **1a** | LSTM | 20 | 0.40 | 0.53 | **0.44** |
| **1b** | LSTM | 20 | 0.45 | 0.62 | 0.15 |
| **2a** | LSTM | 41 | 0.20 | 0.57 | 0.25 |
| **2b** | LSTM | 41 | 0.25 | 0.70 | 0.04 |
| **3a** | LSTM | 32 | 0.30 | 0.59 | 0.26 |
| **3b** | LSTM | 32 | 0.35 | 0.70 | 0.05 |
| **4** | Random Forest | 59 | N/A | 0.70 | 0.00 |
| **5** | Gradient Boosting | 59 | N/A | **0.76** | 0.10 |
| **6** | FinBERT + CNN | N/A | 0.33 | 0.59 | 0.41 |
| **7** | FinBERT + CNN + Social Media | N/A | 0.33 | 0.63 | 0.10 |
> ⚠️ Note: The team focused on bullish recall because models were biased toward predicting bearish, failing to distinguish between both classes
## Analysis Summary

### Accuracy vs. Recall Trade-off
- **Clear pattern across LSTM models**: As probability threshold increases by just 0.05, accuracy improves by 9-13 percentage points but recall collapses by 21-29 points
- **Best accuracy**: Model 5 (Gradient Boosting) at 0.76, but with only 0.10 recall
- **Best recall**: LSTM with a 0.44 accuracy but 0.53 accuracy.
- **Sweet spot**: Model 6 (FinBERT + CNN) at 0.41, with reasonable 0.59 accuracy

### Feature and Data Impact
- **Traditional technical indicators (20-59 features)**: All LSTM models struggle with class imbalance regardless of feature count and probability threshold.
- **Sentiment-based features (FinBERT)**: Model 6 demonstrates that NLP-derived sentiment features significantly improve bullish class detection.
- **Social media integration**: Model 7 shows adding social media data improved accuracy (+4 points) but catastrophically reduced recall from 0.41 to 0.10, suggesting noise or conflicting signals

### Algorithm Comparison
- **LSTM models (1-3)**: Highly threshold-sensitive; can achieve either decent recall OR accuracy, never both simultaneously
- **Tree-based models (4-5)**: Highest accuracy but severely biased toward majority class; practically unusable for bullish prediction
- **Deep learning + NLP (6-7)**: Most promising approach, with Model 6 achieving the best balance between metrics

### Critical Insights
1. **Class imbalance**: All models default to predicting bearish class to maximize accuracy.
2. **Feature quality matters more than quantity**: Model 6 with sentiment features outperforms Model 4 with 59 technical indicators.
3. **Social media data is detrimental**: Adding social media to FinBERT features improved accuracy but destroyed recall, indicating potential overfitting or irrelevant noise
4. **No model is production-ready**: While the models are above 50/50 coin flip average, best recall is only 0.41-0.44, meaning models miss over half of bullish opportunities

### Model 6 Final Evaluation Metrics

<img width="880" height="790" alt="image" src="https://github.com/user-attachments/assets/f77bd21b-e5d3-488a-baac-6b50ddc8172a" />

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bearish | 0.69 | 0.69 | 0.69 | 207 |
| Bullish | 0.41 | 0.41 | 0.41 | 108 |
| **Accuracy** | | | **0.59** | **315** |
| **Macro avg** | 0.55 | 0.55 | 0.55 | 315 |
| **Weighted avg** | 0.59 | 0.59 | 0.59 | 315 |

### Performance Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC Score | 0.5620 |
| Test Accuracy | 59.37% |

### ROC-AUC
<img width="559" height="448" alt="image" src="https://github.com/user-attachments/assets/d2dff506-e4f8-44a9-b038-2a037b156917" />

### SHAP Analysis
<img width="763" height="940" alt="image" src="https://github.com/user-attachments/assets/44d2a0a9-787d-4fd9-affd-1b2f88756963" />


### Next steps
1. **Priority: Address class imbalance** using SMOTE or other undersampling techniques before further experimentation
2. **Ensemble approach**: Combine Model 6's sentiment analysis with Model 1a's technical analysis for potentially better balance

### Team Responsibilities
- **Kirti Vardhan**: Feature engineering and modeling experiments 
- **Julian Bueno**: Feature engineering and modeling experiments 
- **Juan Bueno**: Feature engineering and modeling experiments
- **Vincent Van Schaik**: Technical development including sentiment scraping, labeling, feature engineering, data cleaning, EDA, modeling workflows, and MLFlow on Databricks inegration. 

---
## Guiding Questions
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


### Machine Learning Guiding Questions
* **What are the specific objectives and success criteria for your machine learning model?** – The objective is to flag next-day price jumps of at least 1%; success is judged by hold-out accuracy, ROC-AUC, and class-level precision/recall recorded in the notebooks (e.g., XGBoost reached 0.7178 accuracy with ROC-AUC 0.5866 on the reserved test window, while Logistic Regression achieved 0.6963 accuracy).
* **How can you select the most relevant features for training?** – The current workflow relies on domain-driven feature lists of spreads, momentum, and volatility indicators, then inspects model coefficients/feature importances logged through MLflow to decide which engineered metrics warrant retention.
* **Are there any missing values or outliers that need to be addressed through preprocessing?** – Rolling calculations and the shifted target create leading NaNs that are removed.
* **Which machine learning algorithms are suitable for the problem domain?** – Baseline Logistic Regression, Random Forest, and XGBoost classifiers capture tabular signal interactions, while an LSTM sequence model ingests 10-day windows to learn temporal dependencies.
* **What techniques are available to validate and tune the hyperparameters?** – Models are evaluated on a chronological 80/20 split with MLflow autologging so manual hyperparameter adjustments are tracked; the LSTM leverages a 20% validation split with early stopping to prevent overfitting, and Databricks experiments can be rerun with altered settings for comparative analysis.
* **How should the data be split into training, validation, and test sets?** – Use `train_test_split` with `shuffle=False` to hold out the most recent 20% of observations for testing, and rely on the Keras `validation_split=0.2` argument during LSTM training to carve out an in-sample validation fold for early stopping.
* **Are there any ethical implications or biases associated with the machine learning model?** – The model only observes historical market data, so there is no personal information, but it can still propagate optimism bias if traders over-trust signals with low bullish recall; documentation highlights the class imbalance (32% bullish) to discourage overconfident deployment.
* **How can you document the machine learning pipeline and model architecture for future reference?** – Version the notebooks alongside helper utilities, capture each training run with MLflow autologging on Databricks, and store scaler/estimator artifacts so that future contributors can trace parameters, metrics, and model summaries directly from the tracking UI.

---
## Team Members Reflection Videos
* Julian Bueno - [Reflection](https://youtu.be/AwxNpUw-MMU)

## Acknowledgements
* Project completed by Juan Bueno, Julian Bueno, Kirti Vardhan and Vincent Van Schaik for the Data Sciences Institute (University of Toronto) Certificate team project requirement.
* We thank the following instructional team for their support and guidance: Phil Van-Lane (he/him) phil.vanlane@mail.utoronto.ca and Aditya Kulkarni (he/him) aditya.kulkarni@mail.utoronto.ca.

## Disclaimer

* This project is presented for educational and research purposes only and does not constitute financial, investment, trading, or any other form of professional advice. The models, datasets, and analyses included in this repository are experimental and should not be relied upon to make financial decisions. You are solely responsible for any actions you take based on the information or tools provided here.

* Historical Bitcoin data was sourced from Perplexity Finance. While efforts were made to ensure data quality, no guarantees are provided regarding accuracy, completeness, or suitability for any purpose. The developers and contributors assume no liability for losses or damages arising from the use, interpretation, or application of this project.

* Use discretion and consult a qualified financial professional before making investment decisions.
