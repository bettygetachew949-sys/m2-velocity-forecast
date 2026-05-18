# M2 Velocity Forecasting Using Econometrics & Machine Learning

## Overview
This project analyzes macroeconomic drivers of U.S. M2 velocity using econometric models and machine learning methods. The goal is to predict changes in monetary velocity and identify high-velocity regimes using macro-financial indicators.

## Data
Macroeconomic time-series data from FRED, including:
- CPI
- Unemployment rate
- Industrial production
- Real interest rates
- M2 velocity

## Methods
- Data cleaning and merging (R, tidyverse)
- Feature engineering (lags, differences)
- OLS regression (baseline econometric model)
- Logistic regression (classification: Up/Down, High/Low)
- KNN classification
- LASSO / Elastic Net (glmnet)
- PCA + dimensionality reduction
- Cross-validation and model tuning

## Key Outputs
- Directional prediction of M2 velocity changes
- High vs low velocity regime classification
- Model comparison across ML and econometric methods

## Tools
R, tidyverse, caret, glmnet, ggplot2, class, MASS

## Results
Multiple models achieved strong predictive performance in classifying M2 velocity movements, with regularized logistic regression and PCA-based models improving robustness.

## Author
Betelhem Kebede
Quantitative Economics | Econometrics | Machine Learning
