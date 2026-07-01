# UK Road Accident Severity Prediction

Machine learning pipeline to predict collision severity (Fatal / Serious / Slight) from UK Department for Transport (DfT) Road Casualty Statistics, built as coursework for the Big Data module and as an independent ML portfolio project.

## Dataset

- Source: DfT Road Casualty Statistics (collision file), 2019–2023
- 503,373 rows after cleaning, 44 raw columns → 30 final columns after feature engineering
- Class distribution is heavily imbalanced: **83% Slight, 15% Serious, 2% Fatal**

## Data Cleaning

- Identified and removed 5 data leakage columns (e.g. fields that directly restated the target) — caught after an initial run produced suspiciously high accuracy
- Dropped 10 identifier columns with no predictive value, 4 redundant "historic" columns, and 4 high-missingness columns
- Replaced sentinel values (-1) with NaN across 15 columns; dropped rows with missing values in critical fields (speed limit, light/weather conditions, urban/rural, coordinates — under 0.05% of data lost); imputed remaining fields with column mode

## Feature Engineering

- Extracted hour, month, and season from timestamp fields
- Built 7 binary risk flags: is_rush_hour, is_weekend, is_dark, is_bad_weather, is_highspeed, is_urban, is_junction, is_hazard
- Final feature set: 30 columns

## Preprocessing

- StandardScaler applied to 7 continuous columns (longitude, latitude, speed limit, hour, month, number of vehicles, number of casualties)
- Stratified 80/20 train/test split to preserve class distribution (random_state=42)

## Handling Class Imbalance

Used `class_weight='balanced'` in both classifiers rather than synthetic oversampling (SMOTE), based on the reasoning that generating synthetic Fatal samples felt inappropriate for government safety-reporting data, particularly given the difficulty of interpolating geographic coordinates. This is a defensible but not definitive choice — SMOTE-Tomek or ADASYN are noted as the first thing to try in any extension of this project.

## Models

**Logistic Regression** (baseline) — `max_iter=1000`, `class_weight='balanced'`
**Random Forest** (main model) — `n_estimators=100`, `class_weight='balanced'`, `min_samples_leaf=5`, `max_depth=20`
**PySpark MLlib multinomial Logistic Regression** — full pipeline rebuilt in Apache PySpark 4.1.1 to demonstrate scalability to the full dataset

## Results

| Model               | Accuracy | Fatal F1 | Serious F1 | Slight F1 |
| ------------------- | -------- | -------- | ---------- | --------- |
| Logistic Regression | 50%      | 0.07     | 0.27       | 0.67      |
| Random Forest       | 66%      | 0.11     | 0.35       | 0.79      |
| PySpark MLlib LR    | 76.66%   | –        | –          | –         |

**Note on the PySpark result:** the higher accuracy is not evidence of a better model. MLlib's regularization and class-weighting behavior differ from scikit-learn's, causing the Spark model to lean more heavily toward the majority class — closer to a naive majority-class predictor than a genuinely stronger classifier. This is flagged deliberately: raw accuracy on an imbalanced dataset is a misleading metric on its own. The PySpark section's purpose was to demonstrate the pipeline scales to a distributed engine, not to outperform the scikit-learn models.

**Fatal class detail (Random Forest):** Precision 0.08, Recall 0.19, F1 0.11. Logistic Regression achieves higher Fatal recall (0.63) but at precision of only 0.04 — meaning it flags "Fatal" on the majority of collisions and is right roughly 1 in 25 times. Random Forest trades recall for precision, producing fewer but more trustworthy Fatal predictions, and is the recommended model of the two for this task.

Fatal prediction remains difficult across both models — consistent with the literature on severity prediction at this level of class imbalance (2% prevalence).

## Feature Importance

Top predictors (Random Forest): number_of_casualties, longitude, latitude, speed_limit, number_of_vehicles. Engineered flags is_dark, is_highspeed, and is_urban also contributed meaningfully.

## Tech Stack

Python, Pandas, Scikit-learn, Apache PySpark (MLlib), Matplotlib, Seaborn

## How to Run

[fill in: e.g. `pip install -r requirements.txt`, then open and run the notebook — add your actual steps here]

## Future Work

- SMOTE-Tomek or ADASYN as alternatives to class weighting for the Fatal class
- Cost-sensitive boosting (XGBoost/LightGBM with focal loss)
- Additional features: vehicle type, driver age band, real-time traffic flow data
- Calibrated probabilistic output for threshold-based operational use (e.g. flag any collision above a chosen Fatal-probability threshold, rather than single-class prediction)

## Acknowledgment

The written report accompanying this project was co-authored with Raman Bade as part of Big Data module coursework. The data pipeline, modeling, and analysis in this repository are independent work.
