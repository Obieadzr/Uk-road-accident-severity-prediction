# UK Road Accident Severity Prediction

Machine learning pipeline to predict collision severity (Fatal / Serious / Slight) from UK Department for Transport (DfT) Road Casualty Statistics, built as coursework for the Big Data module and as an independent ML portfolio project. Deployed as a live interactive demo.

**[Live Demo](https://uk-road-accident-severity-prediction.streamlit.app/)**

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
- Built 10 binary risk flags: is_rush_hour, is_weekend, is_dark, is_bad_weather, is_highspeed, is_urban, is_junction, is_hazards
- Final feature set: 28 columns feeding the model

## Preprocessing

- StandardScaler applied to 7 continuous columns (longitude, latitude, speed limit, hour, month, number of vehicles, number of casualties)
- Stratified 80/20 train/test split to preserve class distribution (random_state=42)
- **Caught a second leakage bug during development:** an earlier version of the pipeline fit the scaler on the full dataset *before* splitting into train/test, meaning the scaler's mean/std were computed with knowledge of test data. Fixed by re-ordering the pipeline — split first, fit scaler on train only, transform test with that fitted scaler — and retrained all models on the corrected data. All results below reflect the corrected, leakage-free pipeline.
- **Caught a third bug post-deployment:** the saved `scaler.pkl` used by the live app had been silently corrupted — a notebook cell that fits `StandardScaler` had been re-run a second time on already-scaled data (without a kernel restart), producing an identity transform (mean≈0, scale=1) that passed raw, unscaled values straight through. Since XGBoost's decision splits were learned on properly scaled training data, feeding it unscaled live inputs pushed predictions into structurally unfamiliar regions of the trees — a mundane, low-risk test case (daylight, dry roads, 30mph, no junction) was misclassified as Fatal with 99.67% confidence. Fixed by restarting the kernel and re-running the scaling step exactly once; verified by checking `scaler.mean_` / `scaler.scale_` directly before reuse, and by re-testing the same input (correctly predicted Slight, P(Fatal)=0.006, after the fix).

## Handling Class Imbalance

Initial approach used `class_weight='balanced'` in scikit-learn classifiers rather than synthetic oversampling, on the reasoning that generating synthetic Fatal samples felt inappropriate for government safety-reporting data, particularly given the difficulty of interpolating geographic coordinates.

This was revisited: `class_weight='balanced'` alone was not sufficient — Random Forest's Fatal-class recall remained low (0.19-0.20) regardless. Two further techniques were applied and evaluated:

- **SMOTE** oversampling on the training set only (never on test data), used with XGBoost
- **Threshold tuning** on predicted probabilities — since accuracy and even F1-score are misleading on a 2%-prevalence minority class, the decision threshold for calling a case "Fatal" was tuned directly against a target recall using `precision_recall_curve`, rather than relying on the model's default 0.5-style cutoff

This reflects a deliberate framing of the problem as **cost-sensitive classification**: a missed fatal accident (false negative) is judged far more costly than a false alarm (false positive), so the final model is tuned to prioritize Fatal-class recall over overall accuracy or precision.

## Models

**Logistic Regression** (baseline) — `max_iter=1000`, `class_weight='balanced'`
**Random Forest** — `n_estimators=100`, `class_weight='balanced'`, `min_samples_leaf=5`, `max_depth=20`
**XGBoost** (final/deployed model) — `n_estimators=3000` (early stopping, `early_stopping_rounds=50`), `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, trained on SMOTE-resampled data, `multi:softprob` objective for probability output, custom decision threshold (0.0091) applied to the Fatal class at inference time
**PySpark MLlib multinomial Logistic Regression** — full pipeline rebuilt in Apache PySpark 4.1.1 to demonstrate scalability to the full dataset

## Results

| Model                          | Accuracy | Fatal Recall | Serious Recall | Slight Recall |
| ------------------------------- | -------- | ------------ | --------------- | -------------- |
| Logistic Regression              | 50%      | 0.63         | –               | –              |
| Random Forest                    | 66%      | 0.19–0.20    | –               | –              |
| XGBoost + SMOTE (default cutoff) | 76%      | 0.01–0.02    | 0.05–0.08       | 0.97–0.99      |
| XGBoost + single Fatal threshold | 60%      | 0.60         | 0.01            | 0.85           |
| **XGBoost + cascaded 2-stage threshold (deployed)** | 58%      | 0.60         | **0.29**        | 0.66           |

An intermediate attempt balanced all three classes equally with SMOTE (5,989 → 308,748 
for Fatal, but also inflating Serious 87,961 → 308,748 — a 3.5x distortion). This visibly 
skewed live predictions toward Serious regardless of input and was abandoned in favor of 
targeted SMOTE (oversampling only Fatal, `sampling_strategy={0: 30000}`, leaving 
Serious/Slight at their real training counts).

**Note on the PySpark result:** the higher accuracy (76.66%) is not evidence of a better model. MLlib's regularization and class-weighting behavior differ from scikit-learn's, causing the Spark model to lean more heavily toward the majority class — closer to a naive majority-class predictor than a genuinely stronger classifier. This is flagged deliberately: raw accuracy on an imbalanced dataset is a misleading metric on its own. The PySpark section's purpose was to demonstrate the pipeline scales to a distributed engine, not to outperform the other models.

**Why the deployed model has lower accuracy than Random Forest:** this is intentional, not a regression. Accuracy is dominated by the majority Slight class (83% of the data); optimizing for minority-class recall directly conflicts with it. The deployed model uses a **two-stage cascaded threshold** rather than a single cutoff:

1. Flag Fatal if `P(Fatal) ≥ 0.0255` (tuned for Fatal recall ≈ 0.60)
2. If not Fatal, flag Serious if `P(Serious) / (P(Serious) + P(Slight)) ≥ 0.2407` (tuned separately, for Serious recall ≈ 0.29–0.40), else Slight

The second stage exists because a single Fatal threshold alone left Serious almost undetectable (recall ≈ 0.01) — plain argmax between Serious and Slight almost always picks Slight, since it outnumbers Serious roughly 3.5-to-1 in the data. Rescuing Fatal from being crowded out exposed the same crowding-out problem for Serious one level down; the cascade addresses both, at a further, acknowledged cost to Slight-class recall (0.85 → 0.66).

**Takeaway:** no model in this project achieves high precision *and* high recall on Fatal, consistent with the literature on severity prediction at ~2% class prevalence. The cascaded threshold is a genuine improvement — all three classes now have non-trivial recall/F1 rather than one being effectively dead — but it is a tuned tradeoff across all three classes, not a solved problem, and the exact thresholds (0.60 / 0.29–0.40 recall targets) were chosen somewhat arbitrarily and would benefit from a more principled cost-matrix approach (see Future Work).

## Feature Importance

Top predictors (Random Forest): number_of_casualties, longitude, latitude, speed_limit, number_of_vehicles. Engineered flags is_dark, is_highspeed, and is_urban also contributed meaningfully.

## Deployment

The final XGBoost model is deployed as an interactive Streamlit app:
- Raw collision-report-style inputs (weather, light conditions, road type, etc.) via dropdowns matching the official DfT STATS19 code book
- Feature engineering and scaling pipeline replicated exactly from training (same binary-flag logic, same fitted `StandardScaler`)
- Predicts severity using the tuned Fatal-recall threshold rather than a default argmax cutoff
- Hosted on Streamlit Community Cloud — see link at top of this README

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn (SMOTE), Apache PySpark (MLlib), Streamlit, Matplotlib, Seaborn

## How to Run

**Notebooks (training pipeline):**
```
pip install -r requirements.txt
```
Run notebooks in order: data cleaning → feature engineering → scaling/split → model training (Logistic Regression, Random Forest, XGBoost) → evaluation.

**Live demo (locally):**
```
pip install -r requirements.txt
streamlit run app.py
```
Requires `Models/xgb_model.pkl`, `Models/scaler.pkl`, and `Models/feature_columns.pkl` to be present (generated by the training notebooks).

## Limitations

- Fatal-class prediction remains fundamentally difficult given ~2% class prevalence; no model tested achieves both usable precision and recall simultaneously on this class
- SMOTE generates synthetic training examples for the Fatal class — these may not fully represent the diversity of real fatal accidents, and the ~50x oversampling ratio used here carries real risk of overfitting to synthetic patterns
- The deployed model's low Fatal precision (~0.03-0.05) means the large majority of "Fatal" predictions are false alarms; this tradeoff was a deliberate choice for this project but would need further work (e.g. a human-in-the-loop review step) before any real operational use
- The two-stage cascaded threshold improves Serious recall substantially (0.01 → 0.29) but at a real cost to Slight-class recall (0.85 → 0.66) and overall accuracy (66% → 58%) — every class's performance here is a tuned tradeoff, not simultaneously optimized
- Threshold values (0.0255, 0.2407) and their target recalls (0.60, 0.40) were chosen somewhat heuristically rather than via a formal cost-benefit/cost-matrix analysis — a more rigorous approach is listed under Future Work

## Future Work

- Replace the two heuristically-chosen threshold targets with a formal cost-matrix approach (e.g. explicitly weighting the cost of each misclassification type) rather than picking target recalls somewhat arbitrarily
- SMOTE-Tomek or ADASYN as alternatives to plain SMOTE for the Fatal class
- Cost-sensitive boosting with explicit per-class cost matrices, rather than post-hoc threshold tuning
- Additional features: vehicle type, driver age band, real-time traffic flow data
- Model monitoring / drift detection if ever moved beyond a portfolio demo toward real operational use