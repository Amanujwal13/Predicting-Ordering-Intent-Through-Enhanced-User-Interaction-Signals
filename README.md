# Predicting Ordering Intent Through Enhanced User Interaction Signals

## Project overview
This project predicts whether an online shopping session results in a purchase (`MadePurchase`) using user interaction and session-context features from the Online Shoppers Intention dataset.

The notebook compares three classification models:
- Logistic Regression
- Random Forest
- Gradient Boosting

The modeling workflow uses a 60% training set, 20% validation set, and 20% untouched test set. Classification thresholds are selected on the validation set by maximizing F1 and then locked before final test evaluation.

## Project files
```text
project/
|-- data/
|   `-- online_shoppers_intention.csv
|-- ml.ipynb
|-- README.md
`-- Predicting_Ordering_Intent_Summary_Report.pdf
```

The notebook also supports placing `online_shoppers_intention.csv` in the project root as a fallback.

## Dataset summary
- Original dataset size: 12,330 rows x 18 columns
- After duplicate removal: 12,205 rows x 18 columns
- Predictor columns: 17
- Target: `MadePurchase`
- Original target distribution: 84.53% no purchase, 15.47% purchase
- Missing values observed in the dataset: none
- Duplicate rows removed: 125

### Feature groups
Numeric:
`AcctPagesViewed`, `AcctPageTime`, `InfoPagesViewed`, `InfoPageTime`, `ProductPagesViewed`, `ProductPageTime`, `AvgBounceRate`, `AvgExitRate`, `AvgPageValue`, `ProximityToSpecialDay`

Categorical:
`VisitMonth`, `UserOS`, `UserBrowser`, `UserRegion`, `SourceChannel`, `UserCategory`, `IsWeekendVisit`

## Methodology
1. Load the dataset with pandas.
2. Inspect shape, descriptive statistics, missing values, and target distribution.
3. Remove duplicate rows.
4. Separate predictors (`X`) and binary target (`y`).
5. Check the target association of `AvgPageValue` and retain it unless a temporal audit shows it is unavailable at prediction time.
6. Create categorical and numeric feature lists.
7. Split the data using stratification:
   - 60% train: 7,323 rows
   - 20% validation: 2,441 rows
   - 20% test: 2,441 rows
8. Build preprocessing pipelines using imputation and one-hot encoding; Logistic Regression additionally uses `StandardScaler`.
9. Train the three models.
10. Select one threshold per model on the validation set by maximizing F1 over thresholds from 0.10 to 0.90.
11. Lock the selected thresholds.
12. Evaluate once on the untouched test set using accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
13. Reuse the same final test predictions for confusion matrices, feature analysis, and subgroup analysis.

## Models and settings
### Logistic Regression
- `class_weight="balanced"`
- `max_iter=3000`
- `random_state=42`

### Random Forest
- `n_estimators=400`
- `class_weight="balanced"`
- `n_jobs=-1`
- `random_state=42`

### Gradient Boosting
- `n_estimators=200`
- `learning_rate=0.1`
- `max_depth=3`
- `random_state=42`
- Balanced sample weights via `compute_sample_weight`

## Final test results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 88.78% | 63.17% | 61.34% | 62.24% | 0.9110 | 0.6659 | 0.63 |
| Random Forest | 89.96% | 68.46% | 67.19% | 67.82% | 0.9229 | 0.7240 | 0.40 |
| Gradient Boosting | 89.06% | 62.53% | 68.29% | 65.38% | 0.9322 | 0.7246 | 0.68 |

## Key insights
- `AvgPageValue` is the largest Random Forest feature importance in the notebook (about 0.341).
- `AvgPageValue` has a correlation of about 0.492 with `MadePurchase`.
- `AvgBounceRate` and `AvgExitRate` are strongly correlated (about 0.902).
- `ProductPagesViewed` and `ProductPageTime` are strongly correlated (about 0.860).
- Thresholds are model-specific because probability distributions and precision-recall trade-offs differ by model.

## Important modeling note: `AvgPageValue`
The relatively strong association of `AvgPageValue` with the target is not automatically evidence of leakage. Before production use, verify that the feature is available at the exact time the purchase-intent prediction is made. If it uses information observed after or because of the purchase, remove it and retrain.

## Subgroup analysis
Random Forest subgroup analysis is performed using `UserCategory`. These groups are dataset segments and are not automatically protected demographic attributes.

| UserCategory | Test samples | False Positive Rate | False Negative Rate |
|---|---:|---:|---:|
| Returning_Visitor | 2,084 | 5.95% | 36.84% |
| New_Visitor | 344 | 3.64% | 23.71% |
| Other | 13 | 7.69% | 0.00% |

The `Other` group is very small, so its rates are unstable and should not be over-interpreted.

## How to run
### 1. Clone or copy the project
Keep the notebook and dataset in the structure shown above.

### 2. Create a virtual environment (recommended)
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 4. Launch Jupyter
```bash
jupyter notebook
```
Open `ml.ipynb` and run the cells from top to bottom.

### 5. Dataset path
Preferred:
```text
data/online_shoppers_intention.csv
```
The notebook also checks the project root if the `data/` path is not found.

## Dependencies
- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- jupyter / Jupyter Notebook

No package version is pinned in the notebook, so the dependency list above reflects the packages actually imported or required to execute the notebook.

## Notes for future improvement
- Add cross-validation and hyperparameter tuning if optimization is required.
- Consider a time-based validation/test split if deployment is temporal.
- Define the threshold objective from business cost or capacity constraints rather than F1 alone when moving toward production.
- Verify the temporal availability of `AvgPageValue` before deployment.
- Consider confidence intervals or minimum group-size rules for subgroup evaluation.
