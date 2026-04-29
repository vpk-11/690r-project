# Member 2 Contribution
## CS690R — Clinical Biomarker Extraction via Bio-PM
### Random Forest Analysis & Ablation Study — IRB Stroke Recovery Dataset

---

## Where Member 2 Fits in the Pipeline

The project has three members each handling a distinct stage of the pipeline:

```
Member 1                    Member 2                    Member 3
─────────────────           ─────────────────────       ─────────────────────
Raw accelerometer     →     Random Forest on      →     ICC reliability
data fed into               feature matrix              Clinical correlation
Bio-PM model                                            Longitudinal tracking
                                                        Report writing
Outputs:
biopm_features.npz
(198 × 228 matrix)
```

Member 1 produced the feature matrix. Member 3 validated the biomarkers statistically and clinically. Member 2's job — sitting squarely in the middle — was to answer: **which of these 228 features actually discriminate healthy from stroke, and how robustly?**

---

## What Member 1 Handed Off

Member 1's `biomarker_feature_extraction.ipynb` processed 24-hour wrist accelerometer recordings from the IRB stroke recovery dataset through the Bio-PM model and computed kinematic features. The output was `features/biopm_features.npz` with the following structure:

| Key | Shape | Description |
|---|---|---|
| `features` | (198, 228) | Main feature matrix — one row per clinical visit |
| `features_even` | (198, 228) | Even 30-min blocks (for Member 3's ICC analysis) |
| `features_odd` | (198, 228) | Odd 30-min blocks (for Member 3's ICC analysis) |
| `feature_names` | (228,) | Name of each feature column |
| `labels` | (198,) | 0 = healthy, 1 = stroke |
| `pids` | (198,) | Subject ID for LOSO grouping |
| `arat` | (198,) | ARAT clinical score (0–57) |
| `fma` | (198,) | FMA-UE clinical score (0–66) |
| `weeks` | (198,) | Visit week number |

The 228 features are kinematic statistics computed from the Bio-PM embeddings — things like the mean, standard deviation, percentiles (p10, p50, p90), IQR, and max of acceleration, velocity, and jerk across the X, Y, and Z axes.

**Important context from Member 1:** The baseline logistic regression on the 228-d matrix yielded AUC of 0.514 — essentially random. Member 1 noted this is expected given the severe class imbalance (194 stroke vs. 4 healthy visits) and that ARAT/FMA regression is the more meaningful downstream analysis. Member 2's Random Forest with `class_weight='balanced'` addresses this directly.

---

## What Member 2 Did

### Notebook 1: Base Random Forest (`member2_random_forest_irb.ipynb`)

**The core question:** Can a Random Forest trained on the 228 kinematic features reliably distinguish healthy from stroke visits, even with severe class imbalance?

#### LOSO Cross-Validation

Leave-One-Subject-Out (LOSO) cross-validation was used throughout. In each fold, all visits from one subject are held out as the test set and the model is trained on all remaining subjects. With 30 subjects, this produces 30 folds.

LOSO was chosen over a random train/test split because the same subject has multiple visits across weeks. A random split could put visit 1 of a subject in training and visit 3 in test — the model would learn that person's movement style rather than generalizing to unseen people. LOSO guarantees the test subject is completely unknown to the model in every fold, matching the real clinical deployment scenario.

StandardScaler was fit only on training data inside each fold to prevent data leakage.

#### Random Forest Setup

```python
RandomForestClassifier(
    n_estimators  = 200,
    class_weight  = 'balanced',
    random_state  = 42,
    n_jobs        = -1,
)
```

`class_weight='balanced'` was critical — without it, the 194:4 stroke:healthy imbalance causes the classifier to predict stroke for everything. AUC is reported on pooled predictions across all folds, not averaged per fold, because single-visit subjects produce single-class test folds where per-fold AUC is undefined.

#### Results

| Metric | Value |
|---|---|
| Pooled AUC | **0.780** |
| Pooled F1 | 0.990 (inflated by imbalance — AUC is the reliable metric) |

#### Feature Importances

Feature importances were computed as Mean Decrease in Impurity (MDI), averaged across all 200 trees in all 30 LOSO folds. This means a feature had to be consistently useful across every subject's held-out fold to rank highly — not just lucky in one fold.

**Top finding:** 19 of the top 20 features by importance are Y-axis features (vertical arm motion). The top feature, `Y_acc_std_p10`, had an importance of 0.0417 — significantly ahead of the second-ranked feature at 0.0315.

#### Outputs

| File | Description |
|---|---|
| `results/metrics/rf_loso_results_irb.csv` | Pooled AUC, F1, per-fold breakdown |
| `results/metrics/rf_feature_importances_irb.csv` | All 228 features ranked by importance |
| `results/figures/rf_feature_importance_irb.png` | Top 20 feature importance bar chart |
| `results/figures/rf_biomarker_distributions_irb.png` | Violin plots of top 6 biomarkers |

---

### Notebook 2: Ablation Study (`member2_ablation_study_irb.ipynb`)

Five ablation experiments were run to understand what drives the AUC=0.780 baseline and whether a simpler configuration could outperform it.

All ablations use identical LOSO cross-validation for fair comparison.

---

#### Ablation 1 — Feature Group (Axis)

**Question:** Which axis contributes most to classification?

| Group | N Features | AUC | vs Baseline |
|---|---|---|---|
| **Y-axis only** | — | **0.883** | **+0.103** |
| All axes | 228 | 0.780 | — |
| Z-axis only | — | 0.742 | -0.038 |
| X-axis only | — | 0.624 | -0.156 |
| Magnitude only | — | 0.622 | -0.158 |

**Finding:** Y-axis features alone outperform all 228 features combined by 10.3 AUC points. X and Z axis features add noise when combined with Y-axis features, actively hurting performance. The clinical interpretation is that **vertical arm motion (Y-axis) is the primary movement signature of stroke** — stroke survivors show reduced and irregular upward arm movement compared to healthy controls.

---

#### Ablation 2 — Feature Type (Statistic)

**Question:** Do percentile features carry more signal than means, standard deviations, or IQR?

| Statistic Type | AUC | vs Baseline |
|---|---|---|
| **Percentiles (p10/p50/p90)** | **0.793** | **+0.013** |
| All types | 0.780 | — |
| Std | 0.741 | -0.039 |
| Mean | 0.653 | -0.127 |
| IQR | 0.635 | -0.145 |

**Finding:** Percentile features slightly outperform the full feature set and are substantially better than means or IQR alone. This suggests the **distribution shape of movement across the day** — captured by the 10th, 50th, and 90th percentiles — is more informative than simple summary statistics. Stroke patients likely show more extreme low-end values (reduced activity) that percentiles capture better than means.

---

#### Ablation 3 — Number of Trees

**Question:** Was 200 trees necessary, or does performance plateau earlier?

| N Trees | AUC | vs Baseline |
|---|---|---|
| 10 | 0.551 | -0.229 |
| 25 | 0.619 | -0.161 |
| 50 | 0.706 | -0.075 |
| **100** | **0.780** | **+0.000** |
| 150 | 0.783 | +0.003 |
| 200 | 0.780 | — |
| 300 | 0.785 | +0.005 |
| 500 | 0.787 | +0.007 |

**Finding:** Performance plateaus at 100 trees. Going from 100 to 200 trees provides no improvement. Going to 500 trees adds only 0.007 AUC at double the compute cost. **100 trees is sufficient** — this halves training time with no performance penalty.

---

#### Ablation 4 — Top-K Features

**Question:** How few features are needed to maintain near-baseline performance?

| Top-K Features | AUC | vs Baseline |
|---|---|---|
| **5** | **0.959** | **+0.179** |
| 10 | 0.957 | +0.177 |
| 20 | 0.934 | +0.154 |
| 30 | 0.930 | +0.150 |
| 50 | 0.867 | +0.087 |
| 75 | 0.883 | +0.103 |
| 100 | 0.822 | +0.041 |
| 150 | 0.833 | +0.053 |
| 228 | 0.650 | -0.130 |

**Finding:** This is the most striking result of the entire analysis. Using only the **top 5 features** by importance gives AUC of **0.959** — far exceeding the baseline of 0.780. Performance actually *decreases* as more features are added, with all 228 features dropping to 0.650.

This is the **curse of dimensionality**: with only 198 samples and 228 features, the Random Forest gets confused by irrelevant features that outnumber the informative ones. The top 5 Y-axis features contain nearly all the discriminative signal. This is a strong finding for a clinical tool — you only need to measure 5 things rather than 228.

---

#### Ablation 5 — Classifier Comparison

**Question:** Was Random Forest the right classifier?

| Classifier | AUC | vs Baseline |
|---|---|---|
| **Random Forest (200)** | **0.780** | **—** |
| Logistic Regression | 0.682 | -0.098 |
| Gradient Boosting | 0.280 | -0.500 |
| SVM (RBF kernel) | 0.139 | -0.641 |

**Finding:** Random Forest is clearly the right choice. Gradient Boosting and SVM perform catastrophically — likely because they are more sensitive to the 194:4 class imbalance and do not handle it as effectively as Random Forest's `class_weight='balanced'` parameter. Logistic Regression is reasonable but cannot capture the non-linear decision boundaries that Random Forest can.

---

#### Optimal Configuration

Combining the best findings from all ablations:

| Setting | Baseline | Optimal |
|---|---|---|
| Features | All 228 | Top 5 (Y-axis) |
| N Trees | 200 | 100 |
| Classifier | Random Forest | Random Forest |
| **AUC** | **0.780** | **~0.959** |

---

## How Member 2's Work Connects to Member 3

Member 3's `analysis.ipynb` ran ICC reliability, Spearman clinical correlation, PCA biomarker space, and longitudinal sensitivity analysis on the same 228-d feature matrix.

**The Y-axis vs Z-axis divergence is the most interesting cross-member finding:**

- **Member 2 (Random Forest importance):** Y-axis features dominate — `Y_acc_std_p10`, `Y_jerk_std_p10`, etc. are the best classifiers of healthy vs. stroke.
- **Member 3 (Spearman correlation):** Z-axis features dominate — `Z_jerk_mean_iqr` (ρ=0.766), `Z_acc_mean_iqr` (ρ=0.750) have the strongest correlations with ARAT and FMA clinical scores.

These findings are complementary, not contradictory. They answer different clinical questions:

| Clinical Question | Best Features | Method |
|---|---|---|
| "Does this patient have a motor impairment?" | Y-axis features | RF importance (Member 2) |
| "How severely impaired is this patient?" | Z-axis features | Spearman correlation (Member 3) |

In a real clinical deployment, both would be used: Y-axis features for screening/diagnosis, Z-axis features for monitoring recovery severity over time.

---

## Output Files Summary

| File | Type | Description |
|---|---|---|
| `member2_random_forest_irb.ipynb` | Notebook | Base RF analysis |
| `member2_ablation_study_irb.ipynb` | Notebook | All 5 ablations |
| `results/metrics/rf_loso_results_irb.csv` | CSV | Pooled AUC, F1 |
| `results/metrics/rf_feature_importances_irb.csv` | CSV | 228 features ranked |
| `results/metrics/ablation1_axis_irb.csv` | CSV | Axis ablation results |
| `results/metrics/ablation2_stattype_irb.csv` | CSV | Stat type ablation results |
| `results/metrics/ablation3_ntrees_irb.csv` | CSV | N trees ablation results |
| `results/metrics/ablation4_topk_irb.csv` | CSV | Top-K ablation results |
| `results/metrics/ablation5_classifier_irb.csv` | CSV | Classifier ablation results |
| `results/figures/rf_feature_importance_irb.png` | Figure | Top 20 feature importance bar chart |
| `results/figures/rf_biomarker_distributions_irb.png` | Figure | Violin plots of top 6 biomarkers |
| `results/figures/ablation_summary_irb.png` | Figure | All 5 ablations in one grid |
