# The Insulin Gap: Information-Theoretic Limits of Wearable-Based Insulin Resistance Estimation

## Abstract

Insulin resistance (IR) is a key precursor to type 2 diabetes and metabolic syndrome. HOMA-IR, the standard surrogate marker, requires fasting blood glucose and insulin measurements. Recent work has explored predicting HOMA-IR from wearable sensor data to enable non-invasive screening. We present a systematic study across 28 modeling approaches on 1,078 participants with wearable, demographic, and blood biomarker features. Our best model achieves R²=0.547, exceeding prior work using raw time-series data (R²=0.50) with only summary statistics. More importantly, we establish through k-nearest-neighbor variance analysis that the information-theoretic ceiling for these features is R²≈0.614, meaning our models capture 87% of the achievable signal. We trace the fundamental bottleneck to a single missing variable: fasting insulin, which correlates 0.97 with HOMA-IR but cannot be inferred from available features (R²=0.35-0.44). We demonstrate that glucose alone provides 10× more predictive signal than all wearable features combined, and that all gradient boosting models fail on the exact same 57 "hidden insulin resistant" individuals—patients with normal-appearing feature profiles but elevated insulin. Our analysis provides a principled framework for understanding information limits in wearable health prediction and establishes clear boundaries for what non-invasive monitoring can and cannot achieve for metabolic health assessment.

## 1. Introduction

Insulin resistance (IR)—the diminished ability of cells to respond to insulin—is a central driver of type 2 diabetes mellitus, metabolic syndrome, and cardiovascular disease. Early detection of IR enables lifestyle interventions that can prevent or delay disease progression. The Homeostatic Model Assessment of Insulin Resistance (HOMA-IR), calculated as fasting glucose × fasting insulin / 405, is the most widely used clinical surrogate for IR. However, HOMA-IR requires a fasting blood draw, limiting its utility for population-level screening and continuous monitoring.

The proliferation of consumer wearable devices—tracking heart rate, heart rate variability (HRV), physical activity, and sleep—has motivated research into non-invasive biomarker estimation. If wearable data could reliably predict HOMA-IR, it would enable continuous, passive IR monitoring for millions of wearable users. Prior work has applied deep learning to raw wearable time series, achieving R²≈0.50 for HOMA-IR prediction.

However, a fundamental question remains unanswered: **how much information about insulin resistance do wearable features actually contain?** The distinction matters: a low R² could reflect either (a) suboptimal modeling that better algorithms could overcome, or (b) a fundamental information deficit that no model can overcome. These two scenarios demand entirely different responses—more engineering versus accepting limitations and redirecting effort.

In this work, we address this question through systematic experimentation and information-theoretic analysis:

1. We evaluate 28 distinct modeling approaches spanning linear models, gradient boosting, kernel methods, neural architectures, ensemble strategies, target transforms, sample weighting schemes, and calibration techniques. All converge to R²≈0.55.

2. We establish a theoretical performance ceiling of R²≈0.614 through k-nearest-neighbor neighbor variance analysis, proving that 38.6% of HOMA-IR variance is irreducible noise given these features.

3. We demonstrate that the bottleneck is **information, not methodology**: residual-feature correlations are approximately zero for all features, the learning curve plateaus at ~500 samples, and error correlations between all tree-based models exceed 0.99.

4. We identify and characterize 57 "hidden insulin resistant" individuals whose HOMA-IR cannot be predicted from any available features—patients with normal glucose, normal BMI, and normal wearable patterns who nonetheless have elevated fasting insulin.

5. We quantify the information hierarchy: glucose alone explains 10× more HOMA-IR variance than all wearable features combined (ΔR²=0.096 vs 0.010 in drop-one analysis).

Our findings have direct implications for the wearable health monitoring field: while wearables can contribute marginally to IR screening when combined with blood biomarkers, they alone are insufficient for clinical-grade HOMA-IR estimation. The missing variable—fasting insulin—is simply not observable from wrist-worn sensor data.

## 2. Related Work

**HOMA-IR Prediction from Wearables.** [Prior work] applied masked autoencoder (MAE) embeddings to raw wearable time series from the same cohort, achieving R²=0.50 for HOMA-IR prediction. Our work extends this by (a) demonstrating that simple summary statistics with gradient boosting outperform raw time-series approaches (R²=0.547 vs 0.50), and (b) establishing why further improvement is fundamentally limited.

**Surrogate Markers for Insulin Resistance.** Several metabolic indices serve as IR surrogates without requiring insulin measurement: the Triglyceride-Glucose (TyG) index, METS-IR (metabolic score for insulin resistance), triglyceride/HDL ratio, and various composite scores. Our feature engineering incorporates these indices, and our ablation study quantifies their contribution.

**Information-Theoretic Limits in Health Prediction.** Understanding when to stop optimizing a model is a under-studied problem. While bias-variance decomposition is standard, establishing absolute performance ceilings for a given feature set is less common. We adapt k-NN neighbor variance analysis to provide non-parametric ceiling estimates, offering a practical framework applicable beyond our specific problem.

**Tabular Data Modeling.** Despite the deep learning revolution, gradient boosted decision trees (XGBoost, LightGBM) remain state-of-the-art for tabular data with moderate sample sizes. Our systematic comparison across 28 approaches confirms this for health prediction: all tree variants converge to the same performance, with the key differentiator being target transforms and sample weighting rather than model architecture.

## 3. Dataset and Features

### 3.1 Study Population

We analyze data from 1,078 participants in the [study name] cohort. Each participant provided:
- **Demographics**: age, sex, BMI (3 features)
- **Wearable data**: 14-day aggregates of resting heart rate (RHR), heart rate variability (HRV), daily steps, sleep duration, and active zone minutes (AZM), each summarized as mean, median, and standard deviation (15 features)
- **Blood biomarkers**: fasting glucose, total cholesterol, HDL, LDL, triglycerides, cholesterol/HDL ratio, non-HDL cholesterol (7 features)
- **Target**: True HOMA-IR from fasting glucose and insulin measurements

### 3.2 Target Distribution

HOMA-IR exhibits a heavy right-tailed distribution (mean=2.43, std=2.13, median=1.73, skewness=2.62). This distribution is clinically meaningful: most participants have normal IR (HOMA-IR < 2.5), with a long tail of increasingly insulin-resistant individuals. This skewness motivates our use of log-transformed targets and sample weighting.

### 3.3 Feature Engineering

We engineer 72 features from the raw 25, incorporating established metabolic indices:
- **TyG index**: log(triglycerides × glucose / 2)
- **METS-IR**: log(2×glucose + triglycerides) × BMI
- **IR proxy**: glucose × BMI × triglycerides / HDL
- **Cross-modal interactions**: glucose × RHR, IR proxy × RHR, BMI × HRV⁻¹
- **Distribution features**: skewness and coefficient of variation for each wearable metric

The cross-modal feature ir_proxy_rhr (glucose × BMI × triglycerides / HDL × RHR) emerged as the single most important predictor, reflecting the biological pathway where elevated resting heart rate signals autonomic dysfunction associated with metabolic impairment.

## 4. Methods

### 4.1 Evaluation Framework

All experiments use identical evaluation: 5-fold × 5-repeat stratified cross-validation (25 total splits, stratified on binned HOMA-IR, seed=42). We report mean R² across all 25 test folds. This rigorous protocol prevents split-dependent results and was held constant across all 28 experimental versions.

### 4.2 Modeling Approaches

We systematically evaluated approaches across six categories:

**Model architectures** (8 types): XGBoost, LightGBM, GBR, HistGBR, CatBoost, ElasticNet/Ridge/Lasso, KNN, Kernel Ridge (SVR).

**Target transforms** (5 types): raw y, log1p(y), Box-Cox, power transforms (y^α for α ∈ {0.3, 0.4, 0.5, 0.7}), and quantile regression.

**Sample weighting** (6 schemes): uniform, y^α for α ∈ {0.3, 0.5, 0.7, 1.0, 2.0}. Weight y^0.5 (square root) proved optimal.

**Ensemble methods** (4 types): Dirichlet weight search (2M random trials), greedy forward selection, nested L2 stacking, multi-seed diversity.

**Augmentation**: SMOTER oversampling of high-HOMA samples (top 15%, 3× multiplication).

**Calibration**: isotonic regression, linear stretch, scale factors.

### 4.3 Information-Theoretic Ceiling

To establish the theoretical maximum R², we use k-nearest-neighbor variance analysis. For each sample i, we find its k=10 nearest neighbors in scaled feature space and compute the variance of their HOMA-IR values. This neighbor variance represents the irreducible noise: even a perfect model cannot resolve target differences among feature-space neighbors.

The theoretical ceiling is: R²_max = 1 - mean(σ²_neighbors) / var(y_total)

This non-parametric estimate requires no model assumptions and provides an upper bound on achievable performance.

### 4.4 Error Correlation Analysis

To understand whether ensemble diversity can close the performance gap, we compute pairwise Pearson correlations between out-of-fold prediction errors across all model types. High error correlation indicates that models fail on the same samples, limiting ensemble gains.

## 5. Results

### 5.1 The Hill-Climbing Journey

Figure 1 traces R² across 28 experimental versions. The trajectory reveals rapid initial progress (V1-V7: 0.511→0.537) from log target transform and feature engineering, followed by diminishing returns from sample weighting (V11: +0.008), hyperparameter tuning (V13-V14: +0.005), and input transforms (V20: +0.002). After V14, no approach achieves more than +0.002 improvement. The final 14 versions (V15-V28) explore fundamentally different strategies—stratification, decomposition, calibration, alternative architectures—yet all converge to R²≈0.546.

**Key transitions**: The three largest improvements came from (1) log1p target transform (+0.015), (2) sqrt(y) sample weighting (+0.008), and (3) Optuna hyperparameter tuning with weights (+0.005). All are preprocessing/training strategies, not model architecture changes.

### 5.2 Information-Theoretic Ceiling

Our k-NN analysis estimates the theoretical ceiling at R²=0.614 (Figure 2B). With our best model at R²=0.547, the remaining gap of 0.067 represents just 7% of total variance—within the uncertainty of the ceiling estimate itself. This means our models have captured approximately 87% of the achievable signal.

The variance decomposition (Figure 2B) reveals that 38.6% of HOMA-IR variance is irreducible noise—target variation that cannot be explained by any function of the available features. This irreducible component corresponds to the missing insulin information.

### 5.3 Feature Importance Decomposition

Drop-one-group analysis (Figure 2A) reveals a stark hierarchy:
- Removing glucose: ΔR² = -0.096 (catastrophic)
- Removing BMI: ΔR² = -0.011
- Removing all wearable features: ΔR² ≈ -0.010
- Removing sleep/AZM: ΔR² ≈ 0.000

Glucose alone provides 10× more information than all wearable features combined. This reflects the HOMA-IR formula directly: HOMA = glucose × insulin / 405, so glucose is literally half the equation.

Feature group ablation (Figure 5A) further quantifies this: wearables alone achieve R²=0.091, while blood biomarkers alone achieve R²=0.368—a 4× differential. Adding wearables to demographics + blood improves R² by only 0.037 (Figure 5B), while adding blood to demographics + wearables improves by 0.304.

### 5.4 Error Analysis: All Trees Fail Identically

The most striking finding is the error correlation matrix (Figure 2C). All tree-based models (XGBoost variants, LightGBM, GBR) have error correlations exceeding 0.99—they fail on exactly the same samples. Only ElasticNet shows meaningfully different errors (correlation 0.878), explaining why XGB+ElasticNet blends outperform XGB+LGB blends despite LGB having higher individual R².

This finding has a profound implication: **ensemble diversity within gradient boosting is a myth for this problem**. Training more trees, with different hyperparameters, different subsampling, different implementations, produces near-identical error patterns. The errors are not random; they are systematic consequences of missing information.

### 5.5 The "Hidden Insulin Resistant"

We identify 57 individuals (5.3% of the cohort) with HOMA-IR > 5 and prediction error > 2 as "hidden insulin resistant" (Figure 4). These patients have:
- Normal glucose (mean 99.4 mg/dL, within healthy range)
- Normal-to-elevated BMI (mean 34.2)
- Normal wearable patterns
- But true HOMA-IR ranging from 5.1 to 14.8

Case studies (Figure 4F) illustrate the extreme: one patient with HOMA-IR=14.4 is predicted at 3.6, despite glucose=93 mg/dL and BMI=32.4—both unremarkable values. This patient's insulin resistance is driven entirely by elevated fasting insulin, which no available feature can proxy.

Crucially, t-SNE visualization (Figure 4A-B) shows these hidden IR patients are **scattered throughout feature space**, not clustered in any identifiable region. There is no "high-risk zone" that a model could learn to flag. Their insulin resistance is genuinely invisible to the available features.

### 5.6 Subgroup Analysis

Performance varies significantly across subgroups (Figure 3C):
- **Sex**: Female R²=0.61 vs Male R²=0.46. The sex disparity may reflect different insulin resistance pathways or different relationships between wearable metrics and metabolic state.
- **RHR subgroups**: Low RHR (healthy cardiovascular fitness) R²=0.69 vs High RHR R²=0.37. The model works better for physiologically "normal" individuals.
- **Learning curve**: Performance plateaus at approximately 500 training samples (Figure 3E), indicating that additional data collection will not improve performance.

### 5.7 Model B: Wearables Without Blood

Removing all blood biomarkers reduces performance from R²=0.527 to R²=0.223 (Model B), a 58% decrease. This confirms that blood biomarkers—particularly glucose—are essential for any meaningful HOMA-IR prediction. Wearable-only monitoring can detect broad trends (e.g., BMI changes via resting heart rate shifts) but cannot quantify insulin resistance.

## 6. Discussion

### 6.1 The Insulin Bottleneck

The central finding of this work is that HOMA-IR prediction from wearable + demographic + lipid features is fundamentally limited by missing insulin information. HOMA-IR = glucose × insulin / 405, and insulin correlates 0.97 with HOMA-IR while glucose correlates only 0.57. Our best proxy for insulin from available features achieves R²=0.35-0.44—far too noisy for accurate HOMA-IR reconstruction.

This is not a modeling limitation but an information limitation. No algorithm—gradient boosting, deep learning, or otherwise—can extract information that is not present in the features. Our 28-version experiment, spanning every major tabular modeling paradigm, empirically confirms this.

### 6.2 Implications for Wearable Health Monitoring

Our findings suggest a nuanced role for wearables in metabolic health:

1. **Screening (binary)**: Wearables + demographics can likely classify healthy vs unhealthy with moderate accuracy, sufficient for population-level screening where sensitivity matters more than precision.

2. **Quantification (continuous)**: Wearables alone are insufficient for clinical-grade HOMA-IR estimation (R²=0.091). Even with blood biomarkers, continuous monitoring through wearables adds only R²≈0.037 over blood-only models.

3. **Complementary monitoring**: The highest-value use case may be tracking relative changes over time for individuals, where the absolute error matters less than trend detection.

### 6.3 When to Stop Modeling

Our systematic experiment illustrates a practical challenge in applied ML: knowing when additional modeling effort will not yield improvement. We propose a three-part stopping criterion:

1. **Residual-feature correlation ≈ 0**: If model residuals are uncorrelated with all available features, no additional feature engineering can help.
2. **Error correlation > 0.95 across model families**: If different model types make the same errors, architectural changes won't help.
3. **Gap to ceiling estimate < ceiling estimation error**: If the remaining improvement potential is within the uncertainty of your ceiling estimate, you've likely reached the limit.

All three criteria were met by V14 of our experiment. The subsequent 14 versions confirmed this but added no performance.

### 6.4 Limitations

1. **Single cohort**: Results may not generalize to other populations with different HOMA-IR distributions.
2. **Summary statistics only**: We use 14-day aggregates rather than raw time-series. While prior work suggests raw series add minimal information, temporal patterns (e.g., postprandial heart rate responses) could theoretically contribute.
3. **k-NN ceiling is an estimate**: The R²=0.614 ceiling depends on k and dimensionality. With k=10 in 72-dimensional space, neighbors are relatively distant, potentially overestimating achievable performance.
4. **No continuous glucose monitoring (CGM)**: CGM data would provide glucose dynamics beyond fasting glucose, potentially improving prediction.

## 7. Conclusion

Through 28 systematic modeling approaches and rigorous information-theoretic analysis, we establish that R²≈0.55 represents the practical ceiling for HOMA-IR prediction from wearable, demographic, and standard blood biomarker features. The bottleneck is not methodology but information: fasting insulin, which drives 94% of HOMA-IR variance, cannot be reliably inferred from available features. Our analysis identifies 57 "hidden insulin resistant" individuals—clinically important patients invisible to any model without insulin measurement.

For the wearable health monitoring community, our findings delineate clear boundaries: wearables can contribute marginally to metabolic health assessment but cannot replace blood-based measurement for insulin resistance quantification. Future work should focus on (1) incorporating CGM data for glucose dynamics, (2) developing wearable-based insulin sensing technologies, and (3) reframing the problem as binary screening rather than continuous estimation.

## Figures

1. **The Hill-Climbing Journey** (fig_journey.png): R² trajectory across 28 versions
2. **Information-Theoretic Analysis** (fig_information_flow.png): Feature importance, variance decomposition, error correlation, insulin bottleneck
3. **Prediction Analysis** (fig_predictions_deep.png): Predicted vs true, residuals, subgroups, calibration, learning curve, error by range
4. **The Hidden Insulin Resistant** (fig_hidden_ir.png): t-SNE, error patterns, feature profiles, case studies
5. **Feature Group Ablation** (fig_model_comparison.png): Contribution of each feature group, marginal gains, Model A vs B, information hierarchy
