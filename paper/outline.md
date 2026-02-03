# NeurIPS Paper: Information-Theoretic Limits of Wearable-Based Insulin Resistance Estimation

## Title Options
1. "How Much Can Wearables Tell Us About Insulin Resistance? An Information-Theoretic Analysis"
2. "The Insulin Gap: Quantifying Information Limits in Wearable-Based HOMA-IR Prediction"
3. "Beyond Model Selection: Understanding Why Wearables Cannot Fully Predict Insulin Resistance"

## Abstract
Insulin resistance (IR) is a precursor to type 2 diabetes and metabolic syndrome. HOMA-IR, the gold-standard surrogate, requires fasting blood draws. Recent work has explored predicting HOMA-IR from wearable sensor data. We present a systematic study of 28 modeling approaches on 1,078 participants with wearable, demographic, and blood biomarker features. Our best model achieves R²=0.547, beating prior work (R²=0.50) using only summary statistics. More importantly, we prove through k-NN neighbor variance analysis that the theoretical ceiling is R²≈0.61 — our models capture 87% of the achievable signal. We trace the fundamental bottleneck to a single missing variable: fasting insulin. We quantify that glucose alone explains 10× more variance than all wearable features combined, and show that all gradient boosting models fail on the exact same samples (error correlation >0.99). These "hidden insulin resistant" individuals have normal-appearing feature profiles but elevated insulin — invisible to any model without blood-based insulin measurement. Our analysis provides a principled framework for understanding information limits in health prediction from wearables.

## 1. Introduction
- Insulin resistance → T2D pathway
- HOMA-IR = glucose × insulin / 405 (requires blood draw)
- Promise of wearables for non-invasive health monitoring
- Prior work: raw time-series models achieve R²≈0.50
- Our question: not "what model is best?" but "how much information do wearable features contain about HOMA-IR?"
- Contributions:
  1. Information-theoretic ceiling analysis (R²≈0.61)
  2. Systematic 28-version experiment covering all standard approaches
  3. Proof that the bottleneck is information, not modeling
  4. Characterization of "hidden insulin resistant" individuals
  5. Actionable insights: which features matter, which don't

## 2. Related Work
- HOMA-IR prediction from wearables (the original paper we're building on)
- Surrogate markers for insulin resistance (TyG, METS-IR, etc.)
- Information-theoretic limits in health prediction
- Tabular data modeling (XGBoost, LightGBM, deep tabular)

## 3. Dataset and Features
- 1,078 participants, 25 features (3 demo + 15 wearable + 7 blood)
- HOMA-IR distribution: mean=2.43, std=2.13, skewness=2.62
- Feature engineering: metabolic proxies (TyG, METS-IR, ir_proxy_rhr)
- Table 1: Feature descriptions and statistics
- Figure 1: HOMA-IR distribution + feature correlations

## 4. Methods
### 4.1 Evaluation Framework
- 5-fold × 5-repeat stratified CV (25 splits, seed 42)
- Honest evaluation: no data leakage, no nested stacking inflation
- Metrics: R², MAE, RMSE, bias

### 4.2 Modeling Approaches (28 versions)
- Linear: ElasticNet, Ridge, Lasso
- Tree-based: XGBoost, LightGBM, GBR, HGBR, CatBoost
- Kernel: SVR, Kernel Ridge
- Instance-based: KNN
- Ensemble: Dirichlet blending, nested stacking
- Target transforms: log1p, Box-Cox, power transforms
- Sample weighting: sqrt(y), y^α sweep
- Loss functions: MSE, MAE, Huber, Quantile
- Oversampling: SMOTER
- Calibration: isotonic, linear stretch

### 4.3 Information-Theoretic Ceiling
- k-NN neighbor variance analysis
- For each sample, find k=10 nearest neighbors in feature space
- Compute variance among neighbor targets = irreducible noise
- Theoretical max R² = 1 - mean(neighbor_variance) / var(y)

## 5. Results
### 5.1 Model Performance Trajectory
- Figure 2: R² trajectory across 28 versions (the journey)
- Table 2: Top models and their configurations
- Key finding: plateau at R²≈0.55 regardless of approach

### 5.2 Information-Theoretic Ceiling
- Figure 3: Ceiling analysis visualization
- Theoretical max R² = 0.614
- Current best = 0.547, capturing 87% of achievable signal
- Gap of 0.067 — within noise/method limitations

### 5.3 Feature Importance Decomposition
- Figure 4: Drop-one feature group importance
- Glucose: -0.096 (by far dominant)
- BMI: -0.011
- All wearables combined: -0.010
- Sleep/AZM: ≈ 0.000
- Key insight: glucose provides 10× more signal than all wearables

### 5.4 Error Analysis
- Figure 5: Error correlation matrix across model families
- All trees: >0.99 correlation (fail on same samples)
- ElasticNet: 0.878 (provides genuine diversity)
- Residual-feature correlations ≈ 0 (no unused signal)
- Corr(residual, y_true) = 0.705 → systematic mean regression

### 5.5 Subgroup Analysis
- Figure 6: Performance by sex, BMI, activity level
- Sex disparity: Female R²=0.61 vs Male R²=0.46
- RHR subgroups: Low R²=0.69 vs High R²=0.37
- "Hidden insulin resistant": normal features, high HOMA
- Figure 7: t-SNE + case studies of worst predictions

### 5.6 Model B: Wearables Only
- Without blood biomarkers: R²=0.259
- Dramatic drop confirms blood features (especially glucose) are essential
- Wearable-only monitoring has inherent limitations

## 6. Discussion
### 6.1 The Insulin Bottleneck
- HOMA = glucose × insulin / 405
- Insulin corr with HOMA = 0.969
- Insulin is not predictable from wearables (R²=0.35-0.44 from features)
- This is the fundamental limit

### 6.2 Implications for Wearable Health Monitoring
- Wearables can screen (binary healthy/unhealthy) but not quantify IR
- Glucose monitoring (CGM) is far more valuable than activity tracking for IR
- Model B shows wearables alone achieve R²≈0.26 — insufficient for clinical use

### 6.3 When More Modeling Doesn't Help
- 28 approaches, same ceiling
- Error analysis as a stopping criterion
- Practical guide: when to stop optimizing and accept the information limit

### 6.4 Limitations
- Single cohort, summary statistics only
- k-NN ceiling is an estimate (depends on k, dimensionality)
- Raw time-series might contain additional signal (but prior work suggests not much)

## 7. Conclusion
- R²≈0.55 is near the ceiling for HOMA-IR prediction without insulin
- The bottleneck is information, not methodology
- Glucose alone dominates; wearables contribute marginally
- Proper error analysis should precede model shopping
- Framework generalizable to other health prediction tasks

## Figures (Publication Quality)
1. HOMA-IR distribution + feature correlation heatmap
2. R² trajectory across 28 versions (the hill-climbing journey)
3. Information ceiling analysis (neighbor variance)
4. Feature importance decomposition (waterfall/bar chart)
5. Error correlation matrix (heatmap) across model families
6. Subgroup performance breakdown (sex × BMI × activity)
7. t-SNE with error overlay + "hidden insulin resistant" case studies
8. Predicted vs True with calibration analysis
9. Model B vs Model A comparison (information content of feature groups)
10. Learning curve (samples vs R²) showing plateau
