# WEAR-ME DL V2: HOMA-IR Prediction

Predicting **True_HOMA_IR** from demographics, wearables, and blood biomarkers.

## Best Results (Honest, No Leakage)

| Version | Method | Best Single R² | Best Blend R² | Key Insight |
|---------|--------|---------------|--------------|-------------|
| V1 | 16+ models baseline | 0.5110 (ElasticNet eng) | — | Linear models competitive with trees |
| V2 | 28-model stacking | 0.5110 | 0.5276 (Dirichlet) | L2 stacking leaked (0.66→0.50 honest) |
| V3 | Nested validation | — | 0.5025 (nested L2) | Confirmed V2 leakage |
| V4 | 101 V4 features | 0.5146 (XGB d6) | — | More features help trees slightly |
| V5 | Insulin proxy decomp | 0.4531 | 0.5104 (two-stage) | Failed: error amplification |
| V6 | Optuna (30 trials) | 0.5111 (XGB) | — | Killed by timeout; marginal gains |
| V7 | Research-informed | 0.5271 (XGB d3 log eng) | **0.5368** (Top-15) | **Log target +0.015**, V7 features best |
| V8b | Multi-seed diversity | 0.5287 (XGB d3 log s2024) | 0.5350 | Similar models hurt blending |
| V9 | Family mega-blend | 0.5287 | 0.5361 | Diversity > quantity: XGB(59%)+ElasticNet(33%) |
| V10 | Residual analysis + greedy blend | 0.5287 | 0.5365 | Residuals unpredictable; extreme HOMA [8+] bias=+4.94; 3-model greedy = 15-model random |
| V11 | Tail fix: sample weighting | **0.5367** (sqrt weight) | **0.5414** | sqrt(y) sample weighting = NEW BEST. Upweighting high HOMA helps model learn tail. |
| V12 | Weight exponent search | 0.5367 (y^0.5 optimal) | 0.5414 | Confirmed y^0.5 is optimal exponent. Weighting helps XGB +0.008 but not LGB/HGBR. No blend gain. |
| V13 | Optuna weighted XGB + MLP | 0.5406 (Optuna wsqrt) | 0.5452 | Optuna re-tuned XGB WITH weights: d4, lr=0.017, n=612. MLP catastrophic with log target. |
| **V14** | **Optuna LGB + GBR + feature selection** | **0.5398** (LGB Optuna wsqrt) | **0.5465** | **LGB matches XGB when Optuna-tuned with weights. GBR adds diversity. Feature selection hurts.** |
| V15 | Optuna HGBR + cross-weight blend | 0.5398 (XGB opt w0.5) | 0.5462 | HGBR peaks at 0.5331 (weaker). Multi-weight-exponent blending doesn't beat V14. |
| V16 | Nested target encoding + piecewise | 0.5398 (XGB Optuna wsqrt) | 0.5463 | Nested (honest) TE hurts (0.534-0.536 vs 0.540). Leaky TE inflated to 0.584 — stacking leakage. Piecewise (0.535) worse than single. |
| **V17** | **Model B + power transforms** | **0.5398** (XGB Optuna) | **0.5466** | **Power transforms don't beat log1p** (y^0.3=0.539). **Model B: R²=0.2449** blend (beats paper 0.212 by +0.033). Marginal new Model A best. |
| V18 | Nested stacking + winsorization | 0.5422 (XGB winsor@12) | ~0.546 | **Nested stacking (0.5376) WORSE than simple blend.** Winsorize@12 gives single XGB 0.5422. Stacking adds variance, not signal — confirms simple Dirichlet blend is optimal. |

| V19 | KNN + kernel methods + residual correction | 0.5398 (XGB) | 0.5465 | KNN peaks at 0.4547 (much worse). **Kernel Ridge RBF (0.524) adds blend diversity** replacing ElasticNet. Residual correction hurts. All roads lead to R²≈0.546. |
| **V20** | **QT inputs + quantile reg + Model B Optuna** | **0.5408** (LGB QT wsqrt) | **0.5467** | **QuantileTransformer on inputs helps LGB +0.0014. Model B: R²=0.2592 (XGB Optuna wsqrt, big jump from 0.2449).** |

| V21 | CatBoost + Huber/MAE loss + SMOTER | 0.5388 (XGB SMOTER p85) | 0.5465 | **CatBoost disappointing (0.5259).** MAE helps XGB +0.0015. SMOTER p85 x3 helps +0.0023. Py3.12 env. |
| V22 | MAE + SMOTER + pseudo-insulin | 0.5388 (XGB MSE+SMOTER) | 0.5461 | MAE/SMOTER confirmed on Py3.14. **Pseudo-insulin feature HURTS** (noisy R²=0.34). Interaction explosion hurts. |
| V23 | Optuna MAE + Optuna SMOTER | 0.5371 (XGB Optuna MAE) | — | **MSE-tuned params beat Optuna MAE params.** SMOTER Optuna too conservative. Partial run (killed). |
| V24 | Target decomposition (predict insulin) | 0.5453 (XGB decomposed) | 0.5445 | Predict insulin→reconstruct HOMA. Insulin R²=0.44 from features. Error amplification. Doesn't beat standard. |
| **V25** | **Deep error analysis** | 0.5365 | — | **Theoretical max R²=0.614** (k-NN neighbor variance). Residual-feature corr ~0 (all signal extracted). Glucose drop costs 0.096. Wearables cost 0.01. |
| V26 | Calibration + stretch | 0.5392 (XGB quantile 0.5) | — | Isotonic/linear calibration HURTS. Scale factor 1.03 gives +0.0007. Quantile median slightly better than MSE. Killed during blend. |
| V27 | Hypothesis-driven (stratification) | 0.5384 (XGB d=3) | 0.5412 | **ALL stratification rejected**: sex (-0.018), BMI (-0.006), glucose (-0.022). Not enough data per stratum. Train-test gap (0.29) is information loss, not overfitting. |
| V28 | Maximum diversity blend | 0.5384 (XGB d=3) | 0.5400 | **Error correlations 0.99+ between ALL trees.** Only ElasticNet provides diversity (0.878). Blending more trees is nearly useless. |

**Current Best Model A: R² = 0.5467** (V20 blend: LGB_QT 71% + ElasticNet 29%)
**Current Best Model B: R² = 0.2592** (V20 blend: XGB_Optuna 75% + ElasticNet 20% + Ridge 5%)

## Dataset
- **Samples:** 1,078 participants
- **Features:** 25 (3 demographics + 15 wearable stats + 7 blood biomarkers)
- **Target:** True_HOMA_IR (mean=2.43, std=2.13, skewness=2.62)

## Key Findings

### What Works
1. **Log target transform** (+0.015 R²): HOMA_IR has heavy right tail (skewness=2.62). Predicting log1p(y) and inverting consistently improves all models.
2. **sqrt(y) sample weighting** (+0.008 R²): Upweighting high-HOMA samples forces model to learn the tail. Sweet spot between uniform and y²-weighting.
3. **V7 engineered features** (72 features): TyG, METS-IR, glucose×BMI, trig/HDL, ir_proxy×RHR, log_homa_proxy. Better than raw (25) for trees.
4. **XGBoost depth 3** with low learning rate: Shallow trees + regularization beat deeper models.
5. **Diverse blending**: Mixing different model TYPES (XGB + ElasticNet + LGB) beats blending similar models (5 XGB seeds).

### What Doesn't Work
- **L2 stacking**: Leaks if not nested properly. Honest stacking ≈ simple blending.
- **Insulin proxy decomposition (V5)**: Predicting HOMA/glucose separately amplifies errors.
- **SVR/KNN**: Wrong inductive bias (R²=0.36-0.45).
- **Box-Cox target**: Slightly worse than log1p.
- **PyTorch neural nets**: SIGSEGV on Python 3.14; needs different Python version.
- **Residual correction (V10)**: Meta-learning on prediction errors makes things worse. Residuals have negative R² — they're unpredictable noise given our features.
- **sklearn MLPRegressor with log target**: Predictions explode after expm1 inverse (R² = -billions). Raw target MLP only reaches 0.28-0.36.

### V10 Residual Analysis (Key Discovery)
| HOMA_IR Range | n | MAE | Bias | RMSE |
|---|---|---|---|---|
| Low [0-1) | 215 | 0.38 | -0.36 | 0.51 |
| Normal [1-2) | 391 | 0.49 | -0.27 | 0.71 |
| Elevated [2-3) | 216 | 0.74 | -0.04 | 1.02 |
| High [3-5) | 168 | 1.08 | +0.52 | 1.36 |
| Very High [5-8) | 49 | 2.16 | +1.89 | 2.58 |
| **Extreme [8+)** | **39** | **4.94** | **+4.94** | **5.51** |

Model systematically underpredicts high HOMA_IR. The 88 extreme samples (8%) contribute disproportionately to MSE.

### Why R²≈0.53 is the Ceiling (Analysis)
- **HOMA_IR = glucose × insulin / 405**: Without insulin, fundamental information ceiling.
- Glucose correlation with HOMA_IR = 0.574 → glucose alone explains ~33%.
- Our models capture glucose + partial insulin signal from BMI (r=0.43), triglycerides (r=0.41), HDL (r=-0.30).
- Reference repo achieves R²=0.5948 with **46 blood biomarkers** (vs our 7).
- Original paper achieves R²=0.5 with **raw time-series** + MAE embeddings.
- **We beat the paper** using only summary statistics!

## Evaluation
- **CV:** 5-fold × 5-repeat stratified (25 splits total), seed 42
- **Framework:** `eval_framework.py` — standardized across all versions
- **Blending:** Dirichlet weight search (500K-2M random trials)

## Feature Engineering (V7)
Top features by mutual information:
1. `glucose_bmi` — glucose × BMI
2. `ir_proxy_rhr` — (glucose × BMI × trig / HDL) × resting_heart_rate
3. `log_homa_proxy` — log(glucose) + log(BMI) + log(trig) - log(HDL)
4. `ir_proxy` — glucose × BMI × trig / HDL
5. `mets_ir` — METS-IR index: log(2×glucose + trig) × BMI / log(HDL)

## Model B (Demographics + Wearables Only)
- **Best: R²=0.2447** (V17 blend: ElasticNet + XGB + LGB)
- Best single: ElasticNet α=0.1, l1=0.9 → R²=0.2380
- Paper baseline: R²=0.212 → **We beat it by +0.033**
- XGB with sqrt weighting: R²=0.2232
- Key: Without blood biomarkers, information is severely limited (BMI+age+wearables only)

## Repository Structure
```
data.csv                    # 1078 samples, 25+1 features
eval_framework.py           # Standardized CV + metrics
v1_baseline.py → v15_hgbr_optuna_final.py  # Version progression
v*_results.json             # Saved results per version
```
