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
| **V9** | **Family mega-blend** | **0.5287** | **0.5361** | **Diversity > quantity: XGB(59%)+ElasticNet(33%)** |

**Current Best: R² = 0.5368** (V7 Top-15 Dirichlet blend)

## Dataset
- **Samples:** 1,078 participants
- **Features:** 25 (3 demographics + 15 wearable stats + 7 blood biomarkers)
- **Target:** True_HOMA_IR (mean=2.43, std=2.13, skewness=2.62)

## Key Findings

### What Works
1. **Log target transform** (+0.015 R²): HOMA_IR has heavy right tail (skewness=2.62). Predicting log1p(y) and inverting consistently improves all models.
2. **V7 engineered features** (72 features): TyG, METS-IR, glucose×BMI, trig/HDL, ir_proxy×RHR, log_homa_proxy. Better than raw (25) for trees.
3. **XGBoost depth 3** with low learning rate: Shallow trees + regularization beat deeper models.
4. **Diverse blending**: Mixing different model TYPES (XGB + ElasticNet + LGB) beats blending similar models (5 XGB seeds).

### What Doesn't Work
- **L2 stacking**: Leaks if not nested properly. Honest stacking ≈ simple blending.
- **Insulin proxy decomposition (V5)**: Predicting HOMA/glucose separately amplifies errors.
- **SVR/KNN**: Wrong inductive bias (R²=0.36-0.45).
- **Box-Cox target**: Slightly worse than log1p.
- **PyTorch neural nets**: SIGSEGV on Python 3.14; needs different Python version.

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
- Best: Ridge raw R²=0.2463
- Paper baseline: R²=0.212
- **Not yet seriously attempted** — focus has been on Model A

## Repository Structure
```
data.csv                    # 1078 samples, 25+1 features
eval_framework.py           # Standardized CV + metrics
v1_baseline.py → v9_mega_blend.py  # Version progression
v*_results.json             # Saved results per version
```
