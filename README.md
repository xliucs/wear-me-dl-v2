# WEAR-ME-DL V2: HOMA-IR Prediction from Wearables & Blood Biomarkers

Predicting insulin resistance (True_HOMA_IR) from anonymized wearable (Fitbit) and clinical data.

## Task

Two regression models predicting HOMA-IR:
- **Model A**: All features (demographics + wearables + blood biomarkers) — no ground truth
- **Model B**: No blood biomarkers (demographics + wearables only)

## Dataset

- **Samples**: 1,078 participants
- **Features**: 25 (3 demographics + 15 wearable + 7 blood biomarkers)
- **Target**: True_HOMA_IR (continuous, mean=2.43, std=2.13)

### Feature Groups
| Group | Features |
|-------|----------|
| Demographics (3) | age, sex, BMI |
| Wearables (15) | Resting HR, HRV, Steps, Sleep Duration, AZM Weekly (mean/median/std each) |
| Blood Biomarkers (7) | total cholesterol, HDL, triglycerides, LDL, chol/HDL, non-HDL, glucose |

## Best Results (Stratified 5-Fold CV, 5 Repeats)

| Model | Best R² | Target R² | Gap | Method |
|-------|---------|-----------|-----|--------|
| **A (ALL)** | **0.5110** | 0.65 | 0.139 | ElasticNet on engineered features |
| **B (DW)** | **0.2463** | — | — | Ridge on raw features |

## Version History

| Version | Model A R² | Model B R² | Key Changes |
|---------|-----------|-----------|-------------|
| V1 | 0.5110 | 0.2463 | Baseline: 16 models × {raw, engineered} × {standard, log-target} |
| V2 | *running* | — | Stacking + TabPFN + MI feature selection |

## Evaluation Protocol

All experiments use a **standardized evaluation framework** (`eval_framework.py`):
- **Stratified 5-Fold CV with 5 repeats** (25 total splits)
- Stratification bins: `pd.qcut(y, 5)`
- Fixed `random_state=42` for reproducibility
- OOF predictions averaged across repeats

## Key Findings

### V1 Baseline Observations
1. **Linear models are competitive**: ElasticNet (R²=0.5110) matches tree ensembles (XGB R²=0.5064)
2. **Feature engineering helps linearly**: Engineered features boost linear models but not tree models much
3. **Log-target helps HGBR**: HGBR with log-transform (R²=0.4994→0.5029) benefits from stabilized variance
4. **DW is fundamentally limited**: Without glucose/blood, best R²=0.2463 (BMI is strongest predictor, r≈0.44)
5. **Top MI features**: glucose×BMI, IR_proxy (glucose×BMI×trig/HDL), TyG index — all glucose-related

### Why R²=0.65 is Hard
- HOMA_IR = glucose × insulin / 405
- Without insulin measurements, we can only proxy via glucose + lipid + anthropometric features
- The paper (arxiv:2505.03784) achieved R²=0.5 with 1,165 samples + MAE wearable embeddings
- Our feature engineering captures most linear relationships; the gap requires nonlinear interaction discovery

## Project Structure

```
├── README.md                 # This file
├── eval_framework.py         # Standardized evaluation (CV, metrics, feature engineering)
├── v1_baseline.py            # V1: Comprehensive baseline (16 models, raw/eng, standard/log)
├── v1_run.py                 # V1: Fast baseline runner
├── v2_stacking.py            # V2: Multi-model stacking + feature selection
├── data.csv                  # Dataset (not tracked)
└── .gitignore
```

## Running

```bash
pip install torch pandas scikit-learn xgboost lightgbm tabpfn numpy scipy

# V1 baseline
python v1_run.py

# V2 stacking
python v2_stacking.py
```

## References

- [Insulin Resistance Prediction From Wearables and Routine Blood Biomarkers](https://arxiv.org/abs/2505.03784) (Heydari et al., 2025)
- [TabPFN v2: Accurate predictions on small data](https://www.nature.com/articles/s41586-024-08328-6) (Hollmann et al., Nature 2025)
- [WEAR-ME-DL V1](https://github.com/xliucs/wear-me-dl) (prior iteration)
