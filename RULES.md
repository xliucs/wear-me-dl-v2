# MANDATORY RULES — READ BEFORE EVERY ITERATION

## Progress Updates (CRITICAL)
After EVERY version/iteration completes, you MUST post a message:
1. Version number
2. Best R² achieved  
3. What you tried
4. What you learned (WHY did it work or not?)
5. What you're trying next

Do this BEFORE starting the next version. No silent grinding.

## Research Protocol

### Phase 1 — Understand (first 20% of time budget)
- EDA: distributions, correlations, missingness, outliers
- Baseline models (linear + one tree model) with FIXED eval framework from day 1
- Ceiling analysis: k-NN neighbor variance, theoretical limits from domain knowledge
- Subgroup analysis: performance by key demographics/splits
- t-SNE / PCA to understand feature space structure
- Output: "here's what the data looks like, here's the theoretical ceiling, here's where the model fails"

### Phase 2 — Hypothesis-driven experiments (next 50%)
- Every experiment starts with: "I expect +X R² because [analysis shows Y]"
- If you can't articulate WHY, don't run it
- After EVERY version: commit, README update, post update (version/R²/what/why/next)
- After EVERY version: run full Error Analysis Checklist (see below)
- Use analysis to generate NEXT hypothesis, not just to report results
- The cycle: analyze → hypothesize → experiment → analyze → hypothesize → ...

### Phase 3 — Write up (last 30%)
- Comprehensive post-hoc analysis + publication-quality figures
- Paper draft if appropriate
- If ceiling is reached earlier, enter Phase 3 earlier

## Error Analysis Checklist (EVERY VERSION — NO EXCEPTIONS)

Run ALL of these after every version. Each check drives the next hypothesis.

1. **Residual-feature correlations** — For each feature, compute corr(residual, feature). If any |corr| > 0.05, the model is leaving signal on the table.
   - Example: "corr(residual, steps) = 0.12 → model under-uses steps → try interaction feature glucose×steps"

2. **Subgroup performance breakdown** — Split test set by key categories (sex, BMI quartiles, age groups, target ranges) and compute R²/MAE per group.
   - Example: "R² for BMI>30 dropped from 0.45→0.38 after adding SMOTER → resampling hurts obese subgroup"

3. **Worst-case error analysis** — Look at top-10 highest-error samples. What do they have in common?
   - Example: "8/10 worst errors are young males with high insulin but normal glucose → model can't detect insulin resistance without glucose signal"

4. **Error correlation across models** — corr(residuals_modelA, residuals_modelB). If >0.95, blending won't help.
   - Example: "XGB vs LGB residual corr = 0.991 → both fail on same samples → need fundamentally different model family for diversity"

5. **Prediction distribution check** — Compare histogram of predictions vs true values. Is the model mean-regressing?
   - Example: "True HOMA range [0.3, 18.2] but predictions only span [0.8, 7.1] → model compresses tails → try quantile loss or tail-focused weighting"

6. **Delta analysis** — What changed from previous version? Which samples improved? Which got worse?
   - Example: "V15→V16: 23 samples improved >0.5 MAE, 31 got worse. Worse samples are all low-HOMA → new features help high end but hurt low end"

7. **Feature importance shift** — Compare feature importances (gain/SHAP) across versions. Did new features actually get used? Did importance redistribute?
   - Example: "Added 12 sleep features but total importance = 0.3% → sleep features are noise, remove them. Meanwhile glucose importance dropped from 45%→38% after adding BMI interaction → new feature capturing glucose signal differently"

8. **Embedding / representation analysis** — Visualize learned representations:
   - Neural nets: t-SNE/UMAP of penultimate layer activations, colored by target and by error magnitude
   - Tree models: t-SNE/UMAP of leaf node co-occurrence matrix (which samples end up in same leaves?)
   - Any model: PCA of prediction vectors across ensemble — are models seeing different structure?
   - Example: "t-SNE of leaf co-occurrence shows two distinct clusters that don't align with any single feature → possible latent subpopulation, try clustering-based features"

9. **SHAP interaction analysis** — Beyond single-feature importance, check top feature interactions.
   - Example: "SHAP interaction between glucose and BMI = 0.15 → synergy the model is partially capturing → try explicit glucose×BMI feature or deeper trees"

All 9 checks take ~10 min. The key output is always: **"What should I try next and WHY?"**

## Communication Rules
- Never go >30 min without an update
- If 30 min pass with no R² improvement: stop and message with "stuck at X, tried Y, hypothesis is Z"
- If 3 consecutive experiments show <0.002 gain: message before trying a 4th
- If uncertain whether an approach is sound: ASK before spending an hour on it

## Quality Gates
- If R² jumps >0.05 in one version: verify it's not leakage before reporting
- If train-test gap >0.20: diagnose (overfitting vs info ceiling) before continuing
- Fixed eval_framework.py with frozen CV splits from day 1
- All figures publication-quality from the start, consistent style

## Exit Criteria
- Residual-feature correlations all ~0 = signal exhausted
- Gap to ceiling < estimation error = stop modeling
- 3 consecutive versions <0.002 gain + analysis confirms no unused signal = write up

## System Rules
- Print training progress as it runs (each model, each fold)
- Never run silently for long periods
- If a run crashes, fix and resume immediately
- Commit to repo after each iteration. Update README with results.
