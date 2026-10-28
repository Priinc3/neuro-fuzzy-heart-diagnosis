# Neuro-Fuzzy Heart Diagnosis — Project Report

**ANFIS-based heart-disease classification with explainable fuzzy rules**

Soft Computing course project · GitHub: <https://github.com/Priinc3/neuro-fuzzy-heart-diagnosis> · Live app: <https://neuro-fuzzy-heart.streamlit.app> *(after deploy)* · Related earlier project: <https://github.com/Priinc3/Heartattack_ML>

---

## Abstract

Heart disease is the leading cause of death worldwide, yet clinical risk models are usually black boxes that doctors cannot inspect. This project builds an **ANFIS (Adaptive Neuro-Fuzzy Inference System)** classifier — a hybrid of Takagi–Sugeno fuzzy inference and neural-network learning — that predicts heart-disease risk from **5 interpretable patient inputs** (Age, General Health, BMI, PhysicalHealth days, Comorbidity score) and explains every prediction as **human-readable IF-THEN fuzzy rules** with learned Gaussian membership functions. Trained on a balanced sample of 54,746 records from the CDC BRFSS 2020 survey, the ANFIS reaches **ROC-AUC 0.829** on a held-out test set — about **0.012 below** the best classical black-box baseline (MLP, 0.841) — while using 5 inputs instead of 17. **Fuzzy C-Means** clustering (FPC-optimal c = 2) additionally splits patients into a healthier cluster (41.7% disease rate) and a high-risk cluster (78.0% disease rate). A five-tab Streamlit dashboard serves the model publicly: diagnosis with rule explanations, model comparison, an ANFIS inspector (membership functions + rule book), fuzzy patient clusters, and project documentation. The core soft-computing message: *a small, principled drop in AUC buys a model a clinician can actually read.*

## 1 Introduction

Cardiovascular disease accounts for roughly one in five deaths in the United States (CDC). Population surveys such as the CDC's **Behavioral Risk Factor Surveillance System (BRFSS)** collect hundreds of thousands of self-reported health records every year, making them a natural laboratory for machine-learning risk screening.

Most ML heart-disease studies optimise a single number (accuracy or AUC) and stop. In medicine that is not enough: a model a doctor cannot interrogate will not be trusted, adopted, or debugged. **Explainable AI matters in medicine** because (a) clinicians need to sanity-check that a prediction follows plausible reasoning ("old age + poor self-rated health + prior stroke → high risk", not "ZIP code 4 → high risk"); (b) regulations and informed consent increasingly require justification for automated decisions (Rudin, 2019 — high-stakes decisions should use models that are inherently interpretable, not post-hoc explanations of black boxes); and (c) explanations expose dataset bias and error modes.

**Fuzzy systems** are a natural fit: they reason in linguistic terms ("Age is High", "GenHealth is Poor") with partial memberships, which is exactly how clinicians verbalise risk. Their weakness is that hand-crafted membership functions and rules are hard to tune. **Neural networks** learn from data but are opaque. **ANFIS (Jang, 1993)** combines the two: a Takagi–Sugeno fuzzy inference system whose membership-function and consequent parameters are *learned by gradient descent like a neural network* — fuzzy interpretability with neural accuracy.

This project asks: *how much accuracy, if any, must we give up for that readability?* We compare ANFIS against three classical baselines (Logistic Regression, Random Forest, MLP) on the same data, add Fuzzy C-Means clustering as an unsupervised soft-computing view of the patient population, and deploy everything as a public, interactive dashboard.

## 2 Soft computing background

**Fuzzy sets (Zadeh, 1965).** A fuzzy set A over universe X has membership function μ_A: X → [0,1]; an element can belong partially ("BodyMassIndex is High" to degree 0.6). Crisp sets are the special case μ ∈ {0,1}.

**Membership functions.** The degree of membership is quantified by shapes such as triangles, trapezoids or Gaussians. We use **Gaussian MFs**

  μ_ij(x_i) = exp( −(x_i − c_ij)² / (2σ_ij²) )

with **center c_ij** (where the term fires fully) and **spread σ_ij** (how wide the linguistic label is). "Age is High" with c = 0.883, σ = 0.378 (on the scaled [0,1] Age axis) is a learned example — see §5.4.

**Takagi–Sugeno (TS) inference (Takagi & Sugeno, 1985).** Unlike Mamdani systems, TS rules keep the consequent a *function* of the inputs: r: IF x₁ is A₁ AND … AND xₙ is Aₙ THEN f_r(x) = p_r·x + r_r. All rule outputs are blended by their firing strengths into one crisp value, so no defuzzification step is needed — ideal for a regression/likelihood output.

**Neural learning.** ANFIS is this TS system arranged as a layered network; every parameter (MF centers/spreads, consequent weights) is differentiable, so mini-batch gradient descent (we use **Adam**, Kingma & Ba, 2014) trains it end-to-end with analytic gradients. MF centers and spreads can therefore *move to where the data says the linguistic terms actually lie*.

**Why hybrid.** Fuzzy-only (hand-tuned MFs) is brittle and unscalable; neural-only is opaque. The hybrid gives (i) learnable, data-driven fuzzy terms, (ii) a rule base that is fully inspectable after training, and (iii) a small parameter count — here 2 MFs × 5 inputs plus 32 linear consequents, a few hundred learnable numbers versus thousands of weights in an MLP on 17 features.

**Fuzzy C-Means and the Fuzzy Partition Coefficient.** **FCM (Bezdek, 1981)** minimises J = Σᵢ Σ_c u_ic^m ‖xᵢ − v_c‖², m = 2 (fuzzifier), alternating between membership updates u_ic ∝ ‖xᵢ − v_c‖^(−2/(m−1)) and centroid updates v_c = Σᵢ u_ic^m xᵢ / Σᵢ u_ic^m. Unlike k-means it outputs *soft* memberships — a patient can be 70% "healthy cluster", 30% "high-risk cluster". The number of clusters is chosen with the **Fuzzy Partition Coefficient (FPC, Bezdek 1981)** — the average degree to which memberships are crisp, FPC = (Σᵢᵤ Σ_c u_ic²)/n — maximised over candidate c; higher FPC ⇒ cleaner, less fuzzy partition.

## 3 Dataset

**Source.** CDC BRFSS 2020 annual survey, distributed via Kaggle as *"Personal Key Indicators of Heart Disease"* (kamilpytlak): **319,795 rows**, 17 health/behavior columns plus the binary target `HeartDisease`. Target prevalence is only ≈ 8.8% ("No" ≫ "Yes"), typical for survey data.

**Balancing.** A classifier trained on 9:1 data learns to say "No". We therefore **balance by undersampling**: all HeartDisease == "Yes" rows (27,373) plus an equal count of random "No" rows (seed 42) → **54,746 rows, exactly 50/50**. Consequence (honest framing): the *base rate* is now 50%, not the real-world ~9%; see §7.

**Split.** Stratified 80/20 train/test, seed 42: **43,796 train / 10,950 test**. Scalers and encoders are fit on train only.

**The 5 ANFIS inputs** (frozen contract in `src/features.py`; each min–max scaled to [0,1] with fixed ranges, clipped):

| # | Input | Raw source | How it is built | Range |
|---|---|---|---|---|
| 1 | Age | `AgeCategory` | Midpoint of the 13 survey brackets; "80 or older" → 85 | 18–85 → [0,1] |
| 2 | GenHealth | `GenHealth` | Ordinal rank Excellent=0, Very good=1, Good=2, Fair=3, **Poor=4** — higher = worse health | 0–4 → [0,1] |
| 3 | BMI | `BMI` | Raw body-mass index | 12–60 → [0,1] |
| 4 | PhysicalHealth | `PhysicalHealth` | Days of poor physical health in the last 30 days | 0–30 → [0,1] |
| 5 | Comorbidity | `Stroke`, `DiffWalking`, `KidneyDisease`, `Smoking`, `Diabetic` | Count 0–5: each of Stroke / DifficultyWalking / KidneyDisease / Smoking adds 1 if "Yes"; **Diabetic mapping: "Yes" = 1, "No, borderline diabetes" = 0.5, "Yes (during pregnancy)" = 1** | 0–5 → [0,1] |

The classical baselines and the FCM dashboard use the full **17 raw columns** (BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer) via StandardScaler + OneHotEncoder. ANFIS deliberately compresses 17 → 5 (the comorbidity score is a domain-informed feature aggregation) to stay interpretable.

## 4 Methodology

### 4.1 ANFIS architecture (first-order Takagi–Sugeno, 5 layers)

Grid partition: **2 Gaussian MFs per input × 5 inputs = 2⁵ = 32 rules**, labels Low/High. Inputs scaled to [0,1].

| Layer | Role | Formula |
|---|---|---|
| L1 — fuzzification | Gaussian membership of each input | μ_ij(x_i) = exp(−(x_i − c_ij)²/(2σ_ij²)) |
| L2 — rule firing | AND of memberships (product t-norm) | w_r = Πᵢ μ_{i,j(r,i)}(x_i) |
| L3 — normalisation | relative firing strength | w̄_r = w_r / Σ_s w_s |
| L4 — consequent | first-order linear consequent | f_r(x) = p_r·x + r_r |
| L5 — output | weighted sum → probability | logit = Σ_r w̄_r f_r(x), P(disease) = σ(logit) |

**Training.** Binary cross-entropy loss; **Adam, lr = 0.01, batch = 512, 40 epochs**, seed 42; analytic gradients for c, σ, p, r (pure NumPy; sklearn used only for metrics). Initial MFs: centers 0.25/0.75, σ = 0.25 per input. Train time: **5.1 s** (CPU). Training curve (`artifacts/anfis_history.json`): loss 0.574 → 0.506, validation AUC 0.817 → 0.829 over the 40 epochs.

### 4.2 Baselines

On the 17-column raw features: **LogisticRegression** (linear, interpretable coefficients), **RandomForest** (bagged trees, feature importance), **MLP** (2-layer neural net, black box). Same 80/20 split, same metrics. LR is the "how much does interpretability cost" reference: it is itself interpretable, but through 30+ one-hot coefficients rather than 5 linguistic variables.

### 4.3 Fuzzy C-Means

`scikit-fuzzy` cmeans on a 10k subsample of the scaled training matrix, **c = 2..6, m = 2** (error 0.005, maxiter 300, seed 42). Best c by FPC. Cluster centroids and per-cluster disease rates are shown in the dashboard's cluster tab.

## 5 Results

All numbers below are taken directly from `artifacts/metrics_anfis.json`, `artifacts/metrics_baselines.json`, `artifacts/fcm.json` (test set = 10,950 rows).

### 5.1 Model comparison

| Model | Inputs | Accuracy | Precision | Recall | F1 | ROC-AUC | Train s |
|---|---|---|---|---|---|---|---|
| **ANFIS** | 5 | 0.7521 | 0.7515 | 0.7532 | 0.7523 | **0.8288** | 5.11 |
| LogisticRegression | 17 | 0.7643 | 0.7575 | 0.7775 | 0.7674 | 0.8405 | 0.21 |
| RandomForest | 17 | 0.7574 | 0.7545 | 0.7629 | 0.7587 | 0.8327 | 0.49 |
| **MLP (best classical)** | 17 | 0.7616 | 0.7664 | 0.7527 | 0.7595 | **0.8413** | 2.52 |

Confusion matrices (rows = actual No/Yes, cols = predicted No/Yes):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| ANFIS | 4111 | 1364 | 1351 | 4124 |
| LogisticRegression | 4112 | 1363 | 1218 | 4257 |
| RandomForest | 4116 | 1359 | 1298 | 4177 |
| MLP | 4219 | 1256 | 1354 | 4121 |

Key observations:

- All four models cluster tightly (AUC 0.829–0.841): with 5 well-built features, most of the achievable signal is captured.
- **ANFIS − MLP AUC = 0.8413 − 0.8288 = 0.0125** (≈ 0.012), ANFIS − LR = 0.0117. That is the entire price of going from 17 inputs/30+ coefficients to 5 fuzzy variables with readable rules.
- ANFIS trains in 5.1 s and its rules remain stable across restarts (fixed seed).

### 5.2 Fuzzy C-Means — FPC and clusters

**FPC by c** (higher = more crisp partition):

| c | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| FPC | **0.7024** | 0.5636 | 0.4700 | 0.3920 | 0.3355 |

**FPC is maximised at c = 2** → the survey population splits into two fuzzy patient types (a two-cluster reading is also the most human: "healthier" vs "high-risk"). Cluster centroids (scaled 0–1 space) and disease rates:

| Cluster | Age | GenHealth (↑ = worse) | BMI | PhysHealth (↑ = worse) | Comorbidity | Disease rate |
|---|---|---|---|---|---|---|
| 1 — healthier | 0.618 | 0.327 | 0.335 | 0.033 | 0.155 | **0.4173** |
| 2 — high-risk | 0.717 | 0.727 | 0.375 | 0.591 | 0.405 | **0.7798** |

The high-risk cluster reports much worse general health (0.727 vs 0.327), 12× more poor-health days (0.591 vs 0.033) and ~2.6× the comorbidity load (0.405 vs 0.155) — yet its disease rate is 78%, not 100%, and the healthier cluster is still at 42% (not 0%). Both numbers make sense in a 50/50-balanced sample (base rate 0.5) with *fuzzy* membership: patients belong partially to both clusters, so neither cluster is pure. The gradient (0.417 → 0.780) is the signal.

### 5.3 Example explanations (real ANFIS output)

Two profiles from the test scenarios (`prompts/batch-B1-TEST.md` §4), run through `ANFIS.top_rules` on the deployed model. Reading note: **GenHealth is High / PhysicalHealth is High means *worse* health** (the scales rank Poor=4 and more sick-days as larger values).

**High-risk profile** — Age "80 or older", GenHealth Poor, BMI 40, PhysicalHealth 30 days, Stroke/DiffWalking/KidneyDisease/Smoking/Diabetic all Yes (Comorbidity = 5):

```
ANFIS probability: 0.9713
Top rules:
  [74.4% -> out 0.979] IF Age is High AND GenHealth is High AND BMI is High
      AND PhysicalHealth is High AND Comorbidity is High THEN risk = 0.98
  [9.3% -> out 0.944] IF Age is Low AND GenHealth is High AND BMI is High
      AND PhysicalHealth is High AND Comorbidity is High THEN risk = 0.94
  [6.1% -> out 0.857] IF Age is High AND GenHealth is Low AND BMI is High
      AND PhysicalHealth is High AND Comorbidity is High THEN risk = 0.86
  [5.6% -> out 0.984] IF Age is High AND GenHealth is High AND BMI is High
      AND PhysicalHealth is Low AND Comorbidity is High THEN risk = 0.98
  [2.0% -> out 0.935] IF Age is High AND GenHealth is High AND BMI is High
      AND PhysicalHealth is High AND Comorbidity is Low THEN risk = 0.93
```

The dominant rule (74.4% of the firing weight) is the "everything bad" rule with consequent risk ≈ 0.98; the prediction 0.9713 is essentially that rule's output, slightly moderated by rules where one term fires Low. A clinician can read this directly: *this patient is high risk because age, self-rated health, sick-days and comorbidity are all in the worst fuzzy band.*

**Low-risk profile** — Age "18-24", GenHealth Excellent, BMI 22, PhysicalHealth 0, all five comorbidity items No:

```
ANFIS probability: 0.0299
Top rules:
  [86.1% -> out 0.019] IF Age is Low AND GenHealth is Low AND BMI is High
      AND PhysicalHealth is Low AND Comorbidity is Low THEN risk = 0.02
  [7.6% -> out 0.393] IF Age is High AND GenHealth is Low AND BMI is High
      AND PhysicalHealth is Low AND Comorbidity is Low THEN risk = 0.39
  [3.5% -> out 0.382] IF Age is Low AND GenHealth is Low AND BMI is High
      AND PhysicalHealth is Low AND Comorbidity is High THEN risk = 0.38
  [1.2% -> out 0.206] IF Age is Low AND GenHealth is High AND BMI is High
      AND PhysicalHealth is Low AND Comorbidity is Low THEN risk = 0.21
  [1.0% -> out 0.273] IF Age is Low AND GenHealth is Low AND BMI is Low
      AND PhysicalHealth is Low AND Comorbidity is Low THEN risk = 0.27
```

The winning rule (86.1%) fires "BMI is High" even though BMI 22 is normal — because the *learned* BMI membership functions degenerated (see §5.4): BMI turned out to be weakly informative in this feature set, so the "High" Gaussian became very broad (σ = 0.856) and covers most of the BMI axis while the "Low" one collapsed to a narrow spike at 0. Its consequent is still risk ≈ 0.02, and the prediction is 0.0299. The explanation remains honest and readable — including this quirk.

### 5.4 Learned membership functions

Training moved the MFs from their initial positions (centers 0.25/0.75, σ 0.25 everywhere; `artifacts/anfis_init_mf.json`) to data-driven positions (scaled space, from `artifacts/anfis_model.npz`):

| Input | Low (c, σ) | High (c, σ) | Interpretation |
|---|---|---|---|
| Age | (0.142, 0.416) | (0.883, 0.378) | "young" ≈ 18–35, "old" ≈ 65+ |
| GenHealth (↑ worse) | (0.192, 0.359) | (0.923, 0.311) | "good health" ≈ Excellent/Very good, "poor health" ≈ Fair/Poor |
| BMI | (0.000, 0.070) | (0.196, 0.856) | weakly informative input → "High" MF became near-omnipresent |
| PhysicalHealth (↑ worse) | (0.060, 0.410) | (0.943, 0.223) | "few sick-days" vs "many sick-days" |
| Comorbidity | (0.096, 0.315) | (0.714, 0.280) | "clean history" vs "multimorbid" |

## 6 Discussion — the honest trade-off

**What we give up.** ANFIS ROC-AUC 0.8288 vs MLP 0.8413 and LR 0.8405: a gap of ≈ **0.012** (and 0.0086 vs RandomForest). In accuracy the gap is ≈ 0.01. If the goal were purely a leaderboard score, the MLP would win. The confusion matrices are also close: ANFIS trades ~90 extra false negatives for ~65 extra false positives relative to LR.

**What we get.**

1. **5 inputs instead of 17.** Age midpoint, self-rated health, BMI, sick-days, comorbidity score. A patient fills five fields — no race, no sex, no mental-health questions needed — and the model never touches protected attributes.
2. **Every prediction ships with readable rules.** As §5.3 shows, the explanation is IF-THEN English plus firing percentages plus the learned MF curves. No SHAP/LIME layer needed — the model *is* the explanation (in Rudin's 2019 sense).
3. **Learned, not hand-crafted, fuzzy terms.** The MFs in §5.4 were found by gradient descent, i.e. the "linguistic categories" are consistent with what the data says, while remaining inspectable.
4. **Soft, graded outputs at every level** — fuzzy memberships in, normalized rule blending in the middle, probability out — a coherent soft-computing pipeline rather than one fuzzy component bolted onto a neural net.

The frank summary for the viva: **ANFIS costs ≈ 0.012 AUC and buys a 5-input model whose every decision can be read aloud in a clinical conversation.** Whether that trade is worth it depends on the deployment: for triage screening (this project), interpretability wins; for a benchmark competition, it does not.

## 7 Limitations

1. **Self-reported survey data.** BRFSS is questionnaire-based: BMI is self-reported (people under-report), "poor physical health days" is subjective recall, and `HeartDisease` is self-reported diagnosis. Noise caps the achievable AUC for *any* model — likely why all four models sit within 0.012 of each other.
2. **Balanced sampling changes the base rate.** Real prevalence is ≈ 8.8%; we trained/evaluated at 50%. Metrics (accuracy 0.75, AUC 0.83) are therefore *not* comparable to a real-population screening score without recalibration; PPV/NPV would shift drastically.
3. **Grid partition grows as 2ⁿ rules.** 5 inputs × 2 MFs = 32 rules is cheap; 3 MFs would be 243 rules and 10 inputs × 3 would be 59,049 — grid partition does not scale. Clustering-based rule generation (§8) is the standard cure.
4. **Membership functions can degenerate.** The BMI MFs (§5.4) show what happens when an input is weakly informative: one Gaussian collapses, the other becomes near-omnipresent. The rule text stays readable but the "BMI is High" term no longer means "high BMI".
5. **Not clinically validated.** No external validation cohort, no calibration curve, no fairness audit across sex/race/age subgroups. This is an educational model.
6. **Comorbidity score is our design choice.** It aggregates 5 conditions equally; a clinician would weight them differently (prior stroke ≠ current smoker).

## 8 Future work

- **Metaheuristic MF tuning:** replace/augment Adam with **GA or PSO** over centers/spreads (classic soft-computing synergy: GA-ANFIS), optimizing AUC *or* rule compactness directly.
- **Clustering-based rule generation:** initialise one rule per FCM (or subtractive-clustering) cluster instead of the full grid — keeps rule count near-linear in inputs, enabling more inputs without the 2ⁿ blow-up.
- **3 MFs per input** (Low/Medium/High) with the clustering initialisation above; check whether AUC closes the 0.012 gap.
- **Recalibration for real prevalence** (e.g. reweighting or Platt scaling) so probabilities are meaningful at ~9% base rate.
- **Fairness audit:** per-subgroup (sex, race, age-band) TPR/FPR; the 5-input design already avoids direct protected attributes but proxies remain.
- **Mamdani variant + defuzzification** for an even more linguistic output band ("low / borderline / elevated / high risk") instead of a bare probability.

## 9 Conclusion

We built and publicly deployed a complete soft-computing diagnostic system: a 32-rule, 5-input first-order Takagi–Sugeno ANFIS trained by Adam on BCE loss (AUC 0.829, 0.012 under the best 17-input black-box baseline), explained by Gaussian membership functions and readable IF-THEN rules; Fuzzy C-Means (FPC-optimal c = 2) that separates patients into 41.7%-vs-78.0%-disease-rate fuzzy clusters; and a five-tab Streamlit dashboard that makes the whole pipeline interactive. The project demonstrates the central soft-computing thesis on a real 319,795-row health survey: **interpretability has a measurable but small price, and fuzzy-neural hybrids are how you pay it deliberately.**

## Screenshots

> ⚠️ **Take after deploy** (app sleeps after 12 h idle — open it first). Save as PNG in `docs/screens/`, then commit. Do **not** fabricate images.

| File | Content | Where |
|---|---|---|
| `docs/screens/01-diagnose.png` | Diagnose tab: filled form, risk gauge, top fuzzy rules for the high-risk profile of §5.3 | https://neuro-fuzzy-heart.streamlit.app |
| `docs/screens/02-model-comparison.png` | Model comparison tab: 4-model metrics table + ROC curves | tab 2 |
| `docs/screens/03-inside-anfis.png` | Inside the ANFIS: learned Gaussian MF curves (§5.4) + rule book | tab 3 |
| `docs/screens/04-clusters.png` | Fuzzy patient clusters: FCM scatter + cluster disease rates | tab 4 |

## References

1. Jang, J.-S. R. (1993). *ANFIS: Adaptive-Network-Based Fuzzy Inference System.* IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665–685. DOI: [10.1109/21.256541](https://doi.org/10.1109/21.256541)
2. Takagi, T., & Sugeno, M. (1985). *Fuzzy identification of systems and its applications to modeling and control.* IEEE Trans. SMC, 15(1), 116–132. DOI: [10.1109/TSMC.1985.6313399](https://doi.org/10.1109/TSMC.1985.6313399)
3. Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms.* Plenum Press, New York. (Fuzzy C-Means; Fuzzy Partition Coefficient)
4. Zadeh, L. A. (1965). *Fuzzy sets.* Information and Control, 8(3), 338–353. DOI: [10.1016/S0019-9958(65)90241-X](https://doi.org/10.1016/S0019-9958(65)90241-X)
5. Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization.* arXiv: [1412.6980](https://arxiv.org/abs/1412.6980)
6. Rudin, C. (2019). *Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.* Nature Machine Intelligence, 1, 206–215. DOI: [10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x)
7. CDC — Behavioral Risk Factor Surveillance System (BRFSS): <https://www.cdc.gov/brfss/index.html>
8. Kaggle — *Personal Key Indicators of Heart Disease* (kamilpytlak): <https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease>
9. scikit-fuzzy documentation (FCM implementation): <https://scikit-fuzzy.readthedocs.io/>
10. Streamlit documentation — Community Cloud deploy & app management: <https://docs.streamlit.io/deploy/streamlit-community-cloud>
11. Project repository: <https://github.com/Priinc3/neuro-fuzzy-heart-diagnosis> · Live app: <https://neuro-fuzzy-heart.streamlit.app> · Earlier classical-ML project: <https://github.com/Priinc3/Heartattack_ML>
