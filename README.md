# Neuro-Fuzzy Heart Diagnosis ❤️🧠

**ANFIS (Adaptive Neuro-Fuzzy Inference System) heart-disease classifier with explainable fuzzy rules — a Soft Computing project.**

🔗 **Live app:** https://neuro-fuzzy-heart.streamlit.app *(after deploy — deploy steps in [`docs/DEPLOY.md`](docs/DEPLOY.md))*

---

## What it does

An interactive Streamlit dashboard predicts heart-disease risk from 5 patient inputs and — unlike a black-box model — **explains every prediction as human-readable IF-THEN fuzzy rules** with learned Gaussian membership functions.

- **Soft computing techniques:** ANFIS (Jang 1993) — a hybrid of Takagi–Sugeno fuzzy inference and neural gradient learning — plus **Fuzzy C-Means clustering** (Bezdek 1981) to show natural patient subgroups.
- **Data:** CDC BRFSS 2020 behavioral-health survey (Kaggle: *Personal Key Indicators of Heart Disease*), 319,795 rows, balanced to 54,746.
- **5 tabs:** 🩺 Diagnose (form + risk + rules) · 📊 Model comparison · 🧠 Inside the ANFIS (membership functions, rules) · 🔵 Fuzzy patient clusters · ℹ️ About.

## Results (test set, 10,950 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|
| **ANFIS (5 inputs)** | 0.752 | 0.751 | 0.753 | 0.752 | **0.829** | 5.1 s |
| Logistic Regression (17 inputs) | 0.764 | 0.758 | 0.778 | 0.767 | 0.841 | 0.2 s |
| Random Forest (17 inputs) | 0.757 | 0.755 | 0.763 | 0.759 | 0.833 | 0.5 s |
| MLP (17 inputs) | 0.762 | 0.766 | 0.753 | 0.760 | 0.841 | 2.5 s |

The honest trade-off: ANFIS's ROC-AUC is ≈ 0.012 below the best classical model, but it uses **5 inputs instead of 17** and every prediction ships with readable rules and membership functions. Full analysis in [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Retrain from scratch

```bash
python scripts/prepare_data.py    # download + clean + balance + split
python scripts/train_anfis.py     # ANFIS model + metrics
python scripts/train_baselines.py # LogisticRegression / RandomForest / MLP
python scripts/run_fcm.py         # Fuzzy C-Means c=2..6
```

## Dataset credit

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2020, distributed via
Kaggle ["Personal Key Indicators of Heart Disease"](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) (kamilpytlak).

## Related

Earlier (classical-ML only) version of this project: <https://github.com/Priinc3/Heartattack_ML>

> ⚠️ **Disclaimer:** educational project, **not medical advice**. Predictions come from a self-reported survey dataset and must not guide real clinical decisions.
