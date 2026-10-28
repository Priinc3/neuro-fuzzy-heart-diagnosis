"""Plain-English wording for the dashboard (Phase 04).

Turns fuzzy terms ("Age is High") and artifact contents into words a
non-ML audience can read. No numbers are invented here — only labels.
"""
from __future__ import annotations

DISCLAIMER = "Educational project — not medical advice. Do not use this for real diagnosis."

# Risk bands on the gauge: green < 30%, amber 30-60%, red > 60%.
RISK_BANDS = ((0.30, "Low", "green"), (0.60, "Moderate", "orange"), (1.01, "High", "red"))

GAUGE_COLORS = {
    "Low": {"band": "#e7f5e9", "bar": "#2e8b57", "text": "green"},
    "Moderate": {"band": "#fdf3d7", "bar": "#e0a215", "text": "orange"},
    "High": {"band": "#fbe3e3", "bar": "#d64550", "text": "red"},
}


def risk_label(p: float) -> tuple[str, str]:
    """(label, streamlit-color-name) for a probability in [0,1]."""
    for hi, label, color in RISK_BANDS:
        if p < hi:
            return label, color
    return "High", "red"


# (feature, mf_label) -> plain words used inside fired rules.
TERM_WORDS = {
    ("Age", "Low"): "younger age",
    ("Age", "High"): "older age",
    ("GenHealth", "Low"): "good general health",
    ("GenHealth", "High"): "poor general health",
    ("BMI", "Low"): "low BMI",
    ("BMI", "High"): "high BMI",
    ("PhysicalHealth", "Low"): "few physically-unhealthy days",
    ("PhysicalHealth", "High"): "many physically-unhealthy days",
    ("Comorbidity", "Low"): "few coexisting conditions",
    ("Comorbidity", "High"): "several coexisting conditions",
}


def term_word(feature: str, label: str) -> str:
    """Plain words for one fuzzy term; falls back to 'f is Label'."""
    return TERM_WORDS.get((feature, str(label)), f"{feature} is {label}")


def rule_to_words(rule: dict, feature_names: list[str] | None = None) -> str:
    """Rewrite one fired rule dict (from ANFIS.top_rules) into plain words.

    Uses rule["terms"] = [(feature, label), ...] when present; otherwise
    falls back to the raw text so the tab never crashes on a new format.
    """
    terms = rule.get("terms")
    if not terms:
        return str(rule.get("text", "fired rule"))
    if feature_names is None:
        feature_names = [t[0] for t in terms]
    words = [term_word(f, l) for f, l in terms]
    risk = rule.get("output")
    tail = f" then estimated risk ≈ {float(risk) * 100:.0f}%" if risk is not None else ""
    return "If the patient is " + " and ".join(words) + "," + tail


MF_DESCRIPTION = (
    "Each input is described by two Gaussian membership functions (Low / High). "
    "A value can belong to both at once — that softness is what makes it fuzzy."
)

LAYERS_MD = """**ANFIS is a 5-layer network** (Jang, 1993):

| Layer | Name | What happens here |
|---|---|---|
| 1 | **Input / fuzzify** | Each of the 5 inputs is scored by Gaussian membership functions — "how High is this BMI?" returns a degree 0–1. |
| 2 | **Rule firing** | Every combination of memberships is multiplied (fuzzy AND) → firing strength of each rule, e.g. *IF age High AND BMI High THEN risk high*. |
| 3 | **Normalize** | Firing strengths are divided by their sum → relative rule weights. |
| 4 | **Consequent** | Each rule outputs a learned linear function of the inputs (the Takagi–Sugeno part). |
| 5 | **Aggregate** | Weighted sum of rule outputs → one probability of heart disease. |

**Why it is "neuro"-fuzzy:** the membership-function centers/widths and the rule
consequents are *learned by gradient descent* like neural-network weights —
fuzzy reasoning, neural training."""

ABOUT_MD = """## What is soft computing?

**Soft computing** is a family of techniques that tolerate imprecision and
uncertainty to produce good-enough answers fast — the way humans reason.
The main members used here are **fuzzy logic** (degrees of truth instead of
yes/no), **neural networks** (learning from data) and **clustering**
(finding groups without labels). It shines when the problem is
non-linear, noisy and needs to be *explained*.

## Why ANFIS instead of a "crisp" ML model?

| | Crisp ML (LogReg / RF / MLP) | ANFIS |
|---|---|---|
| Output | a number, take it or leave it | a number **plus the rules and memberships that produced it** |
| Internal logic | opaque weights / trees | human-readable fuzzy rules ("IF age High AND BMI High …") |
| Membership | hard boundaries (age 49 ≠ 50) | soft boundaries (age 49 is 30% "High", 70% "Low") |

The comparison tab shows honestly how much accuracy, if any, is traded for
that explainability on the same test split.

## The ANFIS model here

- Hand-written in NumPy (no PyTorch) — first-order **Takagi–Sugeno** ANFIS
  (Jang 1993, DOI [10.1109/21.256541](https://doi.org/10.1109/21.256541)).
- 5 inputs (age, general health, BMI, physically-unhealthy days, count of
  coexisting conditions), 2 Gaussian membership functions each → 32 rules.
- Trained by mini-batch gradient descent, like a neural network.

## The data

- **CDC BRFSS 2020** health survey, ~319,000 U.S. respondents, via the Kaggle
  dataset *Personal Key Indicators of Heart Disease*. Target = self-reported
  heart disease (Yes/No).
- Class imbalance handled by **balanced undersampling** (all positives +
  an equal number of random negatives, seed 42 → 54,746 rows), then a
  stratified 80/20 train/test split.
- 17 raw inputs per respondent (BMI, smoking, stroke history, sleep, …).

## Fuzzy C-Means

**Fuzzy C-Means** (Bezdek 1981) clusters patients so that *every* patient
belongs to every cluster with a membership degree summing to 1 — soft
assignment instead of k-means' hard labels. The Fuzzy Partition Coefficient
(FPC) tells us how crisp the best partition is; we scan *c* = 2…6 and keep
the best.

## History

This dashboard is the soft-computing remake of an earlier classical-ML
project: [Priinc3/Heartattack_ML](https://github.com/Priinc3/Heartattack_ML)
(Logistic Regression, RandomForest, MLP — kept here as the baselines).
"""

FORM_HELP = {
    "BMI": "Body mass index, kg/m².",
    "Smoking": "Have you smoked at least 100 cigarettes in your life?",
    "AlcoholDrinking": "Heavy drinkers (adult men >14 drinks/week; women >7).",
    "Stroke": "Ever been told you had a stroke?",
    "PhysicalHealth": "Days of poor physical health in the last 30 days.",
    "MentalHealth": "Days of poor mental health in the last 30 days.",
    "DiffWalking": "Serious difficulty walking or climbing stairs?",
    "Sex": "Biological sex as recorded in the survey.",
    "AgeCategory": "Survey age bracket; converted internally to mid-point years.",
    "Race": "Self-reported race/ethnicity category from the survey.",
    "Diabetic": "Diabetes status (borderline / gestational counted separately).",
    "PhysicalActivity": "Physical activity or exercise in the last 30 days (other than the job).",
    "GenHealth": "Self-rated general health: Excellent … Poor.",
    "SleepTime": "Average hours of sleep per 24 h.",
    "Asthma": "Ever been told you have asthma?",
    "KidneyDisease": "Ever told you have kidney disease (not stones)?",
    "SkinCancer": "Ever told you had skin cancer?",
}
