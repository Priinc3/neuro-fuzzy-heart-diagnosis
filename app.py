"""Neuro-Fuzzy Heart Diagnosis — Streamlit dashboard (Phase 04).

Five tabs: 🩺 Diagnose · 📊 Model comparison · 🧠 Inside the ANFIS ·
🔵 Fuzzy patient clusters · ℹ️ About.

Every tab codes against the frozen contract in prompts/00-plan.md and
shows a friendly placeholder instead of crashing when an artifact has
not been produced yet (siblings 02-anfis / 03-baselines-fcm run in
parallel). Educational demo — not medical advice.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # importable no matter where Streamlit starts
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.features import (AGE_MIDPOINTS, ANFIS_FEATURES, GENHEALTH_MAP,
                          anfis_matrix, raw_from_form)
import ui.charts as C
import ui.data as D
import ui.text as T

st.set_page_config(page_title="Neuro-Fuzzy Heart Diagnosis", page_icon="❤️", layout="wide")

st.title("🫀 Neuro-Fuzzy Heart Diagnosis")
st.caption(
    "ANFIS (neuro-fuzzy) heart-disease risk with the fuzzy rules explained in words — "
    "CDC BRFSS 2020. Educational demo, not medical advice."
)
tabs = st.tabs(["🩺 Diagnose", "📊 Model comparison", "🧠 Inside the ANFIS",
                "🔵 Fuzzy patient clusters", "ℹ️ About"])

YES_NO = ["No", "Yes"]
DEFAULTS = {
    "BMI": 27.4, "PhysicalHealth": 0, "MentalHealth": 0, "SleepTime": 7,
    "AgeCategory": list(AGE_MIDPOINTS)[7], "Sex": "Female",
    "Race": "White", "GenHealth": list(GENHEALTH_MAP)[2],
    "Smoking": "No", "AlcoholDrinking": "No", "Diabetic": "No",
    "PhysicalActivity": "Yes", "Stroke": "No", "DiffWalking": "No",
    "Asthma": "No", "KidneyDisease": "No", "SkinCancer": "No",
}


def _form_state() -> dict:
    """Current form values from widget keys (defaults before first render)."""
    return {k: st.session_state.get("f_" + k, v) for k, v in DEFAULTS.items()}


def _model_input() -> tuple[pd.DataFrame, np.ndarray] | None:
    """(raw 1-row df, scaled (1,5) matrix) or None with a plain error shown."""
    try:
        raw = raw_from_form(_form_state())
        return raw, anfis_matrix(raw)
    except Exception:
        st.error("Could not build the model input from those answers — please "
                 "check the values and try again.")
        return None


# ================================================================= tab 1
def render_diagnose() -> None:
    st.subheader("🩺 Diagnose")
    st.caption("Enter the 17 survey answers; the ANFIS turns them into a risk "
               "probability and shows the fuzzy reasoning behind it.")

    with st.form("patient_form", border=True):
        c = st.columns(4)
        c[0].number_input("BMI", 12.0, 60.0, DEFAULTS["BMI"], 0.1,
                                        format="%.1f", key="f_BMI",
                                        help=T.FORM_HELP["BMI"])
        c[1].number_input("Poor physical-health days (of 30)",
                                                   0, 30, DEFAULTS["PhysicalHealth"],
                                                   key="f_PhysicalHealth",
                                                   help=T.FORM_HELP["PhysicalHealth"])
        c[2].number_input("Poor mental-health days (of 30)",
                                                 0, 30, DEFAULTS["MentalHealth"],
                                                 key="f_MentalHealth",
                                                 help=T.FORM_HELP["MentalHealth"])
        c[3].number_input("Sleep (h/day)", 0, 24, DEFAULTS["SleepTime"],
                                              key="f_SleepTime", help=T.FORM_HELP["SleepTime"])

        c = st.columns(4)
        c[0].selectbox("Age category", list(AGE_MIDPOINTS), index=7,
                                             key="f_AgeCategory",
                                             help=T.FORM_HELP["AgeCategory"])
        c[1].selectbox("Sex", ["Female", "Male"], key="f_Sex",
                                     help=T.FORM_HELP["Sex"])
        c[2].selectbox("Race", ["White", "Black", "Hispanic", "Asian",
                                               "American Indian/Alaskan Native", "Other"],
                                      key="f_Race", help=T.FORM_HELP["Race"])
        c[3].selectbox("General health", list(GENHEALTH_MAP), index=2,
                                           key="f_GenHealth", help=T.FORM_HELP["GenHealth"])

        c = st.columns(4)
        c[0].selectbox("Smoking", YES_NO, key="f_Smoking",
                                         help=T.FORM_HELP["Smoking"])
        c[1].selectbox("Heavy alcohol drinking", YES_NO,
                                                 key="f_AlcoholDrinking",
                                                 help=T.FORM_HELP["AlcoholDrinking"])
        c[2].selectbox("Diabetic", ["No", "No, borderline diabetes",
                                                       "Yes", "Yes (during pregnancy)"],
                                          key="f_Diabetic", help=T.FORM_HELP["Diabetic"])
        c[3].selectbox("Physical activity (last 30d)",
                                                  ["Yes", "No"], key="f_PhysicalActivity",
                                                  help=T.FORM_HELP["PhysicalActivity"])

        c = st.columns(5)
        c[0].selectbox("Stroke", YES_NO, key="f_Stroke",
                                        help=T.FORM_HELP["Stroke"])
        c[1].selectbox("Difficulty walking", YES_NO,
                                             key="f_DiffWalking",
                                             help=T.FORM_HELP["DiffWalking"])
        c[2].selectbox("Asthma", YES_NO, key="f_Asthma",
                                        help=T.FORM_HELP["Asthma"])
        c[3].selectbox("Kidney disease", YES_NO,
                                               key="f_KidneyDisease",
                                               help=T.FORM_HELP["KidneyDisease"])
        c[4].selectbox("Skin cancer", YES_NO, key="f_SkinCancer",
                                            help=T.FORM_HELP["SkinCancer"])
        submitted = st.form_submit_button("❤️ Assess risk", type="primary", width="stretch")

    if submitted:
        st.session_state["assessed"] = True  # keeps the explanation visible across reruns

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🧠 Neuro-fuzzy (ANFIS)**")
        if not st.session_state.get("assessed"):
            st.info("Fill the form and press **Assess risk** — nothing is stored or logged.")
        else:
            built = _model_input()
            if built is not None:
                raw, X = built
                model, msg = D.load_anfis()
                if model is None:
                    st.info(msg)
                else:
                    p_arr = D.predict_proba(model, X)
                    if p_arr is None:
                        st.error("Prediction failed — the model file may be from an older "
                                 "run. Re-run `python scripts/train_anfis.py`.")
                    else:
                        p = float(np.clip(p_arr[0], 0.0, 1.0))
                        label, color = T.risk_label(p)
                        st.plotly_chart(C.risk_gauge(p, "heart-disease risk"), key="gauge_anfis")
                        st.markdown(f"That is **:{color}[{label} risk]** ({p * 100:.1f}%).")
    with col_b:
        st.markdown("**⚙️ Classical ML (LogReg) says…**")
        if not st.session_state.get("assessed"):
            st.info("The Logistic Regression baseline answer appears here too, for a "
                    "crisp-vs-fuzzy comparison.")
        else:
            built = _model_input()
            if built is not None:
                raw, _ = built
                lr, lr_msg = D.load_lr()
                if lr is None:
                    st.info(lr_msg)
                else:
                    try:
                        plr = float(np.asarray(lr.predict_proba(raw)).reshape(-1)[1])
                    except Exception:
                        st.info("LogReg baseline unavailable — re-run "
                                "`python scripts/train_baselines.py`.")
                    else:
                        label_lr, color_lr = T.risk_label(plr)
                        st.plotly_chart(C.risk_gauge(plr, "LogReg baseline"), key="gauge_lr")
                        st.markdown(f"LogReg: **{plr * 100:.1f}%** — "
                                    f"**:{color_lr}[{label_lr} risk]**. Same patient, "
                                    "crisp (non-fuzzy) model.")

    st.subheader("Why? — the fuzzy reasoning")
    model, msg = D.load_anfis()
    if model is None:
        st.info(msg)
        st.caption(T.DISCLAIMER)
        return

    if not st.session_state.get("assessed"):
        st.info("Press **Assess risk** to see the membership degrees and fired rules "
                "for that patient here.")
        st.caption(T.DISCLAIMER)
        return

    built = _model_input()
    if built is None:
        return
    _, X = built

    mem = D.memberships(model, X[0])
    if mem:
        st.markdown(T.MF_DESCRIPTION)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(C.membership_bars(mem), key="mem_bars")
        with c2:
            st.dataframe(pd.DataFrame(mem).T.style.format("{:.3f}"),
                         key="mem_table")
            st.caption("Each cell = how much this patient belongs to that fuzzy "
                       "label (Low / High). A 49-year-old is partly 'younger', "
                       "partly 'older' — no hard cut-off.")
    else:
        st.info("Membership degrees not available — re-run "
                "`python scripts/train_anfis.py` to regenerate the model.")

    rules = D.top_rules(model, X[0], k=5)
    if rules:
        st.markdown("**Top fired rules, in plain words** — strength = how strongly "
                    "the rule fired for this patient (fuzzy AND of its memberships):")
        strengths = [float(r.get("strength", 0.0)) for r in rules]
        top = max(strengths) or 1.0
        for r, s in zip(rules, strengths):
            words = T.rule_to_words(r, list(mem.keys()) if mem else None)
            out = r.get("output")
            tail = f" → rule risk **{float(out) * 100:.0f}%**" if out is not None else ""
            st.markdown(f"- {words}{tail}")
            st.progress(min(1.0, s / top), text=f"strength {s:.2f}")
    else:
        st.info("Fired rules not available — re-run `python scripts/train_anfis.py`.")

    st.caption(T.DISCLAIMER)


# ================================================================= tab 2
def render_compare() -> None:
    st.subheader("📊 Model comparison — ANFIS vs crisp ML")
    anfis_m, a_msg = D.load_metrics_anfis()
    base_m, b_msg = D.load_metrics_baselines()
    all_metrics: dict[str, dict] = {}
    if anfis_m:
        all_metrics["ANFIS"] = anfis_m
    for name in ("LogisticRegression", "RandomForest", "MLP"):
        if base_m and name in base_m:
            all_metrics[name] = base_m[name]

    if not all_metrics:
        st.info(a_msg or b_msg or "No metrics yet — run the training scripts.")
        return

    cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train (s)"]
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"]
    rows = [[m] + [float(d.get(k, np.nan)) for k in keys] for m, d in all_metrics.items()]
    df = pd.DataFrame(rows, columns=["Model"] + cols).set_index("Model")

    styled = (df.style
              .format({c: "{:.3f}" for c in cols[:-1]} | {"Train (s)": "{:.1f}"})
              .highlight_max(subset=cols[:-1],
                             props="background-color:#fdeaea;color:#8f1d2c;font-weight:600;")
              .highlight_min(subset=["Train (s)"],
                             props="background-color:#eaf5ec;color:#1e5c2e;font-weight:600;"))
    st.dataframe(styled, key="metrics_table")
    st.caption("Best value per column highlighted (fastest trainer for time). "
               "All models use the same balanced data and 80/20 split.")

    # One-line takeaway computed from the numbers (no hardcoded claims).
    best_auc = max(all_metrics, key=lambda m: all_metrics[m].get("roc_auc", -1.0))
    best_recall = max(all_metrics, key=lambda m: all_metrics[m].get("recall", -1.0))
    fastest = min(all_metrics, key=lambda m: all_metrics[m].get("train_seconds", np.inf))
    line = (f"On this test split, **{best_auc}** leads ROC-AUC "
            f"({all_metrics[best_auc].get('roc_auc', float('nan')):.3f}), "
            f"**{best_recall}** catches the most true heart-disease cases (recall "
            f"{all_metrics[best_recall].get('recall', float('nan')):.3f}), and "
            f"**{fastest}** trained fastest ({all_metrics[fastest].get('train_seconds', float('nan')):.1f} s).")
    if "ANFIS" in all_metrics:
        gap = (all_metrics[best_auc].get("roc_auc", 0.0)
               - all_metrics["ANFIS"].get("roc_auc", 0.0))
        line += (" ANFIS is on top — the explainability comes free."
                 if gap <= 0 else
                 f" ANFIS trades {gap:.3f} ROC-AUC for its explainable fuzzy rules.")
    st.info(line)

    st.plotly_chart(C.metrics_grouped_bars(all_metrics), key="metrics_bars")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(C.roc_overlay(all_metrics), key="roc")
    with c2:
        st.plotly_chart(C.train_time_bars(all_metrics), key="time")
    st.plotly_chart(C.confusion_heatmaps(all_metrics), key="cm")


# ================================================================= tab 3
def render_inside() -> None:
    st.subheader("🧠 Inside the ANFIS")
    model, m_msg = D.load_anfis()
    init, i_msg = D.load_init_mf()
    after: dict | None = None
    if model is not None:
        try:
            after = model.mf_params()
        except Exception:
            after = None

    st.markdown("#### Membership functions — before vs after training")
    if init is None and after is None:
        st.info(i_msg or m_msg or "Model not trained yet — run `python scripts/train_anfis.py`")
    else:
        st.markdown(T.MF_DESCRIPTION)
        if after is None:
            st.caption("Showing the **initial (untrained)** membership functions — "
                       "train the model to see how they move.")
        st.plotly_chart(C.mf_curves_figure(init, after), key="mf")
        try:
            params = after or init
            n_mf = int(np.asarray(params["centers"], dtype=float).shape[1])
        except Exception:
            n_mf = 2
        st.caption(f"Architecture: 5 inputs × {n_mf} Gaussians → {n_mf}⁵ = {n_mf ** 5} rules. "
                   "The top tick row on each panel is the feature's real units "
                   "(from the frozen RANGES); the bottom row is the scaled 0–1 "
                   "value the network actually sees.")

    st.markdown("#### Training curve")
    hist, h_msg = D.load_history()
    if hist and (hist.get("loss") or hist.get("val_auc")):
        st.plotly_chart(C.history_figure(hist), key="hist")
        losses = [v for v in (hist.get("loss") or []) if v == v]
        aucs = [v for v in (hist.get("val_auc") or []) if v == v]
        if losses and aucs:
            st.caption(f"Loss fell {losses[0]:.3f} → {losses[-1]:.3f} while validation AUC "
                       f"reached {aucs[-1]:.3f} over {len(losses)} epochs — the membership "
                       "functions and rule consequents are learned by gradient descent, "
                       "like neural-network weights.")
    else:
        st.info(h_msg or "No training history — run `python scripts/train_anfis.py`")

    st.markdown("#### The 5 layers")
    st.plotly_chart(C.anfis_layers_figure(), key="layers")
    st.markdown(T.LAYERS_MD)


# ================================================================= tab 4
def render_clusters() -> None:
    st.subheader("🔵 Fuzzy patient clusters (Fuzzy C-Means)")
    fcm, msg = D.load_fcm()
    if fcm is None:
        st.info(msg or "Clusters not computed yet — run `python scripts/run_fcm.py`")
        return

    fpc = fcm.get("fpc_by_c") or {}
    if fpc:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(C.fpc_figure(fpc, fcm.get("best_c")), key="fpc")
        with c2:
            best_c = fcm.get("best_c")
            best_val = float(fpc.get(str(best_c), max(fpc.values())))
            st.metric("Chosen clusters (c)", best_c,
                      help="Highest Fuzzy Partition Coefficient wins.")
            st.caption(f"FPC = {best_val:.3f}. FPC runs from 1/c toward 1 — closer to 1 "
                       "means crisper, more separate clusters, but every patient still "
                       "keeps a **soft** membership in every cluster.")
    else:
        st.info("No FPC scan found in `fcm.json`.")

    points, reason = D.points_frame(fcm)
    if points is None:
        st.info(f"Patient scatter unavailable ({reason}).")
    else:
        c1, c2 = st.columns(2)
        x_feat = c1.selectbox("X feature", ANFIS_FEATURES, index=2, key="sc_x")
        y_feat = c2.selectbox("Y feature", ANFIS_FEATURES, index=0, key="sc_y")
        st.plotly_chart(C.cluster_scatter(points, x_feat, y_feat, fcm.get("centers")),
                        key="scatter")
        st.caption("Each dot is a sampled patient (≤2000) in two scaled features; "
                   "✕ marks the learned cluster centers. Patients near a boundary "
                   "belong partially to *both* clusters — that softness is the "
                   "'fuzzy' part.")

    rates = fcm.get("cluster_disease_rate")
    if isinstance(rates, list) and rates:
        counts = ([int((points["cluster"] == i).sum()) for i in range(len(rates))]
                  if points is not None else None)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(C.cluster_rate_bars(rates, counts), key="rate_bars")
        with c2:
            st.dataframe(pd.DataFrame({
                "Cluster": [f"cluster {i}" for i in range(len(rates))],
                "Disease rate": [f"{r * 100:.1f}%" for r in rates],
                "Sampled members": counts or ["—"] * len(rates),
            }), hide_index=True, key="rate_table")
            st.caption("Heart-disease rate per cluster — clusters separate risk levels "
                       "without using labels (unsupervised).")
    else:
        st.info("No cluster disease rates found in `fcm.json`.")


# ================================================================= tab 5
def render_about() -> None:
    st.markdown(T.ABOUT_MD)
    st.info(T.DISCLAIMER)


# ----------------------------------------------------------------- wiring
with tabs[0]:
    render_diagnose()
with tabs[1]:
    render_compare()
with tabs[2]:
    render_inside()
with tabs[3]:
    render_clusters()
with tabs[4]:
    render_about()
