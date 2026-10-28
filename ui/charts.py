"""Pure-Plotly figure builders for the dashboard (Phase 04).

Every function takes plain data (dicts/lists/frames) and returns a
plotly Figure — no Streamlit calls here, so they are easy to reuse/test.
Heart-theme colors: red = high risk, blue = low, amber = middle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.features import ANFIS_FEATURES, RANGES
from ui.text import GAUGE_COLORS, risk_label, rule_to_words

LOW_COLOR, HIGH_COLOR, BASE_COLOR = "#4c78a8", "#d64550", "#999999"
_MODEL_COLORS = {"ANFIS": "#d64550", "LogisticRegression": "#4c78a8",
                 "RandomForest": "#2e8b57", "MLP": "#e0a215"}
_METRIC_NAMES = [("accuracy", "Accuracy"), ("precision", "Precision"),
                 ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]


def _style(fig: go.Figure, legend_horizontal: bool = True) -> go.Figure:
    fig.update_layout(template="plotly_white",
                      margin=dict(l=10, r=10, t=45, b=10))
    if legend_horizontal:
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    return fig


# ----------------------------------------------------------------- diagnose
def risk_gauge(p: float, title: str) -> go.Figure:
    """Half-donut gauge, banded green/amber/red on the 30/60 thresholds."""
    label, _ = risk_label(p)
    colors = GAUGE_COLORS[label]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(p) * 100,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": colors["bar"], "thickness": 0.28},
            "steps": [
                {"range": [0, 30], "color": GAUGE_COLORS["Low"]["band"]},
                {"range": [30, 60], "color": GAUGE_COLORS["Moderate"]["band"]},
                {"range": [60, 100], "color": GAUGE_COLORS["High"]["band"]},
            ],
            "threshold": {"line": {"color": "#333", "width": 2},
                          "thickness": 0.9, "value": float(p) * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=25, r=25, t=70, b=15),
                      template="plotly_white")
    return fig


def membership_bars(mem: dict) -> go.Figure:
    """Membership degrees per feature: paired Low/High bars, 0..1."""
    rows = [{"Feature": f, "Membership function": label, "Degree": float(d)}
            for f, labels in mem.items() for label, d in labels.items()]
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Degree", y="Feature", color="Membership function",
                 orientation="h", barmode="group", text_auto=".2f",
                 color_discrete_map={"Low": LOW_COLOR, "High": HIGH_COLOR},
                 range_x=[0, 1.08], category_orders={"Feature": ANFIS_FEATURES})
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textposition="outside", cliponaxis=False)
    _style(fig)
    fig.update_layout(height=320, xaxis_title="membership degree", yaxis_title=None)
    return fig


def rule_strengths(rules: list[dict], feature_names: list[str]) -> go.Figure:
    """Top fired rules as horizontal strength bars labelled in plain words."""
    rows = [{"Rule": rule_to_words(r, feature_names),
             "Strength": float(r.get("strength", 0.0)),
             "Rule risk": float(r.get("output", np.nan))}
            for r in rules]
    df = pd.DataFrame(rows).sort_values("Strength")
    fig = px.bar(df, x="Strength", y="Rule", orientation="h", text_auto=".2f",
                 color="Rule risk", color_continuous_scale="Reds", range_x=[0, 1])
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}<br>strength %{x:.3f}<extra></extra>")
    _style(fig, legend_horizontal=False)
    fig.update_layout(height=300, xaxis_title="rule strength (fuzzy AND of memberships)",
                      yaxis_title=None, coloraxis_colorbar=dict(title="rule risk"))
    return fig


# ----------------------------------------------------------------- compare
def metrics_grouped_bars(all_metrics: dict[str, dict]) -> go.Figure:
    rows = [{"Model": m, "Metric": nice, "Score": float(d[key])}
            for m, d in all_metrics.items()
            for key, nice in _METRIC_NAMES if key in d]
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Model", y="Score", color="Metric", barmode="group",
                 text_auto=".2f", range_y=[0, 1.12],
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(textposition="outside", cliponaxis=False)
    _style(fig)
    fig.update_layout(height=420, yaxis_title="score (test split)")
    return fig


def roc_overlay(all_metrics: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    for m, d in all_metrics.items():
        roc = d.get("roc") or {}
        fpr, tpr = roc.get("fpr"), roc.get("tpr")
        if not fpr or not tpr:
            continue
        auc = d.get("roc_auc")
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC {auc:.3f})" if auc is not None else m,
                        line=dict(color=_MODEL_COLORS.get(m), width=2.5))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                    line=dict(color="#bbb", dash="dash"), hoverinfo="skip")
    fig.update_xaxes(title="false-positive rate", range=[0, 1])
    fig.update_yaxes(title="true-positive rate", range=[0, 1.02])
    _style(fig)
    fig.update_layout(height=420)
    return fig


def confusion_heatmaps(all_metrics: dict[str, dict]) -> go.Figure:
    models = list(all_metrics)
    fig = make_subplots(rows=1, cols=len(models),
                        subplot_titles=[f"{m}<br>acc {all_metrics[m].get('accuracy', float('nan')):.2f}"
                                        for m in models])
    for i, m in enumerate(models, start=1):
        cm = np.asarray(all_metrics[m].get("confusion_matrix", [[0, 0], [0, 0]]), dtype=float)
        fig.add_trace(go.Heatmap(z=cm, text=cm, texttemplate="%{text:.0f}",
                                 colorscale="Reds", showscale=False,
                                 x=["pred No", "pred Yes"], y=["actual No", "actual Yes"]),
                      row=1, col=i)
        fig.update_yaxes(autorange="reversed", row=1, col=i)
    fig.update_layout(height=300)
    return _style(fig, legend_horizontal=False)


def train_time_bars(all_metrics: dict[str, dict]) -> go.Figure:
    rows = [{"Model": m, "Seconds": float(d.get("train_seconds", np.nan))}
            for m, d in all_metrics.items()]
    df = pd.DataFrame(rows).sort_values("Seconds")
    fig = px.bar(df, x="Seconds", y="Model", orientation="h", text_auto=".1f",
                 color="Model", color_discrete_map=_MODEL_COLORS)
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x:.2f} s<extra></extra>")
    _style(fig, legend_horizontal=False)
    fig.update_layout(height=300, xaxis_title="training time (s)", yaxis_title=None,
                      showlegend=False)
    return fig


# ----------------------------------------------------------------- inside anfis
def _gauss(x: np.ndarray, c: float, s: float) -> np.ndarray:
    s = max(float(s), 1e-3)
    return np.exp(-((x - c) ** 2) / (2.0 * s ** 2))


_GENHEALTH_WORDS = {0: "Excellent", 1: "Very good", 2: "Good", 3: "Fair", 4: "Poor"}
_UNITS = {"Age": "y", "PhysicalHealth": "d"}
_TITLES = {
    "Age": "Age (18–85 y)",
    "GenHealth": "GenHealth (0=Excellent → 4=Poor)",
    "BMI": "BMI (12–60)",
    "PhysicalHealth": "Phys. unhealthy days (0–30)",
    "Comorbidity": "Coexisting conditions (0–5)",
}


def _real_ticktext(feature: str, scaled_vals) -> list[str]:
    lo, hi = RANGES[feature]
    out = []
    for v in scaled_vals:
        real = lo + float(v) * (hi - lo)
        if feature == "GenHealth":
            label = _GENHEALTH_WORDS.get(int(round(real)), f"{real:.1f}")
            out.append(f"{v:g}\n{label}")
        else:
            unit = _UNITS.get(feature, "")
            out.append(f"{v:g}\n{real:.1f}" + (f" {unit}" if unit else ""))
    return out


def mf_curves_figure(before: dict | None, after: dict | None) -> go.Figure:
    """Gaussian MFs per input over [0,1]: dashed grey = before training,
    colored = after training. Secondary tick line in real units via RANGES."""
    fig = make_subplots(rows=1, cols=5, subplot_titles=[_TITLES[f] for f in ANFIS_FEATURES],
                        horizontal_spacing=0.045)
    xs = np.linspace(0, 1, 201)
    mf_labels = ["Low", "High"]
    for i, feature in enumerate(ANFIS_FEATURES, start=1):
        for tag, params, color, dash in (("before", before, BASE_COLOR, "dash"),
                                         ("after", after, None, None)):
            if not isinstance(params, dict):
                continue
            centers = np.atleast_2d(np.asarray(params.get("centers"), dtype=float))
            sigmas = np.atleast_2d(np.asarray(params.get("sigmas"), dtype=float))
            labels = params.get("mf_labels") or mf_labels
            if centers.shape[1] < 1 or i - 1 >= centers.shape[0]:
                continue
            for j in range(centers.shape[1]):
                c, s = centers[i - 1][j], sigmas[i - 1][j]
                y = _gauss(xs, c, s)
                show = i == 1
                name = (f"{labels[j] if j < len(labels) else j} (after)" if tag == "after"
                        else "before training")
                line_color = (LOW_COLOR if str(labels[j if j < len(labels) else 0]).lower() == "low"
                              else HIGH_COLOR) if color is None else color
                fig.add_scatter(x=xs, y=y, mode="lines", name=name, legendgroup=tag,
                                showlegend=show, line=dict(color=line_color, width=2.4, dash=dash),
                                row=1, col=i)
        ticks = [0, 0.25, 0.5, 0.75, 1]
        fig.update_xaxes(tickvals=ticks, ticktext=_real_ticktext(feature, ticks), row=1, col=i)
    fig.update_yaxes(range=[-0.04, 1.1])
    _style(fig)
    fig.update_layout(height=320, legend=dict(orientation="h", y=-0.28, x=0.02))
    return fig


def history_figure(history: dict) -> go.Figure:
    """Training curve: loss (left axis) + validation AUC (right axis)."""
    loss = [float(v) for v in (history.get("loss") or [])]
    auc = [float(v) for v in (history.get("val_auc") or [])]
    fig = go.Figure()
    if loss:
        fig.add_scatter(x=list(range(1, len(loss) + 1)), y=loss, name="loss",
                        line=dict(color="#d64550", width=2.4))
    if auc:
        fig.add_scatter(x=list(range(1, len(auc) + 1)), y=auc, name="val AUC",
                        yaxis="y2", line=dict(color="#4c78a8", width=2.4, dash="dot"))
    fig.update_layout(yaxis=dict(title="loss"), yaxis2=dict(title="validation AUC",
                      overlaying="y", side="right", range=[0, 1]))
    fig.update_xaxes(title="epoch")
    _style(fig)
    fig.update_layout(height=340)
    return fig


def anfis_layers_figure() -> go.Figure:
    """Tiny schematic of the 5 ANFIS layers (boxes + arrows)."""
    layers = [("1. Fuzzify", "5 inputs × 2 Gaussians"),
              ("2. Rules", "firing strengths (AND)"),
              ("3. Normalize", "relative strengths"),
              ("4. Consequent", "linear per rule"),
              ("5. Output", "risk probability")]
    fig = go.Figure()
    n = len(layers)
    for i, (name, sub) in enumerate(layers):
        x = i
        fig.add_annotation(x=x, y=0.5, text=f"<b>{name}</b><br><span style='font-size:10px'>{sub}</span>",
                           showarrow=False, font=dict(size=12), align="center",
                           bgcolor="#f7eaea" if i in (0, n - 1) else "#ffffff",
                           bordercolor="#d64550", borderwidth=1.5, borderpad=10)
    for i in range(n - 1):
        fig.add_annotation(x=i + 0.5, y=0.5, ax=i + 0.38, ay=0.5,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowwidth=2, arrowcolor="#888")
    fig.update_xaxes(range=[-0.5, n - 0.5], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=10),
                      template="plotly_white")
    return fig


# ----------------------------------------------------------------- clusters
def fpc_figure(fpc_by_c: dict, best_c=None) -> go.Figure:
    cs = sorted(int(k) for k in fpc_by_c)
    vals = [float(fpc_by_c[str(c)]) for c in cs]
    fig = go.Figure()
    fig.add_scatter(x=cs, y=vals, mode="lines+markers+text", text=[f"{v:.3f}" for v in vals],
                    textposition="top center", name="FPC",
                    line=dict(color="#d64550", width=2.5),
                    marker=dict(size=9, color="#d64550"))
    if best_c is not None and str(best_c) in map(str, cs):
        bv = float(fpc_by_c[str(best_c)])
        fig.add_scatter(x=[int(best_c)], y=[bv], mode="markers", name=f"best c = {best_c}",
                        marker=dict(size=16, color="#4c78a8", symbol="star"))
    fig.update_xaxes(title="number of clusters c", tickvals=cs)
    fig.update_yaxes(title="fuzzy partition coefficient", range=[min(0.5, min(vals) - 0.05), 1.02])
    _style(fig)
    fig.update_layout(height=320)
    return fig


def cluster_scatter(points: pd.DataFrame, x_feat: str, y_feat: str,
                    centers=None) -> go.Figure:
    """2-D scatter of sampled patients coloured by (soft) cluster + centers."""
    df = points.copy()
    df["cluster"] = df["cluster"].astype(int).astype(str)
    fig = px.scatter(df, x=x_feat, y=y_feat, color="cluster",
                     color_discrete_sequence=px.colors.qualitative.T10,
                     hover_data={"disease": ":.0f", "cluster": False},
                     labels={"disease": "has heart disease (1=yes)"},
                     opacity=0.65, range_x=[-0.05, 1.05], range_y=[-0.05, 1.05])
    if centers is not None:
        c = np.asarray(centers, dtype=float)
        if c.ndim == 2 and c.shape[1] == len(ANFIS_FEATURES):
            for k, row in enumerate(c):
                fig.add_scatter(x=[row[ANFIS_FEATURES.index(x_feat)]],
                                y=[row[ANFIS_FEATURES.index(y_feat)]],
                                mode="markers", name=f"center {k}",
                                marker=dict(symbol="x", size=14, color="#111",
                                            line=dict(width=3)))
    fig.update_xaxes(title=f"{x_feat} (scaled 0–1)")
    fig.update_yaxes(title=f"{y_feat} (scaled 0–1)")
    _style(fig)
    fig.update_layout(height=520, legend_title="cluster")
    return fig


def cluster_rate_bars(rates: list[float], counts: list[int] | None = None) -> go.Figure:
    labels = [f"cluster {i}" for i in range(len(rates))]
    text = [f"{r * 100:.1f}%" for r in rates]
    hover = ("<b>%{x}</b><br>disease rate: %{customdata[0]:.1%}"
             + ("<br>sampled members: %{customdata[1]}" if counts else "") + "<extra></extra>")
    custom = np.column_stack([rates, counts]) if counts else np.column_stack([rates, rates])
    fig = go.Figure(go.Bar(x=labels, y=[float(r) * 100 for r in rates], text=text,
                           textposition="outside", cliponaxis=False,
                           marker_color="#d64550", customdata=custom,
                           hovertemplate=hover))
    fig.update_yaxes(title="heart-disease rate in cluster (%)", range=[0, 105])
    _style(fig, legend_horizontal=False)
    fig.update_layout(height=320, showlegend=False)
    return fig
