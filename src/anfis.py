"""ANFIS — first-order Takagi-Sugeno neuro-fuzzy classifier (Jang 1993, DOI 10.1109/21.256541).

SKELETON written by Phase 01. Phase 02 implements the bodies.
The signatures below are the shared contract used by phases 03 and 04 too.

Architecture (grid partition, n_mfs Gaussian MFs per input, x scaled to [0,1]):
  L1  membership:    mu[i, j]  = exp(-(x_i - c_ij)^2 / (2 * sigma_ij^2))
  L2  rule firing:   w_r = prod_i mu[i, j(r, i)]          (2^5 = 32 rules for 5 inputs x 2 MFs)
  L3  normalize:     wbar_r = w_r / sum_s w_s
  L4  consequent:    f_r(x) = p_r . x + r_r
  L5  output:        logit = sum_r wbar_r f_r(x),  P(disease) = sigmoid(logit)
Trained end-to-end by mini-batch Adam on binary cross-entropy with analytic
gradients for c, sigma, p, r (pure NumPy; sklearn used for the val metric only).
"""
# FROZEN CONTRACT — do not change signatures
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.features import ANFIS_FEATURES

_SEED = 42
_EPS = 1e-12
_SIGMA_MIN = 0.05
_INIT_CENTERS = {2: (0.25, 0.75), 3: (0.20, 0.50, 0.80)}
_DEFAULT_LABELS = {2: ("Low", "High"), 3: ("Low", "Medium", "High")}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for 1-D arrays."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class _Adam:
    """Minimal Adam optimizer updating float arrays in place via references."""

    def __init__(self, params: dict[str, np.ndarray], lr: float):
        self.params = params  # dict of references to the live parameter arrays
        self.lr = float(lr)
        self.beta1, self.beta2, self.eps, self.t = 0.9, 0.999, 1e-8, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * g * g
            mhat = self.m[k] / (1.0 - self.beta1**self.t)
            vhat = self.v[k] / (1.0 - self.beta2**self.t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class ANFIS:
    """First-order Takagi-Sugeno ANFIS: Gaussian membership functions per input,
    product (AND) rule layer, normalized rule strengths, learned linear consequents,
    trained by gradient descent like a neural network."""

    def __init__(self, n_inputs=5, n_mfs=2, mf_labels=("Low", "High"), feature_names=ANFIS_FEATURES):
        """Store the architecture and initialize Gaussian MF params (centers/sigmas)
        and rule consequents to small random values. n_mfs MFs per input."""
        self.n_inputs = int(n_inputs)
        self.n_mfs = int(n_mfs)
        if len(mf_labels) != self.n_mfs:
            mf_labels = _DEFAULT_LABELS.get(self.n_mfs, tuple(f"MF{j}" for j in range(self.n_mfs)))
        self.mf_labels = tuple(mf_labels)
        self.feature_names = list(feature_names)[: self.n_inputs]

        self._n_rules = self.n_mfs**self.n_inputs
        # Rule book: rule r fires MF digit i of r (base n_mfs) on input i.
        rules = np.arange(self._n_rules)
        self._rule_idx = np.stack(
            [(rules // self.n_mfs**i) % self.n_mfs for i in range(self.n_inputs)], axis=1
        )  # (n_rules, n_inputs)

        init_c = _INIT_CENTERS.get(self.n_mfs)
        if init_c is None or len(init_c) != self.n_mfs:
            init_c = tuple(np.linspace(0.25, 0.75, self.n_mfs))
        self.centers = np.tile(np.asarray(init_c, dtype=float)[None, :], (self.n_inputs, 1))
        self.sigmas = np.full((self.n_inputs, self.n_mfs), 0.25)

        self._rng = np.random.default_rng(_SEED)
        self.p = self._rng.normal(0.0, 0.01, size=(self._n_rules, self.n_inputs))
        self.r = self._rng.normal(0.0, 0.01, size=self._n_rules)

    # ------------------------------------------------------------------ layers
    def _mu(self, x: np.ndarray) -> np.ndarray:
        """L1 Gaussian memberships. x: (..., n_inputs) -> (..., n_inputs, n_mfs)."""
        return np.exp(-((x[..., :, None] - self.centers) ** 2) / (2.0 * self.sigmas**2))

    def _forward(self, X: np.ndarray) -> dict:
        """Full forward pass for a batch; returns layer intermediates."""
        n, d = X.shape
        mu = self._mu(X)  # (n, d, m)
        # Gather the per-rule memberships: mg[n, r, i] = mu[n, i, rule_idx[r, i]]
        mg = mu[
            np.arange(n)[:, None, None],
            np.arange(d)[None, None, :],
            self._rule_idx[None, :, :],
        ]  # (n, n_rules, n_inputs)
        w = mg.prod(axis=2)  # (n, R) L2 rule firing strengths
        W = w.sum(axis=1) + _EPS
        wbar = w / W[:, None]  # (n, R) L3 normalized
        f = X @ self.p.T + self.r  # (n, R) L4 consequents p.x + r
        z = (wbar * f).sum(axis=1)  # (n,) L5 logit
        return {"mu": mu, "mg": mg, "w": w, "W": W, "wbar": wbar, "f": f, "z": z}

    # ---------------------------------------------------------------- training
    def _loss_grads(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        """Mean BCE loss and analytic gradients for centers, sigmas, p, r."""
        F = self._forward(X)
        n = X.shape[0]
        z = F["z"]
        p_hat = _sigmoid(z)
        pc = np.clip(p_hat, 1e-7, 1.0 - 1e-7)
        loss = float(-(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)).mean())

        gz = (p_hat - y) / n  # (n,) dL/dlogit
        w, W, wbar, f = F["w"], F["W"], F["wbar"], F["f"]
        # dz/dw_r = (f_r - z)/W   ->   dL/dw_r
        gw = gz[:, None] * (f - z[:, None]) / W[:, None]  # (n, R)
        # dz/dp_{r,i} = wbar_r * x_i ;  dz/dr_r = wbar_r
        gp = (wbar * gz[:, None]).T @ X  # (R, d)
        gr = (wbar * gz[:, None]).sum(axis=0)  # (R,)
        # dw_r/dc_ij = w_r (x_i - c_ij)/sigma_ij^2 [rule r uses MF j on input i]
        # dw_r/ds_ij = w_r (x_i - c_ij)^2/sigma_ij^3 [same]
        D = gw * w  # (n, R) dL/dw * w
        gC = np.zeros_like(self.centers)
        gS = np.zeros_like(self.sigmas)
        for i in range(self.n_inputs):
            oh = (self._rule_idx[:, i][:, None] == np.arange(self.n_mfs)[None, :]).astype(float)
            xi = X[:, i]
            sD = D.sum(axis=0)  # (R,)  sum_n D * w
            sDx = (D * xi[:, None]).sum(axis=0)  # (R,)  sum_n D * w * x_i
            sDx2 = (D * (xi * xi)[:, None]).sum(axis=0)  # (R,)  sum_n D * w * x_i^2
            cj = self.centers[i]  # (m,)
            num_c = (oh * (sDx[:, None] - cj[None, :] * sD[:, None])).sum(axis=0)  # (m,)
            num_s = (
                oh
                * (
                    sDx2[:, None]
                    - 2.0 * cj[None, :] * sDx[:, None]
                    + cj[None, :] ** 2 * sD[:, None]
                )
            ).sum(axis=0)  # (m,)
            gC[i] = num_c / self.sigmas[i] ** 2
            gS[i] = num_s / self.sigmas[i] ** 3
        return loss, {"centers": gC, "sigmas": gS, "p": gp, "r": gr}

    def fit(self, X, y, epochs=40, lr=0.01, batch_size=512, X_val=None, y_val=None) -> dict:
        """Train the MF params and rule consequents by mini-batch gradient descent.
        Returns history dict {"loss": [...], "val_auc": [...]} (per epoch)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        opt = _Adam(
            {"centers": self.centers, "sigmas": self.sigmas, "p": self.p, "r": self.r}, lr
        )
        history: dict = {"loss": [], "val_auc": []}
        for _ in range(int(epochs)):
            perm = self._rng.permutation(n)
            total = 0.0
            for start in range(0, n, int(batch_size)):
                idx = perm[start : start + int(batch_size)]
                loss, grads = self._loss_grads(X[idx], y[idx])
                opt.step(grads)
                np.maximum(self.sigmas, _SIGMA_MIN, out=self.sigmas)  # clip sigma >= 0.05
                np.clip(self.centers, 0.0, 1.0, out=self.centers)  # keep MFs inside [0,1]
                total += loss * idx.size
            history["loss"].append(total / n)
            if X_val is not None and y_val is not None:
                history["val_auc"].append(float(roc_auc_score(y_val, self.predict_proba(X_val))))
        return history

    # ------------------------------------------------------------- inference
    def predict_proba(self, X) -> np.ndarray:
        """Forward pass -> P(disease) for each row, shape (n,)."""
        X = np.asarray(X, dtype=float)
        return _sigmoid(self._forward(X)["z"])

    def memberships(self, x_row) -> dict:
        """Gaussian membership degrees for ONE input row (scaled [0,1] space):
        {feature: {label: degree}} e.g. {"Age": {"Low": 0.18, "High": 0.82}, ...}."""
        x = np.asarray(x_row, dtype=float).ravel()
        mu = self._mu(x)  # (n_inputs, n_mfs)
        return {
            self.feature_names[i]: {
                self.mf_labels[j]: float(mu[i, j]) for j in range(self.n_mfs)
            }
            for i in range(self.n_inputs)
        }

    def top_rules(self, x_row, k=5) -> list[dict]:
        """Top-k fired rules for ONE input row, strongest first:
        [{"text": "IF Age is High AND ... THEN risk=0.81", "terms": [("Age", "High"), ...],
          "strength": 0.42, "output": 0.81}]."""
        x = np.asarray(x_row, dtype=float).ravel()
        mu = self._mu(x)  # (d, m)
        mg = mu[np.arange(self.n_inputs)[None, :], self._rule_idx]  # (R, d)
        w = mg.prod(axis=1)  # (R,)
        wbar = w / (w.sum() + _EPS)
        f = self.p @ x + self.r  # (R,) per-rule consequent output at x
        out = _sigmoid(f)
        order = np.argsort(-wbar, kind="stable")[: int(k)]
        rules: list[dict] = []
        for t in order:
            terms = [
                (self.feature_names[i], self.mf_labels[self._rule_idx[t, i]])
                for i in range(self.n_inputs)
            ]
            text = (
                "IF " + " AND ".join(f"{feat} is {lab}" for feat, lab in terms)
                + f" THEN risk = {out[t]:.2f}"
            )
            rules.append(
                {"text": text, "terms": terms, "strength": float(wbar[t]), "output": float(out[t])}
            )
        return rules

    # ----------------------------------------------------------- persistence
    def mf_params(self) -> dict:
        """Learned membership-function parameters:
        {"centers": (n_inputs, n_mfs) nested list, "sigmas": same shape,
         "feature_names": [...], "mf_labels": [...]}."""
        return {
            "centers": self.centers.tolist(),
            "sigmas": self.sigmas.tolist(),
            "feature_names": list(self.feature_names),
            "mf_labels": list(self.mf_labels),
        }

    def save(self, path):
        """Serialize all learned parameters to a .npz file."""
        np.savez(
            path,
            centers=self.centers,
            sigmas=self.sigmas,
            p=self.p,
            r=self.r,
            feature_names=np.asarray(self.feature_names, dtype=str),
            mf_labels=np.asarray(self.mf_labels, dtype=str),
        )

    @classmethod
    def load(cls, path) -> "ANFIS":
        """Deserialize an ANFIS from a .npz file written by save()."""
        data = np.load(path)
        centers = np.asarray(data["centers"], dtype=float)
        model = cls(
            n_inputs=centers.shape[0],
            n_mfs=centers.shape[1],
            mf_labels=tuple(str(v) for v in data["mf_labels"]),
            feature_names=[str(v) for v in data["feature_names"]],
        )
        model.centers = centers
        model.sigmas = np.asarray(data["sigmas"], dtype=float)
        model.p = np.asarray(data["p"], dtype=float)
        model.r = np.asarray(data["r"], dtype=float)
        return model
