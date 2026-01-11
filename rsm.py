"""
最終最強・完全版（1ファイル）
============================================================
★「ノイズ二重カウント回避」＋「sklearn挙動差の自動吸収（自己診断）」統合済み

追加した自己診断（重要）
- sklearn の GPR(return_std) が
    A) latent の不確実性だけ を返す環境
    B) WhiteKernel の観測ノイズ込み を返す環境
  のどちらでも “安全側に”動くように、
  返ってきた std と WhiteKernel 推定ノイズを比較して、
  引き算（latent復元）を自動ON/OFFする。

依存：numpy, scikit-learn
（plotlyは任意）
SciPy不要。
============================================================
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any, Set

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.base import clone as sk_clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    from scipy.spatial.distance import cdist as _cdist
    from scipy.special import erf as _scipy_erf
    from scipy.stats import norm as _scipy_norm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


# ============================================================
# Normal helpers (SciPy推奨だがフォールバック可能)
# ============================================================

def _phi(z: np.ndarray) -> np.ndarray:
    """標準正規分布のPDF（確率密度関数）"""
    if _HAS_SCIPY:
        return _scipy_norm.pdf(z)
    z = np.clip(np.asarray(z), -100, 100)
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def _Phi(z: np.ndarray) -> np.ndarray:
    """標準正規分布のCDF（累積分布関数）"""
    if _HAS_SCIPY:
        return _scipy_norm.cdf(z)
    z = np.clip(np.asarray(z), -100, 100)
    # erfはscipyがあれば高速、なければmath.erfのベクトル化版を使用
    if _HAS_SCIPY:
        return 0.5 * (1.0 + _scipy_erf(z / np.sqrt(2.0)))
    v_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + v_erf(z / np.sqrt(2.0)))

def expected_improvement(
    mu: np.ndarray,
    std: np.ndarray,
    best: float,
    objective: str = "min",
    xi: float = 0.01,
) -> np.ndarray:
    std = np.maximum(np.asarray(std), 1e-12)
    mu = np.asarray(mu)
    if objective == "min":
        imp = (best - mu) - xi
    elif objective == "max":
        imp = (mu - best) - xi
    else:
        raise ValueError("objective must be 'min' or 'max'")
    Z = imp / std
    ei = imp * _Phi(Z) + std * _phi(Z)
    return np.maximum(ei, 0.0)

def z_value(level: float) -> float:
    table = {
        0.80: 1.281551565545,
        0.90: 1.644853626951,
        0.95: 1.959963984540,
        0.975: 2.241402727605,
        0.99: 2.575829303549,
    }
    if level in table:
        return table[level]
    if level >= 0.99:
        return 2.58
    if level >= 0.975:
        return 2.24
    if level >= 0.95:
        return 1.96
    if level >= 0.90:
        return 1.645
    return 1.28


# ============================================================
# Stepwise by partial F with hierarchy (Strong Heredity)
# ============================================================

@dataclass(frozen=True)
class _Term:
    kind: str            # "lin" | "quad" | "inter"
    idxs: tuple          # (i,) or (i,j)
    name: str
    col: int

def _build_terms_from_poly(powers: np.ndarray, feature_names: List[str], include_bias: bool):
    intercept_col = None
    terms: Dict[Tuple[str, tuple], _Term] = {}

    for col, p in enumerate(powers):
        if np.all(p == 0):
            intercept_col = col
            continue
        nz = np.where(p != 0)[0]
        if len(nz) == 1:
            i = int(nz[0])
            if p[i] == 1:
                k = ("lin", (i,))
                terms[k] = _Term("lin", (i,), feature_names[i], col)
            elif p[i] == 2:
                k = ("quad", (i,))
                terms[k] = _Term("quad", (i,), f"{feature_names[i]}^2", col)
        elif len(nz) == 2:
            i, j = int(nz[0]), int(nz[1])
            if p[i] == 1 and p[j] == 1:
                a, b = min(i, j), max(i, j)
                k = ("inter", (a, b))
                terms[k] = _Term("inter", (a, b), f"{feature_names[a]}*{feature_names[b]}", col)

    requires: Dict[Tuple[str, tuple], Set[Tuple[str, tuple]]] = {k: set() for k in terms.keys()}
    for k, t in terms.items():
        if t.kind == "quad":
            i = t.idxs[0]
            requires[k].add(("lin", (i,)))
        elif t.kind == "inter":
            i, j = t.idxs
            requires[k].add(("lin", (i,)))
            requires[k].add(("lin", (j,)))

    return intercept_col, terms, requires

def _closure(selected: Set[Tuple[str, tuple]], requires: Dict[Tuple[str, tuple], Set[Tuple[str, tuple]]]):
    closed = set(selected)
    changed = True
    while changed:
        changed = False
        for k in list(closed):
            for req in requires.get(k, set()):
                if req not in closed:
                    closed.add(req)
                    changed = True
    return closed

def _mandatory_terms(current: Set[Tuple[str, tuple]], requires: Dict[Tuple[str, tuple], Set[Tuple[str, tuple]]]):
    mand = set()
    for k in current:
        mand |= requires.get(k, set())
    return mand

def _ols_rss_df(y: np.ndarray, X: np.ndarray):
    y = np.asarray(y).ravel()
    X = np.asarray(X, dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    n = X.shape[0]
    p = int(rank)
    df_resid = max(n - p, 1)
    return rss, df_resid, beta, resid, rank

def _partial_F(rss_reduced: float, rss_full: float, df_full: int, q: int):
    q = max(int(q), 1)
    num = (rss_reduced - rss_full) / q
    den = rss_full / max(int(df_full), 1)
    if den <= 1e-15:
        return float("inf") if num > 0 else 0.0
    return float(max(num / den, 0.0))

def stepwise_select_F_with_hierarchy(
    y: np.ndarray,
    Zpoly: np.ndarray,
    powers: np.ndarray,
    feature_names: List[str],
    include_bias: bool = True,
    F_enter: float = 2.0,
    F_remove: float = 2.0,
    start_with_linear: bool = True,
    allowed_kinds: Tuple[str, ...] = ("lin", "inter", "quad"),
    verbose: bool = False,
    max_iter: int = 200,
):
    y = np.asarray(y).ravel()
    Zpoly = np.asarray(Zpoly, dtype=float)

    intercept_col, terms, requires = _build_terms_from_poly(powers, feature_names, include_bias)

    allowed = set(allowed_kinds)
    terms = {k: t for k, t in terms.items() if t.kind in allowed}
    requires = {k: v for k, v in requires.items() if k in terms}
    for k in list(requires.keys()):
        requires[k] = {req for req in requires[k] if req in terms}

    selected: Set[Tuple[str, tuple]] = set()
    if start_with_linear:
        for k, t in terms.items():
            if t.kind == "lin":
                selected.add(k)
    selected = _closure(selected, requires)

    def design_from_keys(keys: Set[Tuple[str, tuple]]):
        cols = []
        if include_bias and intercept_col is not None:
            cols.append(intercept_col)
        for k in sorted(keys, key=lambda x: (x[0], x[1])):
            cols.append(terms[k].col)
        X = Zpoly[:, cols]
        return X, cols

    X_cur, cols_cur = design_from_keys(selected)
    rss_cur, df_cur, _, _, rank_cur = _ols_rss_df(y, X_cur)

    it = 0
    changed = True
    while changed and it < max_iter:
        it += 1
        changed = False

        # Forward
        best_k = None
        best_F = 0.0
        remaining = [k for k in terms.keys() if k not in selected]
        for k in remaining:
            trial = _closure(set(selected) | {k}, requires)
            X_full, cols_full = design_from_keys(trial)
            rss_full, df_full, _, _, _ = _ols_rss_df(y, X_full)
            new_cols = len(set(cols_full) - set(cols_cur))
            Fk = _partial_F(rss_reduced=rss_cur, rss_full=rss_full, df_full=df_full, q=new_cols)
            if Fk > best_F:
                best_F = Fk
                best_k = k

        if best_k is not None and best_F >= F_enter:
            selected = _closure(set(selected) | {best_k}, requires)
            X_cur, cols_cur = design_from_keys(selected)
            rss_cur, df_cur, _, _, rank_cur = _ols_rss_df(y, X_cur)
            changed = True
            if verbose:
                print(f"[ADD] {terms[best_k].name:>12s}  F={best_F:.4f}")

        # Backward
        mand = _mandatory_terms(selected, requires)
        worst_k = None
        worst_F = float("inf")

        for k in list(selected):
            if k in mand:
                continue
            trial = _closure(set(selected) - {k}, requires)
            X_red, cols_red = design_from_keys(trial)
            rss_red, df_red, _, _, _ = _ols_rss_df(y, X_red)
            removed_cols = len(set(cols_cur) - set(cols_red))
            Fk = _partial_F(rss_reduced=rss_red, rss_full=rss_cur, df_full=df_cur, q=removed_cols)
            if Fk < worst_F:
                worst_F = Fk
                worst_k = k

        if worst_k is not None and worst_F < F_remove:
            selected = _closure(set(selected) - {worst_k}, requires)
            X_cur, cols_cur = design_from_keys(selected)
            rss_cur, df_cur, _, _, rank_cur = _ols_rss_df(y, X_cur)
            changed = True
            if verbose:
                print(f"[DROP]{terms[worst_k].name:>12s}  F={worst_F:.4f}")

    debug = {"rss": rss_cur, "df_resid": df_cur, "rank": rank_cur, "n_cols": len(cols_cur), "allowed_kinds": allowed_kinds}
    return cols_cur, selected, debug


# ============================================================
# Core model: RSM + GP(residual)
# ============================================================

def _safe_pinv(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.linalg.pinv(A + eps * np.eye(A.shape[0]))

def _extract_white_noise_variance(gp: GaussianProcessRegressor) -> float:
    def walk(k) -> float:
        if isinstance(k, WhiteKernel):
            return float(k.noise_level)
        if hasattr(k, "k1") and hasattr(k, "k2"):
            return walk(k.k1) + walk(k.k2)
        return 0.0
    return walk(gp.kernel_)

def _recover_latent_std_auto(
    std_returned: np.ndarray,
    noise_std: float,
    tol_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sklearn返却stdが
      - latentのみ：std_returned ≈ latent_std
      - ノイズ込み：std_returned^2 ≈ latent_var + noise_var
    のどちらでも「安全側」に latent を復元する。

    ルール（点ごと）：
    - もし std_returned^2 が noise_var より「十分大きい」なら：
         latent_var = max(std_returned^2 - noise_var, 0)
      （＝ノイズ込みとみなして差し引く）
    - そうでないなら：
         latent_var = std_returned^2
      （＝latentのみとみなして差し引かない）

    戻り値：
      latent_std, did_subtract(bool array)
    """
    std_returned = np.asarray(std_returned, dtype=float)
    noise_var = float(noise_std * noise_std)

    var_ret = np.maximum(std_returned**2, 0.0)

    # 「十分大きい」判定：var_ret > noise_var*(1+tol)
    thresh = noise_var * (1.0 + tol_ratio)
    do_sub = var_ret > thresh

    latent_var = np.where(do_sub, np.maximum(var_ret - noise_var, 0.0), var_ret)
    latent_std = np.sqrt(latent_var)

    return latent_std, do_sub


class RSMPlusGP_Production:
    def __init__(
        self,
        include_bias: bool = True,
        gp_kernel=None,
        gp_alpha: float = 0.0,
        gp_n_restarts_optimizer: int = 3,
        gp_optimizer: Optional[str] = "fmin_l_bfgs_b",
        random_state: int = 0,
        # stepwise
        F_enter: float = 2.0,
        F_remove: float = 2.0,
        start_with_linear: bool = True,
        stepwise_verbose: bool = False,
        # model type
        model_type: str = "quadratic",  # "interaction" | "quadratic"
    ):
        self.include_bias = include_bias
        self.gp_alpha = gp_alpha
        self.gp_n_restarts_optimizer = gp_n_restarts_optimizer
        self.gp_optimizer = gp_optimizer
        self.random_state = random_state

        self.F_enter = F_enter
        self.F_remove = F_remove
        self.start_with_linear = start_with_linear
        self.stepwise_verbose = stepwise_verbose

        if model_type not in ("interaction", "quadratic"):
            raise ValueError("model_type must be 'interaction' or 'quadratic'")
        self.model_type = model_type

        if gp_kernel is None:
            gp_kernel = (
                C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                + WhiteKernel(1e-3, (1e-6, 1e0))
            )
        self.gp_kernel = gp_kernel

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=include_bias)

        self.lin = LinearRegression()
        self.gp: Optional[GaussianProcessRegressor] = None

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

        self._rsm_selected_cols: Optional[List[int]] = None
        self._rsm_selected_keys: Optional[Set[Tuple[str, tuple]]] = None
        self._rsm_stepwise_debug: Optional[Dict[str, Any]] = None

        self._Phi_train_: Optional[np.ndarray] = None
        self._rsm_s2_: Optional[float] = None
        self._rsm_XtX_inv_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be 2D array (n_samples, n_features)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have same number of rows")

        self._X_train = X.copy()
        self._y_train = y.ravel().copy()

        d = X.shape[1]
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(d)]
        if len(feature_names) != d:
            raise ValueError("feature_names length must match X columns")
        self._feature_names = feature_names

        Z = self.x_scaler.fit_transform(X)
        t = self.y_scaler.fit_transform(y).ravel()

        Zpoly_all = self.poly.fit_transform(Z)

        allowed_kinds = ("lin", "inter") if self.model_type == "interaction" else ("lin", "inter", "quad")

        selected_cols, selected_keys, dbg = stepwise_select_F_with_hierarchy(
            y=t,
            Zpoly=Zpoly_all,
            powers=self.poly.powers_,
            feature_names=self._feature_names,
            include_bias=self.include_bias,
            F_enter=self.F_enter,
            F_remove=self.F_remove,
            start_with_linear=self.start_with_linear,
            allowed_kinds=allowed_kinds,
            verbose=self.stepwise_verbose,
        )
        self._rsm_selected_cols = selected_cols
        self._rsm_selected_keys = selected_keys
        self._rsm_stepwise_debug = dbg

        Zpoly = Zpoly_all[:, selected_cols]
        self.lin.fit(Zpoly, t)
        t_hat = self.lin.predict(Zpoly)
        r = t - t_hat

        # RSM param uncertainty cache
        self._Phi_train_ = Zpoly.copy()
        rss = float(r @ r)
        n = self._Phi_train_.shape[0]
        rank = int(np.linalg.matrix_rank(self._Phi_train_))
        df = max(n - rank, 1)
        self._rsm_s2_ = rss / df
        self._rsm_XtX_inv_ = _safe_pinv(self._Phi_train_.T @ self._Phi_train_)

        kernel_instance = sk_clone(self.gp_kernel)
        self.gp = GaussianProcessRegressor(
            kernel=kernel_instance,
            alpha=self.gp_alpha,
            normalize_y=False,
            optimizer=self.gp_optimizer,
            n_restarts_optimizer=self.gp_n_restarts_optimizer,
            random_state=self.random_state,
        )
        self.gp.fit(Z, r)
        return self

    def _check_fitted(self):
        if self._X_train is None or self._rsm_selected_cols is None or self.gp is None:
            raise RuntimeError("Call fit() first.")

    def predict(self, X: np.ndarray, return_std: bool = True):
        """
        注意：return_std は sklearn 実装により WhiteKernelノイズ込み/なしが揺れることがある。
        区間やEIでは predict_interval_two_types を通じた std を推奨。
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        Z = self.x_scaler.transform(X)
        Zpoly_all = self.poly.transform(Z)
        Zpoly = Zpoly_all[:, self._rsm_selected_cols]

        t_rsm = self.lin.predict(Zpoly)

        if return_std:
            r_mean, r_std = self.gp.predict(Z, return_std=True)
            t_mean = t_rsm + r_mean
            y_mean = self.y_scaler.inverse_transform(t_mean.reshape(-1, 1)).ravel()
            y_std = r_std * float(self.y_scaler.scale_[0])
            return y_mean, y_std
        else:
            r_mean = self.gp.predict(Z, return_std=False)
            t_mean = t_rsm + r_mean
            return self.y_scaler.inverse_transform(t_mean.reshape(-1, 1)).ravel()

    # ---- Coefficients ----
    def rsm_coefficients_standardized(self) -> Dict[str, Any]:
        self._check_fitted()
        b0 = float(self.lin.intercept_)
        coef = np.asarray(self.lin.coef_).ravel()

        powers_all = self.poly.powers_
        sel_cols = self._rsm_selected_cols
        sel_powers = powers_all[sel_cols]

        d = powers_all.shape[1]
        linear = {i: 0.0 for i in range(d)}
        quad = {i: 0.0 for i in range(d)}
        inter: Dict[Tuple[int, int], float] = {}

        for c, p in zip(coef, sel_powers):
            if np.all(p == 0):
                continue
            nz = np.where(p != 0)[0]
            if len(nz) == 1:
                i = int(nz[0])
                if p[i] == 1:
                    linear[i] += float(c)
                elif p[i] == 2:
                    quad[i] += float(c)
            elif len(nz) == 2:
                i, j = int(nz[0]), int(nz[1])
                if p[i] == 1 and p[j] == 1:
                    key = (min(i, j), max(i, j))
                    inter[key] = inter.get(key, 0.0) + float(c)

        return {"Intercept": b0, "linear": linear, "quad": quad, "inter": inter}

    def rsm_coefficients_original_mixed(self) -> Dict[str, Any]:
        self._check_fitted()
        b = self.rsm_coefficients_standardized()
        mu_x = self.x_scaler.mean_.copy()
        sig_x = self.x_scaler.scale_.copy()
        mu_y = float(self.y_scaler.mean_[0])
        sig_y = float(self.y_scaler.scale_[0])

        d = len(mu_x)
        c0 = mu_y + sig_y * b["Intercept"]
        c_i = {i: sig_y * b["linear"][i] / sig_x[i] for i in range(d)}
        c_ii = {i: sig_y * b["quad"][i] / (sig_x[i] ** 2) for i in range(d)}
        c_ij = {(i, j): sig_y * v / (sig_x[i] * sig_x[j]) for (i, j), v in b["inter"].items()}

        # linear not centered
        a_i = c_i
        a0 = c0 - float(np.sum([c_i[i] * mu_x[i] for i in range(d)]))

        meta = {"mu_x": mu_x, "sigma_x": sig_x, "mu_y": mu_y, "sigma_y": sig_y}
        return {"a0": float(a0), "a_linear": a_i, "c_quad": c_ii, "c_inter": c_ij, "meta": meta}

    def mixed_equation_string(
        self,
        y_name: str = "y",
        decimals: int = 4,
        show_mu_values: bool = True,
        order: Tuple[str, ...] = ("linear", "inter", "quad"),
    ) -> str:
        self._check_fitted()
        info = self.rsm_coefficients_original_mixed()
        a0 = info["a0"]
        a_lin = info["a_linear"]
        c_quad = info["c_quad"]
        c_inter = info["c_inter"]
        mu_x = info["meta"]["mu_x"]
        names = self._feature_names or [f"x{i+1}" for i in range(len(mu_x))]

        def fmt(v: float) -> str:
            return f"{v:.{decimals}f}"

        def add_term(expr: str, coeff: float) -> str:
            if abs(coeff) < 10 ** (-(decimals + 1)):
                return ""
            sign = " + " if coeff >= 0 else " - "
            return f"{sign}{fmt(abs(coeff))}·{expr}"

        eq = f"{y_name} = {fmt(a0)}"

        if "linear" in order:
            for i, nm in enumerate(names):
                eq += add_term(nm, a_lin.get(i, 0.0))

        if "inter" in order:
            for (i, j) in sorted(c_inter.keys()):
                mu_i = fmt(mu_x[i]) if show_mu_values else f"μ_{names[i]}"
                mu_j = fmt(mu_x[j]) if show_mu_values else f"μ_{names[j]}"
                eq += add_term(f"({names[i]} - {mu_i})·({names[j]} - {mu_j})", c_inter[(i, j)])

        if "quad" in order:
            for i, nm in enumerate(names):
                mu_i = fmt(mu_x[i]) if show_mu_values else f"μ_{nm}"
                eq += add_term(f"({nm} - {mu_i})^2", c_quad.get(i, 0.0))

        return eq

    def standardized_importance_ranking(self, top_k: int = 20) -> List[Tuple[str, float]]:
        self._check_fitted()
        b = self.rsm_coefficients_standardized()
        names = self._feature_names or []
        rows = [("Intercept", b["Intercept"])]
        for i, v in b["linear"].items():
            rows.append((f"{names[i]}", v))
        for (i, j), v in b["inter"].items():
            rows.append((f"{names[i]}*{names[j]}", v))
        for i, v in b["quad"].items():
            rows.append((f"{names[i]}^2", v))
        return ([rows[0]] + sorted(rows[1:], key=lambda t: abs(t[1]), reverse=True))[:max(1, top_k)]

    # ---- RSM param uncertainty ----
    def _rsm_pred_var_standardized(self, Z: np.ndarray) -> np.ndarray:
        if self._Phi_train_ is None or self._rsm_s2_ is None or self._rsm_XtX_inv_ is None:
            raise RuntimeError("RSM uncertainty cache missing. Fit again.")
        Zpoly_all = self.poly.transform(Z)
        Phi = Zpoly_all[:, self._rsm_selected_cols]
        v = np.einsum("ij,jk,ik->i", Phi, self._rsm_XtX_inv_, Phi)
        return np.maximum(self._rsm_s2_ * v, 0.0)

    # ---- Two-type intervals with AUTO diagnosis ----
    def predict_interval_two_types(
        self,
        X: np.ndarray,
        level: float = 0.95,
        include_rsm_param_uncertainty: bool = True,
        latent_recovery_tol_ratio: float = 0.05,  # ★自己診断の閾値
    ) -> Dict[str, Any]:
        """
        Two-type intervals:
          - posterior: latent(GP) + (optional) RSM param uncertainty
          - predictive: posterior + observation noise (WhiteKernel)

        ★自己診断：
        - return_stdが「ノイズ込み」っぽい点だけ差し引き
        - 「latentのみ」っぽい点は差し引かない
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        Z = self.x_scaler.transform(X)

        # mean & returned std (y unit)
        y_mean, gp_std_y_returned = self.predict(X, return_std=True)

        # WhiteKernel noise (std in y unit)
        noise_var_r = _extract_white_noise_variance(self.gp)
        noise_std_y = float(np.sqrt(max(noise_var_r, 0.0)) * self.y_scaler.scale_[0])

        # latent recovery (AUTO per-point)
        gp_latent_std_y, did_subtract = _recover_latent_std_auto(
            std_returned=gp_std_y_returned,
            noise_std=noise_std_y,
            tol_ratio=latent_recovery_tol_ratio,
        )

        # RSM param uncertainty -> y unit
        rsm_param_std_y = 0.0
        if include_rsm_param_uncertainty:
            v_t = self._rsm_pred_var_standardized(Z)
            rsm_param_std_y = float(self.y_scaler.scale_[0]) * np.sqrt(np.maximum(v_t, 0.0))

        posterior_std = np.sqrt(np.maximum(gp_latent_std_y**2 + rsm_param_std_y**2, 0.0))
        predictive_std = np.sqrt(np.maximum(posterior_std**2 + noise_std_y**2, 0.0))

        z = z_value(level)

        # 診断サマリ
        subtract_rate = float(np.mean(did_subtract)) if did_subtract.size else 0.0

        return {
            "level": level,
            "mean": y_mean,
            "posterior": {"std": posterior_std, "lo": y_mean - z * posterior_std, "hi": y_mean + z * posterior_std},
            "predictive": {"std": predictive_std, "lo": y_mean - z * predictive_std, "hi": y_mean + z * predictive_std},
            "components": {
                "gp_std_returned": gp_std_y_returned,     # raw returned
                "gp_latent_std": gp_latent_std_y,         # auto-recovered
                "did_subtract_noise": did_subtract,       # per-point
                "subtract_rate": subtract_rate,           # overall ratio
                "rsm_param_std": rsm_param_std_y if include_rsm_param_uncertainty else None,
                "obs_noise_std": noise_std_y,
                "white_noise_var_resid_std_space": noise_var_r,
                "latent_recovery_tol_ratio": latent_recovery_tol_ratio,
            },
        }

    def diagnose(self, X_test: Optional[np.ndarray] = None, y_test_true: Optional[np.ndarray] = None,
                 display: bool = True) -> Dict[str, Any]:
        """
        モデル診断ツール：RSMとGPの寄与度、性能指標、推奨事項を提供

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features), optional
            テストデータの入力。Noneの場合は訓練データを使用
        y_test_true : array-like, shape (n_samples,), optional
            テストデータの真値。Noneの場合は訓練データを使用
        display : bool, default=True
            診断レポートを表示するかどうか

        Returns
        -------
        info : dict
            診断情報を含む辞書
            - n_train: 訓練データ数
            - n_features: 特徴量数
            - n_selected_terms: 選択された項数
            - n_total_terms: 全候補項数
            - r2: RSM+GPのR²スコア
            - rmse: RMSE
            - mae: MAE
            - rsm_only_r2: RSMのみのR²スコア
            - improvement: GP改善率 (%)
            - rsm_contribution: RSM寄与度 (%)
            - gp_contribution: GP寄与度 (%)
            - length_scale: カーネルのlength_scale
            - constant_value: カーネルの定数値
            - noise_level: ノイズレベル
            - kernel_status: カーネルパラメータの評価
            - recommendations: 推奨事項のリスト
        """
        self._check_fitted()

        # テストデータの設定
        if X_test is None:
            X_test = self._X_train
        if y_test_true is None:
            y_test_true = self._y_train

        X_test = np.asarray(X_test, dtype=float)
        y_test_true = np.asarray(y_test_true, dtype=float).ravel()

        # 基本統計
        info = {}
        info['n_train'] = self._X_train.shape[0]
        info['n_features'] = self._X_train.shape[1]
        info['n_selected_terms'] = len(self._rsm_selected_cols)
        info['n_total_terms'] = len(self.poly.powers_)

        # RSMのみの予測
        Z_test = self.x_scaler.transform(X_test)
        Zpoly_test = self.poly.transform(Z_test)[:, self._rsm_selected_cols]
        t_rsm = self.lin.predict(Zpoly_test)
        y_rsm_only = self.y_scaler.inverse_transform(t_rsm.reshape(-1, 1)).ravel()

        # RSM+GPの予測
        y_full = self.predict(X_test, return_std=False)

        # 寄与度計算
        rsm_var = np.var(y_rsm_only - np.mean(y_test_true))
        full_var = np.var(y_full - np.mean(y_test_true))
        gp_correction_var = np.var(y_full - y_rsm_only)

        total_var = rsm_var + gp_correction_var
        if total_var > 1e-12:
            info['rsm_contribution'] = rsm_var / total_var * 100
            info['gp_contribution'] = gp_correction_var / total_var * 100
        else:
            info['rsm_contribution'] = 100.0
            info['gp_contribution'] = 0.0

        # 精度指標
        try:
            from sklearn.metrics import r2_score
            info['r2'] = r2_score(y_test_true, y_full)
            info['rsm_only_r2'] = r2_score(y_test_true, y_rsm_only)
        except ImportError:
            ss_tot = np.sum((y_test_true - np.mean(y_test_true))**2)
            ss_res = np.sum((y_test_true - y_full)**2)
            info['r2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            ss_res_rsm = np.sum((y_test_true - y_rsm_only)**2)
            info['rsm_only_r2'] = 1 - ss_res_rsm / ss_tot if ss_tot > 0 else 0.0

        info['rmse'] = np.sqrt(np.mean((y_test_true - y_full)**2))
        info['mae'] = np.mean(np.abs(y_test_true - y_full))
        info['improvement'] = (info['r2'] - info['rsm_only_r2']) * 100

        # カーネルパラメータ
        kernel_params = self.gp.kernel_
        try:
            # C * RBF + WhiteKernel の形式を想定
            info['length_scale'] = kernel_params.k1.k2.length_scale
            info['constant_value'] = kernel_params.k1.k1.constant_value
            info['noise_level'] = kernel_params.k2.noise_level
        except AttributeError:
            # カーネル構造が異なる場合
            info['length_scale'] = None
            info['constant_value'] = None
            info['noise_level'] = None

        # カーネルパラメータの評価
        if info['length_scale'] is not None:
            ls = float(np.mean(info['length_scale'])) if hasattr(info['length_scale'], '__iter__') else float(info['length_scale'])
            if ls < 0.05:
                info['kernel_status'] = "短すぎ（過学習の可能性）"
            elif ls > 10.0:
                info['kernel_status'] = "長すぎ（過平滑化の可能性）"
            else:
                info['kernel_status'] = "適切 ✓"
        else:
            info['kernel_status'] = "不明"

        # 高次項検知（GP補正の空間的パターン分析）
        gp_correction = y_full - y_rsm_only
        gp_correction_magnitude = np.abs(gp_correction)
        avg_correction = np.mean(gp_correction_magnitude)
        max_correction = np.max(gp_correction_magnitude)
        correction_std = np.std(gp_correction_magnitude)

        # 補正の相対的な大きさ
        y_range = np.max(y_test_true) - np.min(y_test_true)
        relative_correction = avg_correction / y_range * 100 if y_range > 1e-10 else 0.0

        # 補正の不均一性（局所的な非線形性の指標）
        correction_cv = correction_std / avg_correction if avg_correction > 1e-10 else 0.0

        info['gp_correction_avg'] = avg_correction
        info['gp_correction_max'] = max_correction
        info['gp_correction_relative'] = relative_correction
        info['gp_correction_cv'] = correction_cv

        # 高次項の存在可能性を判定
        higher_order_detected = False
        higher_order_strength = "なし"

        if info['gp_contribution'] > 30 or info['improvement'] > 15:
            higher_order_detected = True
            higher_order_strength = "強い"
        elif info['gp_contribution'] > 15 or info['improvement'] > 8:
            higher_order_detected = True
            higher_order_strength = "中程度"
        elif info['gp_contribution'] > 5 or info['improvement'] > 3:
            higher_order_detected = True
            higher_order_strength = "弱い"

        info['higher_order_detected'] = higher_order_detected
        info['higher_order_strength'] = higher_order_strength

        # 推奨事項の生成
        recommendations = []

        if info['r2'] > 0.99:
            recommendations.append("✓ モデルは非常に良好に機能しています")
        elif info['r2'] > 0.95:
            recommendations.append("✓ モデルは良好に機能しています")
            if info['n_train'] < 100:
                recommendations.append("  さらなる精度向上にはデータ追加推奨（n→100-150）")
        elif info['r2'] > 0.90:
            recommendations.append("⚠ モデルの精度は実用レベルですが改善余地があります")
            recommendations.append(f"  データ数を増やすことを推奨（現在n={info['n_train']}）")
        else:
            recommendations.append("⚠ モデルの精度が低いです")
            recommendations.append("  データ数を大幅に増やしてください")

        # 高次項検知の推奨事項
        if higher_order_detected:
            if higher_order_strength == "強い":
                recommendations.append(f"⚠ 高次項（3次以上）の存在が強く示唆されます")
                recommendations.append(f"  GP寄与度: {info['gp_contribution']:.1f}%（RSMで捉えられない非線形性）")
                recommendations.append(f"  GP改善率: {info['improvement']:.1f}%（高次項の重要性が高い）")
                if info['n_train'] < 100:
                    recommendations.append(f"  → データを増やすとGPが高次項をより正確に捕捉（n≥100推奨）")
                else:
                    recommendations.append(f"  → 現在のデータ量（n={info['n_train']}）で高次項を良好に捕捉")
                if relative_correction > 10:
                    recommendations.append(f"  → GP補正の大きさ: 応答範囲の{relative_correction:.1f}%")
            elif higher_order_strength == "中程度":
                recommendations.append(f"○ 高次項（3次以上）の存在が示唆されます")
                recommendations.append(f"  GP寄与度: {info['gp_contribution']:.1f}%、改善率: {info['improvement']:.1f}%")
                if info['n_train'] < 100:
                    recommendations.append(f"  → データを増やすとより正確に高次項を捕捉（n≥100推奨）")
            else:  # 弱い
                recommendations.append(f"○ 軽微な高次項の可能性があります")
                recommendations.append(f"  GP寄与度: {info['gp_contribution']:.1f}%（小さい）")
                if info['n_train'] < 50:
                    recommendations.append(f"  → データを増やすと高次項検知の信頼性向上")
        else:
            if info['improvement'] < 3 and info['n_train'] < 100:
                recommendations.append(f"  GPの改善が小さいです（{info['improvement']:.1f}%）")
                recommendations.append("  → データを増やすとGPが高次項を捕捉可能（n≥100推奨）")

        if info['kernel_status'] != "適切 ✓" and info['kernel_status'] != "不明":
            recommendations.append(f"⚠ カーネルパラメータ: {info['kernel_status']}")

        info['recommendations'] = recommendations

        # レポート表示
        if display:
            print("=" * 80)
            print("モデル診断レポート")
            print("=" * 80)

            print("\n【基本統計】")
            print(f"  訓練データ: n={info['n_train']}, d={info['n_features']}")
            print(f"  選択項数: {info['n_selected_terms']}/{info['n_total_terms']}")

            print("\n【予測性能】")
            print(f"  R²:   {info['r2']:.4f}")
            print(f"  RMSE: {info['rmse']:.4f}")
            print(f"  MAE:  {info['mae']:.4f}")

            print("\n【RSM vs GP 寄与度】")
            print(f"  RSM寄与: {info['rsm_contribution']:.1f}% (主に線形・二次構造)")
            print(f"  GP寄与:  {info['gp_contribution']:.1f}% (残差・高次項補正)")
            print(f"  RSMのみのR²: {info['rsm_only_r2']:.4f}")
            print(f"  GP改善:      {info['improvement']:.2f}%")

            if info['rsm_contribution'] > 80:
                print("  → RSM主導型（二次多項式で十分説明）✓")
            elif info['gp_contribution'] > 50:
                print("  → GP主導型（高次・非線形性が支配的）")
            else:
                print("  → バランス良好 ✓")

            print("\n【カーネルパラメータ】")
            if info['length_scale'] is not None:
                ls = info['length_scale']
                if hasattr(ls, '__iter__'):
                    print(f"  length_scale:   {ls}")
                else:
                    print(f"  length_scale:   {ls:.4f}")
                print(f"  constant_value: {info['constant_value']:.4f}")
                print(f"  noise_level:    {info['noise_level']:.4f}")
                print(f"  評価: {info['kernel_status']}")
            else:
                print("  カーネルパラメータ情報なし")

            print("\n【高次項検知】")
            if info['higher_order_detected']:
                strength_emoji = {"強い": "⚠", "中程度": "○", "弱い": "△"}
                emoji = strength_emoji.get(info['higher_order_strength'], "○")
                print(f"  {emoji} 高次項（3次以上）の存在: {info['higher_order_strength']}")
                print(f"  GP補正（平均）: {info['gp_correction_avg']:.4f}")
                print(f"  GP補正（最大）: {info['gp_correction_max']:.4f}")
                print(f"  補正の相対値: 応答範囲の{info['gp_correction_relative']:.1f}%")
                if info['higher_order_strength'] == "強い":
                    print("  → RSMでは捉えられない重要な非線形性が存在します")
                    print("  → GPがこの高次項を効果的に補正しています")
                elif info['higher_order_strength'] == "中程度":
                    print("  → 軽微〜中程度の高次項が存在する可能性があります")
                else:
                    print("  → 軽微な非線形性のみ、ほぼRSMで説明可能")
            else:
                print("  ✓ 高次項の存在は検出されませんでした")
                print("  → RSMの二次多項式で十分に説明可能")
                if info['n_train'] < 100:
                    print("  → データを増やすと潜在的な高次項を検知可能（n≥100推奨）")

            print("\n【推奨事項】")
            for rec in recommendations:
                print(f"  {rec}")

            print("\n" + "=" * 80)

        return info


# ============================================================
# Validation: KFold CV + Plotly (optional)
# ============================================================

def _fit_fold(base_model: RSMPlusGP_Production, X: np.ndarray, y_true: np.ndarray,
              tr: np.ndarray, te: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """単一のfoldを処理（並列化用ヘルパー関数）"""
    m = RSMPlusGP_Production(
        include_bias=base_model.include_bias,
        gp_kernel=base_model.gp_kernel,
        gp_alpha=base_model.gp_alpha,
        gp_n_restarts_optimizer=base_model.gp_n_restarts_optimizer,
        gp_optimizer=base_model.gp_optimizer,
        random_state=base_model.random_state,
        F_enter=base_model.F_enter,
        F_remove=base_model.F_remove,
        start_with_linear=base_model.start_with_linear,
        stepwise_verbose=False,
        model_type=base_model.model_type,
    )
    m.fit(X[tr], y_true[tr], feature_names=feature_names)
    mu, sd = m.predict(X[te], return_std=True)
    return te, mu, sd

def perform_kfold_cv(
    base_model: RSMPlusGP_Production,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
    n_jobs: int = -1,  # -1 = 全CPUコア使用、1 = 逐次処理
    auto_switch: bool = True,  # データサイズに応じて自動切替
    auto_threshold: int = 100,  # 自動切替の閾値
) -> Dict[str, Any]:
    """K-Fold CV（並列化対応版 + 自動切替）

    Args:
        n_jobs: 並列ジョブ数。-1で全コア使用、1で逐次処理。joblibがない場合は自動的に逐次処理。
        auto_switch: Trueの場合、データサイズに応じて並列化を自動切替。
                     n < auto_threshold で逐次処理、n >= auto_threshold で並列処理。
        auto_threshold: 自動切替の閾値（デフォルト: 100）
    """
    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y, dtype=float).ravel()

    d = X.shape[1]
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(d)]

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(kf.split(X))

    y_pred = np.zeros_like(y_true, dtype=float)
    y_std = np.zeros_like(y_true, dtype=float)

    # 自動切替ロジック
    n_samples = X.shape[0]
    if auto_switch and n_samples < auto_threshold:
        actual_n_jobs = 1  # 小規模データは逐次処理（オーバーヘッド回避）
    else:
        actual_n_jobs = n_jobs

    # 並列化（joblibがあれば）
    if _HAS_JOBLIB and actual_n_jobs != 1:
        results = Parallel(n_jobs=actual_n_jobs)(
            delayed(_fit_fold)(base_model, X, y_true, tr, te, feature_names)
            for tr, te in splits
        )
        for te, mu, sd in results:
            y_pred[te] = mu
            y_std[te] = sd
    else:
        # 逐次処理（joblibなしまたはn_jobs=1）
        for tr, te in splits:
            te_idx, mu, sd = _fit_fold(base_model, X, y_true, tr, te, feature_names)
            y_pred[te_idx] = mu
            y_std[te_idx] = sd

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    q2 = float(r2_score(y_true, y_pred))

    z = z_value(0.95)
    cover95 = float(np.mean((y_true >= y_pred - z * y_std) & (y_true <= y_pred + z * y_std)))

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_std": y_std,
        "metrics": {"Q2": q2, "RMSE": rmse, "MAE": mae, "Coverage95_returned_std": cover95},
    }

def plot_actual_vs_predicted_plotly(results: Dict[str, Any], title: str = "CV: Actual vs Predicted"):
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly not installed. pip install plotly")
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_std = results["y_std"]
    m = results["metrics"]

    z = z_value(0.95)
    scatter = go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Prediction",
        error_y=dict(type="data", array=z * y_std, visible=True),
        hovertemplate="Actual=%{x:.4f}<br>Pred=%{y:.4f}<extra></extra>",
    )

    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    span = mx - mn
    pad = 0.05 * span if span > 0 else 1.0
    lims = [mn - pad, mx + pad]

    line = go.Scatter(x=lims, y=lims, mode="lines", name="y=x", line=dict(dash="dash"))
    subtitle = f"Q²={m.get('Q2', np.nan):.3f}, RMSE={m.get('RMSE', np.nan):.3f}, Coverage95(returned)={m.get('Coverage95_returned_std', np.nan):.3f}"
    fig = go.Figure([line, scatter])
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        xaxis=dict(title="Actual", range=lims),
        yaxis=dict(title="Predicted", range=lims, scaleanchor="x", scaleratio=1),
        width=760,
        height=760,
        template="plotly_white",
    )
    return fig


# ============================================================
# Candidate generation + constraints + EI
# ============================================================

def build_candidates(
    bounds: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
    discrete_levels: Optional[Dict[str, List[float]] | List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    lows, highs = bounds[:, 0], bounds[:, 1]

    if discrete_levels is None:
        return rng.uniform(lows, highs, size=(n_candidates, d))

    if isinstance(discrete_levels, dict):
        if feature_names is None:
            raise ValueError("feature_names is required when discrete_levels is dict.")
        levels = [np.asarray(discrete_levels[n], dtype=float) for n in feature_names]
    else:
        levels = [np.asarray(v, dtype=float) for v in discrete_levels]
        if len(levels) != d:
            raise ValueError("discrete_levels length must match #features")

    X = np.column_stack([rng.choice(levels[i], size=n_candidates, replace=True) for i in range(d)])

    for i in range(d):
        X = X[(X[:, i] >= lows[i]) & (X[:, i] <= highs[i])]
    return X

def _min_dist_to_train(Xcand: np.ndarray, Xtrain: np.ndarray) -> np.ndarray:
    """各候補点から訓練データへの最小距離を計算（最適化版）"""
    if _HAS_SCIPY:
        # scipy.spatial.distance.cdist は C実装で2-5倍高速
        d = _cdist(Xcand, Xtrain, metric='euclidean')
        return np.min(d, axis=1)
    # フォールバック：純粋NumPy版
    d = np.linalg.norm(Xcand[:, None, :] - Xtrain[None, :, :], axis=2)
    return np.min(d, axis=1)

def _dopt_delta_logdet(model: RSMPlusGP_Production, Xcand: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    Xtr = model._X_train
    if Xtr is None:
        raise RuntimeError("fit first")
    Ztr = model.x_scaler.transform(Xtr)
    Phi_tr = model.poly.transform(Ztr)[:, model._rsm_selected_cols]
    M = Phi_tr.T @ Phi_tr + eps * np.eye(Phi_tr.shape[1])
    Minv = np.linalg.inv(M)

    Zc = model.x_scaler.transform(Xcand)
    Phi_c = model.poly.transform(Zc)[:, model._rsm_selected_cols]
    q = np.einsum("ij,jk,ik->i", Phi_c, Minv, Phi_c)
    return np.log1p(np.maximum(q, 0.0))

def propose_next_EI_constrained(
    model: RSMPlusGP_Production,
    bounds: np.ndarray,
    candidate_set: Optional[np.ndarray] = None,
    n_candidates: int = 20000,
    discrete_levels: Optional[Dict[str, List[float]] | List[List[float]]] = None,
    constraint_fn: Optional[Callable[[np.ndarray], bool]] = None,
    cost_fn: Optional[Callable[[np.ndarray], float]] = None,
    cost_budget: Optional[float] = None,
    objective: str = "min",
    xi: float = 0.01,
    score_mode: str = "EI",              # "EI" or "EI_per_cost"
    std_mode: str = "predictive",        # "gp_latent" | "posterior" | "predictive"
    best_mode: str = "denoised",         # "observed" | "denoised"
    include_rsm_param_uncertainty: bool = True,
    diversity_lambda: float = 0.0,
    dopt_lambda: float = 0.0,
    top_k: int = 5,
    random_state: int = 0,
) -> List[Dict[str, Any]]:
    model._check_fitted()
    rng = np.random.default_rng(random_state)
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]

    if candidate_set is not None:
        Xcand = np.asarray(candidate_set, dtype=float)
        if Xcand.ndim != 2 or Xcand.shape[1] != d:
            raise ValueError("candidate_set must be (n,d)")
        for i in range(d):
            Xcand = Xcand[(Xcand[:, i] >= bounds[i, 0]) & (Xcand[:, i] <= bounds[i, 1])]
    else:
        Xcand = build_candidates(bounds, n_candidates, rng, discrete_levels, model._feature_names)

    if Xcand.shape[0] == 0:
        raise RuntimeError("No candidates after bounds/levels filter.")

    mask = np.ones(Xcand.shape[0], dtype=bool)
    if constraint_fn is not None:
        # ベクトル化を試みる（高速）、失敗したらループにフォールバック
        try:
            mask &= np.asarray(constraint_fn(Xcand), dtype=bool)
        except (TypeError, ValueError):
            # 制約関数がベクトル化非対応の場合はループ
            mask &= np.array([bool(constraint_fn(x)) for x in Xcand], dtype=bool)

    costs = None
    if cost_fn is not None:
        # コスト関数もベクトル化を試みる
        try:
            costs = np.asarray(cost_fn(Xcand), dtype=float)
        except (TypeError, ValueError):
            # ベクトル化非対応の場合はループ
            costs = np.array([float(cost_fn(x)) for x in Xcand], dtype=float)
        if cost_budget is not None:
            mask &= (costs <= cost_budget)

    Xf = Xcand[mask]
    if Xf.shape[0] == 0:
        raise RuntimeError("No feasible candidates under constraints/budget.")
    cf = costs[mask] if costs is not None else None

    # 最適化: predict を2回呼ばず、predict_interval_two_types で mu と std を一度に取得
    if std_mode == "gp_latent":
        info = model.predict_interval_two_types(Xf, level=0.95, include_rsm_param_uncertainty=False)
        std = info["components"]["gp_latent_std"]
    elif std_mode in ("posterior", "predictive"):
        info = model.predict_interval_two_types(
            Xf, level=0.95, include_rsm_param_uncertainty=include_rsm_param_uncertainty
        )
        std = info["posterior"]["std"] if std_mode == "posterior" else info["predictive"]["std"]
    else:
        raise ValueError("std_mode must be 'gp_latent', 'posterior', or 'predictive'")

    # meanは上記のinfoから取得（重複呼び出しを削減）
    mu = info["mean"]

    if best_mode == "observed":
        best = float(np.min(model._y_train)) if objective == "min" else float(np.max(model._y_train))
    elif best_mode == "denoised":
        den = model.predict(model._X_train, return_std=False)
        best = float(np.min(den)) if objective == "min" else float(np.max(den))
    else:
        raise ValueError("best_mode must be 'observed' or 'denoised'")

    ei = expected_improvement(mu, std, best=best, objective=objective, xi=xi)
    score = ei.copy()

    min_dist = None
    if diversity_lambda > 0:
        min_dist = _min_dist_to_train(Xf, model._X_train)
        score = score + diversity_lambda * min_dist

    dopt_gain = None
    if dopt_lambda > 0:
        dopt_gain = _dopt_delta_logdet(model, Xf)
        score = score + dopt_lambda * dopt_gain

    if score_mode == "EI_per_cost":
        if cf is None:
            raise ValueError("EI_per_cost requires cost_fn")
        score = score / np.maximum(cf, 1e-12)
    elif score_mode != "EI":
        raise ValueError("score_mode must be 'EI' or 'EI_per_cost'")

    k = int(min(max(top_k, 1), Xf.shape[0]))
    idx = np.argsort(-score)[:k]

    out = []
    for ii in idx:
        item = {
            "x": Xf[ii],
            "score": float(score[ii]),
            "EI": float(ei[ii]),
            "mu": float(mu[ii]),
            "std_for_EI": float(std[ii]),
            "std_mode": std_mode,
            "best_mode": best_mode,
            "objective": objective,
            "xi": xi,
            "score_mode": score_mode,
        }
        if cf is not None:
            item["cost"] = float(cf[ii])
        if min_dist is not None:
            item["min_dist_to_existing"] = float(min_dist[ii])
        if dopt_gain is not None:
            item["dopt_delta_logdet"] = float(dopt_gain[ii])
        out.append(item)

    return out


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 60
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (
        3.0
        + 2.5 * X[:, 0]
        - 1.8 * X[:, 1]
        + 4.0 * X[:, 0] ** 2
        + 3.5 * X[:, 0] * X[:, 1]
        - 2.5 * X[:, 1] ** 2
        + 0.8 * np.sin(5 * X[:, 0]) * np.cos(3 * X[:, 1])
        + rng.normal(0, 0.18, size=n)
    )
    feat_names = ["Temp", "Press"]

    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))

    model_type = "quadratic"  # "interaction" or "quadratic"

    model = RSMPlusGP_Production(
        gp_kernel=kernel,
        gp_n_restarts_optimizer=5,
        random_state=42,
        F_enter=2.0,
        F_remove=2.0,
        start_with_linear=True,
        stepwise_verbose=False,
        model_type=model_type,
    )

    print("\n--- Fit all ---")
    model.fit(X, y, feature_names=feat_names)
    print("Stepwise debug:", model._rsm_stepwise_debug)

    print("\n=== Mixed equation (original unit) ===")
    print(model.mixed_equation_string(y_name="Y", decimals=3))

    Xtest = np.array([[0.2, -0.3], [0.8, 0.7]], dtype=float)
    info = model.predict_interval_two_types(Xtest, level=0.95, include_rsm_param_uncertainty=True)

    print("\n=== Two-type intervals (95%) ===")
    print("subtract_rate (auto diagnosis):", info["components"]["subtract_rate"])
    for i in range(Xtest.shape[0]):
        print(f"x={Xtest[i]}  mean={info['mean'][i]:.4f}  "
              f"post=[{info['posterior']['lo'][i]:.4f},{info['posterior']['hi'][i]:.4f}]  "
              f"pred=[{info['predictive']['lo'][i]:.4f},{info['predictive']['hi'][i]:.4f}]")

    # Next experiment proposals
    levels_T = [-1.0, -0.5, 0.0, 0.5, 1.0]
    levels_P = [-1.0, 0.0, 1.0]
    candidate_set = np.array([(t, p) for t in levels_T for p in levels_P], dtype=float)

    def constraint_fn(x):
        t, p = x
        return not (t >= 0.5 and p >= 1.0)

    def cost_fn(x):
        t, p = x
        return 1.0 + 2.0 * max(0.0, t)

    props = propose_next_EI_constrained(
        model=model,
        bounds=np.array([[-1, 1], [-1, 1]], dtype=float),
        candidate_set=candidate_set,
        constraint_fn=constraint_fn,
        cost_fn=cost_fn,
        cost_budget=3.0,
        objective="min",
        xi=0.01,
        score_mode="EI_per_cost",
        std_mode="predictive",
        best_mode="denoised",
        include_rsm_param_uncertainty=True,
        diversity_lambda=0.05,
        dopt_lambda=0.10,
        top_k=5,
        random_state=0,
    )

    print("\n=== Next experiments Top-5 ===")
    for i, p in enumerate(props, 1):
        print(f"#{i} x={p['x']} score={p['score']:.6f} EI={p['EI']:.6f} "
              f"mu={p['mu']:.4f} std={p['std_for_EI']:.4f} cost={p.get('cost', float('nan')):.3f}")
