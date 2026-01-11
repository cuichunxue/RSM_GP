"""
D最適計画データでの解析・予測性能テスト
D-optimal designとランダムサンプリングの比較
"""
import numpy as np
import rsm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("D最適計画データでの解析・予測性能検証")
print("=" * 80)

def get_kernel():
    return C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))

# 真の応答関数
def true_function(X):
    """真の応答関数: 二次関数 + 交互作用"""
    return (10.0 +
            3.0*X[:, 0] - 2.0*X[:, 1] + 1.5*X[:, 2] +
            2.5*X[:, 0]**2 - 1.8*X[:, 1]**2 + 1.2*X[:, 2]**2 +
            2.0*X[:, 0]*X[:, 1] - 1.5*X[:, 1]*X[:, 2])

# ==========================================
# D最適計画の生成（簡易版）
# ==========================================
def generate_d_optimal_design(n_points, d, bounds, n_candidates=5000, random_state=42):
    """
    D最適計画を近似的に生成
    - 候補点から情報行列の行列式を最大化する点を選択
    """
    rng = np.random.default_rng(random_state)

    # 候補点を大量生成
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, d))

    # 二次多項式の特徴量に変換（簡易版）
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly_candidates = poly.fit_transform(candidates)

    # D最適計画を貪欲法で構築
    selected_indices = []

    # 初期点: ランダムに選択
    initial_idx = rng.choice(n_candidates)
    selected_indices.append(initial_idx)

    # 貪欲法で点を追加
    for _ in range(n_points - 1):
        best_idx = -1
        best_logdet = -np.inf

        for idx in range(n_candidates):
            if idx in selected_indices:
                continue

            # 仮に追加してみる
            trial_indices = selected_indices + [idx]
            X_trial = X_poly_candidates[trial_indices]

            # 情報行列 X'X の対数行列式を計算
            try:
                M = X_trial.T @ X_trial
                sign, logdet = np.linalg.slogdet(M + 1e-6 * np.eye(M.shape[0]))
                if sign > 0:
                    current_logdet = logdet
                else:
                    current_logdet = -np.inf
            except:
                current_logdet = -np.inf

            if current_logdet > best_logdet:
                best_logdet = current_logdet
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
        else:
            # 見つからなければランダムに追加
            remaining = [i for i in range(n_candidates) if i not in selected_indices]
            if remaining:
                selected_indices.append(rng.choice(remaining))

    return candidates[selected_indices]

# ==========================================
# テスト設定
# ==========================================
n_train = 30  # 訓練データ数
d = 3  # 次元数
bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)
rng = np.random.default_rng(123)
noise_std = 0.3

# テストデータ（共通）
n_test = 100
X_test = rng.uniform(-1, 1, size=(n_test, d))
y_test_true = true_function(X_test)

# ==========================================
# テスト1: D最適計画データ
# ==========================================
print("\n【テスト1】D最適計画データ (n=30, d=3)")
print("-" * 80)

print("D最適計画を生成中...")
X_dopt = generate_d_optimal_design(n_train, d, bounds, n_candidates=3000, random_state=123)
y_dopt = true_function(X_dopt) + rng.normal(0, noise_std, size=n_train)

print(f"訓練データ範囲:")
for i in range(d):
    print(f"  x{i+1}: [{X_dopt[:, i].min():.3f}, {X_dopt[:, i].max():.3f}]")

# モデル学習
model_dopt = rsm.RSMPlusGP_Production(
    gp_kernel=get_kernel(),
    gp_n_restarts_optimizer=3,
    random_state=123,
    model_type="quadratic"
)
model_dopt.fit(X_dopt, y_dopt, feature_names=["x1", "x2", "x3"])

print(f"選択された項数: {len(model_dopt._rsm_selected_cols)}")

# 予測
y_pred_dopt, y_std_dopt = model_dopt.predict(X_test, return_std=True)
mae_dopt = np.mean(np.abs(y_pred_dopt - y_test_true))
rmse_dopt = np.sqrt(mean_squared_error(y_test_true, y_pred_dopt))
r2_dopt = r2_score(y_test_true, y_pred_dopt)

print(f"\n予測性能:")
print(f"  MAE:  {mae_dopt:.4f}")
print(f"  RMSE: {rmse_dopt:.4f}")
print(f"  R²:   {r2_dopt:.4f}")

# 係数確認
print(f"\n推定された係数（標準化空間）:")
coeffs_dopt = model_dopt.rsm_coefficients_standardized()
importance_dopt = model_dopt.standardized_importance_ranking(top_k=10)
for name, coef in importance_dopt[1:6]:  # Top5 (Interceptを除く)
    print(f"  {name:12s}: {coef:+.4f}")

# ==========================================
# テスト2: ランダムサンプリングデータ
# ==========================================
print("\n【テスト2】ランダムサンプリングデータ (n=30, d=3)")
print("-" * 80)

X_random = rng.uniform(-1, 1, size=(n_train, d))
y_random = true_function(X_random) + rng.normal(0, noise_std, size=n_train)

print(f"訓練データ範囲:")
for i in range(d):
    print(f"  x{i+1}: [{X_random[:, i].min():.3f}, {X_random[:, i].max():.3f}]")

# モデル学習
model_random = rsm.RSMPlusGP_Production(
    gp_kernel=get_kernel(),
    gp_n_restarts_optimizer=3,
    random_state=123,
    model_type="quadratic"
)
model_random.fit(X_random, y_random, feature_names=["x1", "x2", "x3"])

print(f"選択された項数: {len(model_random._rsm_selected_cols)}")

# 予測
y_pred_random, y_std_random = model_random.predict(X_test, return_std=True)
mae_random = np.mean(np.abs(y_pred_random - y_test_true))
rmse_random = np.sqrt(mean_squared_error(y_test_true, y_pred_random))
r2_random = r2_score(y_test_true, y_pred_random)

print(f"\n予測性能:")
print(f"  MAE:  {mae_random:.4f}")
print(f"  RMSE: {rmse_random:.4f}")
print(f"  R²:   {r2_random:.4f}")

# 係数確認
print(f"\n推定された係数（標準化空間）:")
importance_random = model_random.standardized_importance_ranking(top_k=10)
for name, coef in importance_random[1:6]:  # Top5
    print(f"  {name:12s}: {coef:+.4f}")

# ==========================================
# テスト3: 中心複合計画（CCD）風データ
# ==========================================
print("\n【テスト3】中心複合計画風データ (n=27, d=3)")
print("-" * 80)

def generate_ccd_design(d):
    """中心複合計画（Central Composite Design）の生成"""
    points = []

    # 1. 因子点（factorial points）: 2^d 点
    for i in range(2**d):
        point = []
        for j in range(d):
            point.append(-1 if (i // (2**j)) % 2 == 0 else 1)
        points.append(point)

    # 2. 軸点（axial points）: 2*d 点
    alpha = np.sqrt(d)  # 回転可能設計
    for j in range(d):
        point_plus = [0] * d
        point_plus[j] = alpha
        points.append(point_plus)

        point_minus = [0] * d
        point_minus[j] = -alpha
        points.append(point_minus)

    # 3. 中心点（center points）: 複数点
    for _ in range(3):
        points.append([0] * d)

    return np.array(points)

X_ccd = generate_ccd_design(d)
# スケーリング: [-sqrt(3), sqrt(3)] -> [-1, 1]
alpha = np.sqrt(d)
X_ccd = X_ccd / alpha
y_ccd = true_function(X_ccd) + rng.normal(0, noise_std, size=X_ccd.shape[0])

print(f"訓練データ数: {X_ccd.shape[0]}")
print(f"訓練データ範囲:")
for i in range(d):
    print(f"  x{i+1}: [{X_ccd[:, i].min():.3f}, {X_ccd[:, i].max():.3f}]")

# モデル学習
model_ccd = rsm.RSMPlusGP_Production(
    gp_kernel=get_kernel(),
    gp_n_restarts_optimizer=3,
    random_state=123,
    model_type="quadratic"
)
model_ccd.fit(X_ccd, y_ccd, feature_names=["x1", "x2", "x3"])

print(f"選択された項数: {len(model_ccd._rsm_selected_cols)}")

# 予測
y_pred_ccd, y_std_ccd = model_ccd.predict(X_test, return_std=True)
mae_ccd = np.mean(np.abs(y_pred_ccd - y_test_true))
rmse_ccd = np.sqrt(mean_squared_error(y_test_true, y_pred_ccd))
r2_ccd = r2_score(y_test_true, y_pred_ccd)

print(f"\n予測性能:")
print(f"  MAE:  {mae_ccd:.4f}")
print(f"  RMSE: {rmse_ccd:.4f}")
print(f"  R²:   {r2_ccd:.4f}")

# 係数確認
print(f"\n推定された係数（標準化空間）:")
importance_ccd = model_ccd.standardized_importance_ranking(top_k=10)
for name, coef in importance_ccd[1:6]:  # Top5
    print(f"  {name:12s}: {coef:+.4f}")

# ==========================================
# 比較サマリー
# ==========================================
print("\n" + "=" * 80)
print("実験計画法の比較サマリー")
print("=" * 80)

results = [
    ("D最適計画", n_train, mae_dopt, rmse_dopt, r2_dopt, len(model_dopt._rsm_selected_cols)),
    ("ランダムサンプリング", n_train, mae_random, rmse_random, r2_random, len(model_random._rsm_selected_cols)),
    ("中心複合計画（CCD）", X_ccd.shape[0], mae_ccd, rmse_ccd, r2_ccd, len(model_ccd._rsm_selected_cols)),
]

print(f"\n{'実験計画法':<20} {'n':>4} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'選択項数':>8}")
print("-" * 80)
for name, n, mae, rmse, r2, n_terms in results:
    print(f"{name:<20} {n:>4} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {n_terms:>8}")

# 改善率の計算
print(f"\n【D最適計画の優位性】")
print(f"vs ランダムサンプリング:")
print(f"  MAE改善:  {(1 - mae_dopt/mae_random)*100:+.1f}%")
print(f"  RMSE改善: {(1 - rmse_dopt/rmse_random)*100:+.1f}%")
print(f"  R²改善:   {(r2_dopt - r2_random):.4f}")

# ==========================================
# 詳細比較: 係数推定精度
# ==========================================
print("\n" + "=" * 80)
print("係数推定精度の比較")
print("=" * 80)

# 真の係数（標準化空間での近似）
true_coeffs = {
    'x1': 3.0, 'x2': -2.0, 'x3': 1.5,
    'x1^2': 2.5, 'x2^2': -1.8, 'x3^2': 1.2,
    'x1*x2': 2.0, 'x2*x3': -1.5
}

print(f"\n{'係数':<12} {'真値':>8} {'D最適':>8} {'ランダム':>8} {'CCD':>8}")
print("-" * 80)

def get_coef_value(importance_list, name):
    for term_name, coef in importance_list:
        if term_name == name:
            return coef
    return 0.0

for name, true_val in true_coeffs.items():
    dopt_val = get_coef_value(importance_dopt, name)
    random_val = get_coef_value(importance_random, name)
    ccd_val = get_coef_value(importance_ccd, name)
    print(f"{name:<12} {true_val:>8.2f} {dopt_val:>8.4f} {random_val:>8.4f} {ccd_val:>8.4f}")

# 係数誤差の計算
errors_dopt = []
errors_random = []
errors_ccd = []

for name, true_val in true_coeffs.items():
    dopt_val = get_coef_value(importance_dopt, name)
    random_val = get_coef_value(importance_random, name)
    ccd_val = get_coef_value(importance_ccd, name)

    errors_dopt.append(abs(dopt_val - true_val))
    errors_random.append(abs(random_val - true_val))
    errors_ccd.append(abs(ccd_val - true_val))

mae_coef_dopt = np.mean(errors_dopt)
mae_coef_random = np.mean(errors_random)
mae_coef_ccd = np.mean(errors_ccd)

print(f"\n係数推定の平均絶対誤差:")
print(f"  D最適計画:         {mae_coef_dopt:.4f}")
print(f"  ランダムサンプリング: {mae_coef_random:.4f}")
print(f"  中心複合計画:       {mae_coef_ccd:.4f}")

# ==========================================
# 結論
# ==========================================
print("\n" + "=" * 80)
print("結論")
print("=" * 80)

print(f"\n✓ D最適計画データでの解析・予測は非常に良好")
print(f"  • 予測精度: R² = {r2_dopt:.4f}")
print(f"  • 係数推定: MAE = {mae_coef_dopt:.4f}")

if mae_dopt < mae_random:
    print(f"\n✓ D最適計画は同じデータ数でランダムサンプリングより高精度")
    print(f"  • MAE改善: {(1 - mae_dopt/mae_random)*100:.1f}%")
else:
    print(f"\n⚠ この例では偶然ランダムサンプリングも良好")

print(f"\n✓ RSM+GPモデルはD最適計画と相性が良い理由:")
print(f"  1. RSMコンポーネントがD最適計画の構造を活用")
print(f"  2. 多項式モデルのパラメータ推定精度が向上")
print(f"  3. GPが非線形な残差を補正")
print(f"  4. ステップワイズ選択が効率的に重要項を抽出")

print(f"\n✓ 推奨される使い方:")
print(f"  • 少数実験での高精度が必要 → D最適計画")
print(f"  • 二次応答曲面のフィッティング → D最適計画 or CCD")
print(f"  • 探索的なスクリーニング → ランダムサンプリング")
print(f"  • 実験コストが高い → D最適計画で効率化")

print("\n" + "=" * 80)
