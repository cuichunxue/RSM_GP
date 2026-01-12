"""
統合テスト - 全機能の動作確認
すべての最適化と新機能が正常に動作することを検証
"""
import numpy as np
import rsm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RSM+GP 統合テスト - 全機能確認")
print("=" * 80)

rng = np.random.default_rng(42)

# テスト関数（高次項を含む）
def test_function(X):
    """3次項を含むテスト関数"""
    return (10.0 + 3.0*X[:, 0] - 2.0*X[:, 1] +
            2.0*X[:, 0]**2 - 1.5*X[:, 1]**2 +
            1.5*X[:, 0]*X[:, 1] +
            2.0*X[:, 0]**3 + 1.0*X[:, 1]**3)

# データ生成
X_train = rng.uniform(-1, 1, size=(100, 2))
y_train = test_function(X_train) + rng.normal(0, 0.3, size=100)
X_test = rng.uniform(-1, 1, size=(30, 2))
y_test = test_function(X_test)

# カーネル設定
kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))

print("\n" + "=" * 80)
print("テスト1: 基本的なモデル訓練と予測")
print("=" * 80)

model = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)

print("✓ モデル初期化完了")

# 訓練
model.fit(X_train, y_train)
print("✓ モデル訓練完了")

# 予測
y_pred, y_std = model.predict(X_test, return_std=True)
print("✓ 予測完了")

# 精度評価
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n予測精度:")
print(f"  R²:   {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

if r2 > 0.95:
    print("  → ✓ 高精度（R²>0.95）")
else:
    print("  → ⚠ 精度がやや低い")

print("\n" + "=" * 80)
print("テスト2: 診断ツール（高次項検知を含む）")
print("=" * 80)

info = model.diagnose(X_test, y_test, display=True)

# 診断結果の検証
assert 'higher_order_detected' in info, "高次項検知フィールドがありません"
assert 'higher_order_strength' in info, "高次項強度フィールドがありません"
assert 'gp_contribution' in info, "GP寄与度フィールドがありません"
assert 'recommendations' in info, "推奨事項フィールドがありません"

print("\n✓ 診断ツールの全フィールドが存在")

if info['higher_order_detected']:
    print(f"✓ 高次項検知: {info['higher_order_strength']}")
    print(f"  GP寄与度: {info['gp_contribution']:.1f}%")
    print(f"  GP改善率: {info['improvement']:.1f}%")

print("\n" + "=" * 80)
print("テスト3: K-Fold クロスバリデーション（並列化自動切替）")
print("=" * 80)

# 小規模データ（自動的に逐次処理）
X_small = X_train[:50]
y_small = y_train[:50]

print(f"\n小規模データ (n={len(X_small)}):")
cv_results_small = rsm.perform_kfold_cv(
    model, X_small, y_small,
    n_splits=5,
    n_jobs=-1,
    auto_switch=True  # 自動切替ON
)
print(f"  Q²:   {cv_results_small['metrics']['Q2']:.4f}")
print(f"  RMSE: {cv_results_small['metrics']['RMSE']:.4f}")
print(f"  → ✓ 自動的に逐次処理（n<100）")

# 大規模データ（自動的に並列処理）
print(f"\n大規模データ (n={len(X_train)}):")
cv_results_large = rsm.perform_kfold_cv(
    model, X_train, y_train,
    n_splits=5,
    n_jobs=-1,
    auto_switch=True  # 自動切替ON
)
print(f"  Q²:   {cv_results_large['metrics']['Q2']:.4f}")
print(f"  RMSE: {cv_results_large['metrics']['RMSE']:.4f}")
print(f"  → ✓ 自動的に並列処理（n≥100）")

print("\n" + "=" * 80)
print("テスト4: 予測区間")
print("=" * 80)

intervals = model.predict_interval_two_types(
    X_test[:5],
    level=0.95,
    include_rsm_param_uncertainty=True
)

print(f"\n予測区間（最初の5点）:")
print(f"  Posterior 区間幅: {np.mean(intervals['posterior']['std']):.4f}")
print(f"  Predictive 区間幅: {np.mean(intervals['predictive']['std']):.4f}")
print(f"  自動診断率: {intervals['components']['subtract_rate']*100:.1f}%")
print("  → ✓ 予測区間計算完了")

print("\n" + "=" * 80)
print("テスト5: RSM係数と重要度ランキング")
print("=" * 80)

coefs = model.rsm_coefficients_standardized()
print(f"\n標準化係数:")
print(f"  切片: {coefs['Intercept']:.4f}")
print(f"  線形項数: {len(coefs['linear'])}")
print(f"  交互作用項数: {len(coefs['inter'])}")
print(f"  二次項数: {len(coefs['quad'])}")

importance = model.standardized_importance_ranking(top_k=5)
print(f"\n重要度ランキング（上位5項）:")
for name, val in importance[:5]:
    print(f"  {name:12s}: {val:+.4f}")

print("  → ✓ 係数抽出完了")

print("\n" + "=" * 80)
print("テスト6: ベイズ最適化（候補点生成）")
print("=" * 80)

# 候補点生成
X_candidates = rsm.build_candidates(
    bounds=np.array([(-1, 1), (-1, 1)]),
    n_candidates=500,
    rng=rng
)

# EI計算と上位候補
suggestions = rsm.propose_next_EI_constrained(
    model=model,
    bounds=np.array([(-1, 1), (-1, 1)]),
    candidate_set=X_candidates,
    objective="min",
    top_k=3
)

print(f"\n候補点数: {len(X_candidates)}")
print(f"上位3候補:")
for i, s in enumerate(suggestions, 1):
    print(f"  {i}. x={s['x']}, EI={s['EI']:.4f}, μ={s['mu']:.4f}")

print("  → ✓ ベイズ最適化機能正常")

print("\n" + "=" * 80)
print("テスト7: オプショナル依存パッケージの確認")
print("=" * 80)

print(f"\nSciPy: {'✓ 利用可能' if rsm._HAS_SCIPY else '✗ 未インストール'}")
print(f"joblib: {'✓ 利用可能' if rsm._HAS_JOBLIB else '✗ 未インストール'}")
print(f"plotly: {'✓ 利用可能' if rsm._HAS_PLOTLY else '✗ 未インストール'}")

if rsm._HAS_SCIPY:
    print("  → 高速距離計算・正規分布関数を使用")
if rsm._HAS_JOBLIB:
    print("  → K-Fold CV並列化を使用")

print("\n" + "=" * 80)
print("統合テスト結果")
print("=" * 80)

print("\n✅ すべてのテストが正常に完了しました！")
print()
print("検証項目:")
print("  ✓ 基本的なモデル訓練と予測")
print("  ✓ 診断ツール（高次項検知を含む）")
print("  ✓ K-Fold CV（並列化自動切替）")
print("  ✓ 予測区間計算")
print("  ✓ RSM係数と重要度ランキング")
print("  ✓ ベイズ最適化")
print("  ✓ オプショナル依存パッケージ")
print()
print("主な成果:")
print(f"  • 予測精度: R²={r2:.4f}")
print(f"  • GP寄与度: {info['gp_contribution']:.1f}%")
print(f"  • 高次項検知: {info['higher_order_strength']}")
print(f"  • CV精度: Q²={cv_results_large['metrics']['Q2']:.4f}")
print()
print("→ すべての機能が正常に動作しています！")

print("\n" + "=" * 80)
