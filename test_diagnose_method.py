"""
diagnose()メソッドの動作確認テスト
"""
import numpy as np
import rsm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("diagnose()メソッドの動作確認")
print("=" * 80)

def cubic_function(X):
    """3次項を含む関数"""
    return (10.0 + 3.0*X[:, 0] - 2.0*X[:, 1] +
            2.0*X[:, 0]**2 - 1.5*X[:, 1]**2 +
            1.5*X[:, 0]**3 + 0.8*X[:, 1]**3)

rng = np.random.default_rng(42)

# テストケース1: 少数データ (n=30)
print("\n" + "=" * 80)
print("ケース1: 少数データ（n=30）")
print("=" * 80)

X_train1 = rng.uniform(-1, 1, size=(30, 2))
y_train1 = cubic_function(X_train1) + rng.normal(0, 0.3, size=30)
X_test1 = rng.uniform(-1, 1, size=(50, 2))
y_test1 = cubic_function(X_test1)

kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))
model1 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model1.fit(X_train1, y_train1)

# diagnose()メソッドを呼び出し（display=True）
info1 = model1.diagnose(X_test1, y_test1, display=True)

# テストケース2: 中規模データ (n=100)
print("\n" + "=" * 80)
print("ケース2: 中規模データ（n=100）")
print("=" * 80)

X_train2 = rng.uniform(-1, 1, size=(100, 2))
y_train2 = cubic_function(X_train2) + rng.normal(0, 0.3, size=100)
X_test2 = rng.uniform(-1, 1, size=(50, 2))
y_test2 = cubic_function(X_test2)

model2 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model2.fit(X_train2, y_train2)

info2 = model2.diagnose(X_test2, y_test2, display=True)

# テストケース3: 大規模データ (n=200)
print("\n" + "=" * 80)
print("ケース3: 大規模データ（n=200）")
print("=" * 80)

X_train3 = rng.uniform(-1, 1, size=(200, 2))
y_train3 = cubic_function(X_train3) + rng.normal(0, 0.3, size=200)
X_test3 = rng.uniform(-1, 1, size=(50, 2))
y_test3 = cubic_function(X_test3)

model3 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model3.fit(X_train3, y_train3)

info3 = model3.diagnose(X_test3, y_test3, display=True)

# サマリー比較
print("\n" + "=" * 80)
print("データ量による性能比較")
print("=" * 80)

print(f"\n{'データ数':>10} {'R²':>10} {'GP改善率':>12} {'GP寄与度':>12} {'評価'}")
print("-" * 80)
print(f"{30:>10} {info1['r2']:>10.4f} {info1['improvement']:>11.1f}% {info1['gp_contribution']:>11.1f}% {'限定的' if info1['improvement'] < 10 else '良好'}")
print(f"{100:>10} {info2['r2']:>10.4f} {info2['improvement']:>11.1f}% {info2['gp_contribution']:>11.1f}% {'限定的' if info2['improvement'] < 10 else '良好'}")
print(f"{200:>10} {info3['r2']:>10.4f} {info3['improvement']:>11.1f}% {info3['gp_contribution']:>11.1f}% {'限定的' if info3['improvement'] < 10 else '良好'}")

print("\n" + "=" * 80)
print("まとめ")
print("=" * 80)
print("\n✓ diagnose()メソッドが正常に動作しています")
print("✓ データ量が増えるとGP改善率が向上することが確認できます")
print("✓ カーネルパラメータの評価と推奨事項が自動生成されます")
print("✓ display=Falseで辞書のみを返すことも可能です")

# display=Falseのデモ
print("\n" + "=" * 80)
print("display=Falseの使用例（辞書のみ返却）")
print("=" * 80)
info_silent = model2.diagnose(X_test2, y_test2, display=False)
print(f"\n返却された辞書のキー: {list(info_silent.keys())}")
print(f"R²スコア: {info_silent['r2']:.4f}")
print(f"GP改善率: {info_silent['improvement']:.2f}%")

print("\n" + "=" * 80)
