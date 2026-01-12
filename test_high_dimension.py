"""
高次元での高次項検知テスト
GPは高次元で機能するか？次元の呪いの影響は？
"""
import numpy as np
import rsm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("高次元での高次項検知: GPは次元の呪いに勝てるか？")
print("=" * 80)

rng = np.random.default_rng(42)

# ===============================================================
# テスト1: 2次元（ベースライン）- sin/cos
# ===============================================================
print("\n" + "=" * 80)
print("テスト1: 2次元（ベースライン）- sin/cos")
print("=" * 80)

def nonlinear_2d(X):
    """2次元の非線形関数"""
    x1, x2 = X[:, 0], X[:, 1]
    return (10.0 + 2.0*x1 - 1.5*x2 +
            1.5*x1**2 - 1.0*x2**2 +
            5.0*np.sin(3*x1) + 4.0*np.cos(2*x2))

print("\n真のモデル式:")
print("y = 10 + 2x₁ - 1.5x₂ + 1.5x₁² - x₂²")
print("    + 5·sin(3x₁) + 4·cos(2x₂)")
print(f"\n次元数: d=2")
print(f"データ数: n=150")

X_train1 = rng.uniform(-1, 1, size=(150, 2))
y_train1 = nonlinear_2d(X_train1) + rng.normal(0, 0.3, size=150)
X_test1 = rng.uniform(-1, 1, size=(50, 2))
y_test1 = nonlinear_2d(X_test1)

kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))
model1 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model1.fit(X_train1, y_train1)
info1 = model1.diagnose(X_test1, y_test1, display=True)

# ===============================================================
# テスト2: 5次元 - sin/cos（一部の次元のみ）
# ===============================================================
print("\n" + "=" * 80)
print("テスト2: 5次元 - sin/cos（一部の次元のみ非線形）")
print("=" * 80)

def nonlinear_5d(X):
    """5次元、最初の2次元のみ非線形"""
    # 線形項
    linear = 10.0 + 2.0*X[:, 0] - 1.5*X[:, 1] + 1.0*X[:, 2] - 0.8*X[:, 3] + 0.5*X[:, 4]
    # 二次項（すべての次元）
    quadratic = (1.5*X[:, 0]**2 - 1.0*X[:, 1]**2 +
                 0.8*X[:, 2]**2 - 0.6*X[:, 3]**2 + 0.4*X[:, 4]**2)
    # 非線形項（最初の2次元のみ）
    nonlinear = 5.0*np.sin(3*X[:, 0]) + 4.0*np.cos(2*X[:, 1])
    return linear + quadratic + nonlinear

print("\n真のモデル式:")
print("y = 10 + 2x₁ - 1.5x₂ + x₃ - 0.8x₄ + 0.5x₅")
print("    + 1.5x₁² - x₂² + 0.8x₃² - 0.6x₄² + 0.4x₅²")
print("    + 5·sin(3x₁) + 4·cos(2x₂)  ← 最初の2次元のみ非線形")
print(f"\n次元数: d=5")
print(f"データ数: n=150 (データ密度は2次元の1/2.5に低下)")

X_train2 = rng.uniform(-1, 1, size=(150, 5))
y_train2 = nonlinear_5d(X_train2) + rng.normal(0, 0.3, size=150)
X_test2 = rng.uniform(-1, 1, size=(50, 5))
y_test2 = nonlinear_5d(X_test2)

model2 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model2.fit(X_train2, y_train2)
info2 = model2.diagnose(X_test2, y_test2, display=True)

# ===============================================================
# テスト3: 5次元、データ量を増やす（n=500）
# ===============================================================
print("\n" + "=" * 80)
print("テスト3: 5次元、データ量を増やす（n=500）")
print("=" * 80)

print("\n真のモデル式: テスト2と同じ")
print(f"\n次元数: d=5")
print(f"データ数: n=500 (データ密度改善)")
print("期待: データ量増加でGP性能回復")

X_train3 = rng.uniform(-1, 1, size=(500, 5))
y_train3 = nonlinear_5d(X_train3) + rng.normal(0, 0.3, size=500)
X_test3 = rng.uniform(-1, 1, size=(100, 5))
y_test3 = nonlinear_5d(X_test3)

model3 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model3.fit(X_train3, y_train3)
info3 = model3.diagnose(X_test3, y_test3, display=True)

# ===============================================================
# テスト4: 10次元（極端な高次元）
# ===============================================================
print("\n" + "=" * 80)
print("テスト4: 10次元（極端な高次元）")
print("=" * 80)

def nonlinear_10d(X):
    """10次元、最初の2次元のみ非線形"""
    # 線形項
    linear = 10.0 + sum([(2.0 - 0.3*i) * X[:, i] for i in range(10)])
    # 二次項
    quadratic = sum([(1.5 - 0.2*i) * X[:, i]**2 for i in range(10)])
    # 非線形項（最初の2次元のみ）
    nonlinear = 5.0*np.sin(3*X[:, 0]) + 4.0*np.cos(2*X[:, 1])
    return linear + quadratic + nonlinear

print("\n真のモデル式:")
print("y = 10 + Σ(2-0.3i)·xᵢ + Σ(1.5-0.2i)·xᵢ²")
print("    + 5·sin(3x₁) + 4·cos(2x₂)  ← 最初の2次元のみ非線形")
print(f"\n次元数: d=10")
print(f"データ数: n=300")
print("期待: 次元の呪いの影響が顕著、GP性能低下？")

X_train4 = rng.uniform(-1, 1, size=(300, 10))
y_train4 = nonlinear_10d(X_train4) + rng.normal(0, 0.3, size=300)
X_test4 = rng.uniform(-1, 1, size=(100, 10))
y_test4 = nonlinear_10d(X_test4)

model4 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model4.fit(X_train4, y_train4)
info4 = model4.diagnose(X_test4, y_test4, display=True)

# ===============================================================
# テスト5: 10次元、大量データ（n=1000）
# ===============================================================
print("\n" + "=" * 80)
print("テスト5: 10次元、大量データ（n=1000）")
print("=" * 80)

print("\n真のモデル式: テスト4と同じ")
print(f"\n次元数: d=10")
print(f"データ数: n=1000")
print("期待: 大量データで次元の呪いを克服？")

X_train5 = rng.uniform(-1, 1, size=(1000, 10))
y_train5 = nonlinear_10d(X_train5) + rng.normal(0, 0.3, size=1000)
X_test5 = rng.uniform(-1, 1, size=(100, 10))
y_test5 = nonlinear_10d(X_test5)

model5 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model5.fit(X_train5, y_train5)
info5 = model5.diagnose(X_test5, y_test5, display=True)

# ===============================================================
# 総合比較
# ===============================================================
print("\n" + "=" * 80)
print("次元の呪いの影響: 総合比較")
print("=" * 80)

results = [
    ("2次元 (n=150)", 2, 150, info1),
    ("5次元 (n=150)", 5, 150, info2),
    ("5次元 (n=500)", 5, 500, info3),
    ("10次元 (n=300)", 10, 300, info4),
    ("10次元 (n=1000)", 10, 1000, info5),
]

print(f"\n{'ケース':20} {'次元':>6} {'n':>6} {'n/d':>8} {'R²':>10} {'GP寄与':>10} {'GP改善':>10} {'検知'}")
print("-" * 95)

for name, d, n, info in results:
    n_per_d = n / d
    r2 = info['r2']
    gp_contrib = info['gp_contribution']
    gp_improve = info['improvement']
    detected = info['higher_order_strength']

    print(f"{name:20} {d:>6} {n:>6} {n_per_d:>8.1f} {r2:>10.4f} {gp_contrib:>9.1f}% {gp_improve:>9.1f}% {detected:>12}")

print("\n" + "=" * 80)
print("データ密度の影響")
print("=" * 80)

print("\nデータ密度 (n/d) と性能:")
print("  • 2次元 (n=150): n/d=75.0")
print("  • 5次元 (n=150): n/d=30.0  (密度40%に低下)")
print("  • 5次元 (n=500): n/d=100.0 (密度改善)")
print("  • 10次元(n=300): n/d=30.0  (密度40%に低下)")
print("  • 10次元(n=1000): n/d=100.0 (密度改善)")

print("\n" + "=" * 80)
print("結論")
print("=" * 80)

print("\n🔍 次元の呪いの影響:")
print()
print("1. **2次元（ベースライン）**:")
print(f"   R²: {info1['r2']:.4f}")
print(f"   GP寄与度: {info1['gp_contribution']:.1f}%")
print(f"   高次項検知: {info1['higher_order_strength']}")
print()

print("2. **5次元（n=150）**:")
print(f"   R²: {info2['r2']:.4f}")
print(f"   GP寄与度: {info2['gp_contribution']:.1f}%")
print(f"   高次項検知: {info2['higher_order_strength']}")
if info2['gp_contribution'] < info1['gp_contribution']:
    print("   → GPの検知能力が低下（次元の呪い）")
else:
    print("   → GPは次元増加に対応")
print()

print("3. **5次元（n=500）**:")
print(f"   R²: {info3['r2']:.4f}")
print(f"   GP寄与度: {info3['gp_contribution']:.1f}%")
print(f"   高次項検知: {info3['higher_order_strength']}")
if info3['gp_contribution'] > info2['gp_contribution']:
    print("   → データ量増加でGP性能回復 ✓")
print()

print("4. **10次元（n=300）**:")
print(f"   R²: {info4['r2']:.4f}")
print(f"   GP寄与度: {info4['gp_contribution']:.1f}%")
print(f"   高次項検知: {info4['higher_order_strength']}")
if info4['gp_contribution'] < info1['gp_contribution']:
    print("   → 極端な高次元でGP性能低下")
print()

print("5. **10次元（n=1000）**:")
print(f"   R²: {info5['r2']:.4f}")
print(f"   GP寄与度: {info5['gp_contribution']:.1f}%")
print(f"   高次項検知: {info5['higher_order_strength']}")
if info5['gp_contribution'] > info4['gp_contribution']:
    print("   → 大量データで次元の呪いを克服 ✓")

print("\n" + "=" * 80)
print("重要な知見")
print("=" * 80)

print("\n✅ GPと次元の関係:")
print("  1. 低次元（d=2-5）: GPは非線形性を効果的に捕捉")
print("  2. 高次元（d=10以上）: 次元の呪いの影響を受ける可能性")
print("  3. データ密度（n/d）が重要: n/d≥100を推奨")

print("\n✅ RSMの利点:")
print("  • 高次元でも安定（線形・二次項のみ）")
print("  • 少ないデータでも機能")
print("  • RSM+GPの組み合わせが最適")

print("\n✅ 実用的な推奨:")
print("  • d≤5: n≥100で高次項検知可能")
print("  • d=5-10: n≥200を推奨（n/d≥20）")
print("  • d>10: n≥500を推奨（n/d≥50）、またはRSMのみ")

print("\n→ GPは低〜中次元で強力、高次元ではデータ量が鍵！")

print("\n" + "=" * 80)
