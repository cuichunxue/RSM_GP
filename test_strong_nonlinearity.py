"""
より強い非線形性でのテスト
sin/cosやより大きな係数の高次項を使用
"""
import numpy as np
import rsm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("強い非線形性での高次項検知テスト")
print("=" * 80)

rng = np.random.default_rng(42)
kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))

# ===============================================================
# ケース1: sin/cos を含む完全に非線形な関数
# ===============================================================
print("\n" + "=" * 80)
print("ケース1: sin/cos を含む完全に非線形な関数")
print("=" * 80)

def highly_nonlinear_function(X):
    """sin/cosを含む完全に非線形な関数"""
    return (10.0 + 3.0*X[:, 0] - 2.0*X[:, 1] +
            2.0*X[:, 0]**2 - 1.5*X[:, 1]**2 +
            5.0*np.sin(3*X[:, 0]) + 4.0*np.cos(2*X[:, 1]))  # 強い非線形性

X_train1 = rng.uniform(-1, 1, size=(150, 2))
y_train1 = highly_nonlinear_function(X_train1) + rng.normal(0, 0.3, size=150)
X_test1 = rng.uniform(-1, 1, size=(50, 2))
y_test1 = highly_nonlinear_function(X_test1)

model1 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model1.fit(X_train1, y_train1)
info1 = model1.diagnose(X_test1, y_test1, display=True)

# ===============================================================
# ケース2: 大きな係数の3次・4次項
# ===============================================================
print("\n" + "=" * 80)
print("ケース2: 大きな係数の3次・4次項")
print("=" * 80)

def strong_polynomial_function(X):
    """大きな係数の高次項"""
    return (10.0 + 3.0*X[:, 0] - 2.0*X[:, 1] +
            2.0*X[:, 0]**2 - 1.5*X[:, 1]**2 +
            5.0*X[:, 0]**3 + 4.0*X[:, 1]**3 +
            3.5*X[:, 0]**4 - 2.5*X[:, 1]**4)  # 大きな係数の高次項

X_train2 = rng.uniform(-1, 1, size=(150, 2))
y_train2 = strong_polynomial_function(X_train2) + rng.normal(0, 0.5, size=150)
X_test2 = rng.uniform(-1, 1, size=(50, 2))
y_test2 = strong_polynomial_function(X_test2)

model2 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model2.fit(X_train2, y_train2)
info2 = model2.diagnose(X_test2, y_test2, display=True)

# ===============================================================
# ケース3: 3次項のみで2次項が小さい
# ===============================================================
print("\n" + "=" * 80)
print("ケース3: 3次項が支配的（2次項が相対的に小さい）")
print("=" * 80)

def cubic_dominated_function(X):
    """3次項が支配的"""
    return (10.0 + 2.0*X[:, 0] - 1.5*X[:, 1] +
            0.5*X[:, 0]**2 - 0.3*X[:, 1]**2 +  # 小さな2次項
            4.0*X[:, 0]**3 + 3.5*X[:, 1]**3)   # 大きな3次項

X_train3 = rng.uniform(-1, 1, size=(150, 2))
y_train3 = cubic_dominated_function(X_train3) + rng.normal(0, 0.4, size=150)
X_test3 = rng.uniform(-1, 1, size=(50, 2))
y_test3 = cubic_dominated_function(X_test3)

model3 = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model3.fit(X_train3, y_train3)
info3 = model3.diagnose(X_test3, y_test3, display=True)

# ===============================================================
# サマリー
# ===============================================================
print("\n" + "=" * 80)
print("検知結果サマリー")
print("=" * 80)

cases = [
    ("ケース1: sin/cos", info1),
    ("ケース2: 大係数高次項", info2),
    ("ケース3: 3次支配", info3),
]

print(f"\n{'ケース':20} {'検知':8} {'強度':12} {'GP寄与':10} {'GP改善':10} {'R²':10}")
print("-" * 80)

for name, info in cases:
    detected = "はい" if info['higher_order_detected'] else "いいえ"
    strength = info['higher_order_strength']
    gp_contrib = f"{info['gp_contribution']:.1f}%"
    gp_improve = f"{info['improvement']:.1f}%"
    r2 = f"{info['r2']:.4f}"

    print(f"{name:20} {detected:8} {strength:12} {gp_contrib:10} {gp_improve:10} {r2:10}")

print("\n" + "=" * 80)
print("判定基準の妥当性検証")
print("=" * 80)

print("\n現在の判定基準:")
print("  強い:   GP寄与>30% または GP改善>15%")
print("  中程度: GP寄与>15% または GP改善>8%")
print("  弱い:   GP寄与>5%  または GP改善>3%")
print("  なし:   上記に該当しない")

print("\n" + "=" * 80)
