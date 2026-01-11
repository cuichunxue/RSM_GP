"""
診断ツールの実際のデモ
"""
import numpy as np
import rsm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 診断ツールのクラスをインポート（簡易版を再定義）
class DiagnosticReport:
    """診断レポートのクラス"""

    def __init__(self, model, X_test=None, y_test_true=None):
        self.model = model
        self.X_test = X_test if X_test is not None else model._X_train
        self.y_test_true = y_test_true if y_test_true is not None else model._y_train
        self.info = self._collect_diagnostics()

    def _collect_diagnostics(self):
        info = {}
        info['n_train'] = self.model._X_train.shape[0]
        info['n_features'] = self.model._X_train.shape[1]
        info['n_selected_terms'] = len(self.model._rsm_selected_cols)
        info['n_total_terms'] = len(self.model.poly.powers_)

        # RSMのみの予測
        Z_test = self.model.x_scaler.transform(self.X_test)
        Zpoly_test = self.model.poly.transform(Z_test)[:, self.model._rsm_selected_cols]
        t_rsm = self.model.lin.predict(Zpoly_test)
        y_rsm_only = self.model.y_scaler.inverse_transform(t_rsm.reshape(-1, 1)).ravel()

        # RSM+GPの予測
        y_full = self.model.predict(self.X_test, return_std=False)

        # 寄与度計算
        rsm_var = np.var(y_rsm_only - np.mean(self.y_test_true))
        full_var = np.var(y_full - np.mean(self.y_test_true))
        gp_correction_var = np.var(y_full - y_rsm_only)

        info['rsm_contribution'] = rsm_var / (rsm_var + gp_correction_var) * 100
        info['gp_contribution'] = gp_correction_var / (rsm_var + gp_correction_var) * 100

        # 精度指標
        info['r2'] = r2_score(self.y_test_true, y_full)
        info['rmse'] = np.sqrt(np.mean((self.y_test_true - y_full)**2))
        info['mae'] = np.mean(np.abs(self.y_test_true - y_full))
        info['rsm_only_r2'] = r2_score(self.y_test_true, y_rsm_only)
        info['improvement'] = (info['r2'] - info['rsm_only_r2']) * 100

        # カーネルパラメータ
        kernel_params = self.model.gp.kernel_
        info['length_scale'] = kernel_params.k1.k2.length_scale
        info['constant_value'] = kernel_params.k1.k1.constant_value
        info['noise_level'] = kernel_params.k2.noise_level

        # 評価
        ls = info['length_scale']
        if ls < 0.05:
            info['kernel_status'] = "短すぎ（過学習の可能性）"
        elif ls > 10.0:
            info['kernel_status'] = "長すぎ（過平滑化の可能性）"
        else:
            info['kernel_status'] = "適切 ✓"

        return info

    def _generate_recommendations(self):
        recommendations = []
        info = self.info

        if info['r2'] > 0.99:
            recommendations.append("✓ モデルは非常に良好に機能しています")
        elif info['r2'] > 0.95:
            recommendations.append("✓ モデルは良好に機能しています")
            if info['n_train'] < 100:
                recommendations.append("  さらなる精度向上にはデータ追加推奨（n→100-150）")
        elif info['r2'] > 0.90:
            recommendations.append("⚠ モデルの精度は実用レベルですが改善余地があります")
            recommendations.append("  データ数を増やすことを推奨（現在n={})".format(info['n_train']))
        else:
            recommendations.append("⚠ モデルの精度が低いです")
            recommendations.append("  データ数を大幅に増やしてください")

        if info['improvement'] < 5:
            recommendations.append("  GPの改善が小さいです（{:.1f}%）".format(info['improvement']))
            if info['n_train'] < 100:
                recommendations.append("  → データを増やすとGPが高次項を捕捉（n≥100推奨）")

        if info['kernel_status'] != "適切 ✓":
            recommendations.append("⚠ カーネルパラメータ: {}".format(info['kernel_status']))

        return recommendations

    def display(self):
        print("=" * 80)
        print("モデル診断レポート")
        print("=" * 80)

        print("\n【基本統計】")
        print(f"  訓練データ: n={self.info['n_train']}, d={self.info['n_features']}")
        print(f"  選択項数: {self.info['n_selected_terms']}/{self.info['n_total_terms']}")

        print("\n【予測性能】")
        print(f"  R²:   {self.info['r2']:.4f}")
        print(f"  RMSE: {self.info['rmse']:.4f}")
        print(f"  MAE:  {self.info['mae']:.4f}")

        print("\n【RSM vs GP 寄与度】")
        print(f"  RSM寄与: {self.info['rsm_contribution']:.1f}% (主に線形・二次構造)")
        print(f"  GP寄与:  {self.info['gp_contribution']:.1f}% (残差・高次項補正)")
        print(f"  RSMのみのR²: {self.info['rsm_only_r2']:.4f}")
        print(f"  GP改善:      {self.info['improvement']:.2f}%")

        if self.info['rsm_contribution'] > 80:
            print("  → RSM主導型（二次多項式で十分説明）✓")
        elif self.info['gp_contribution'] > 50:
            print("  → GP主導型（高次・非線形性が支配的）")
        else:
            print("  → バランス良好 ✓")

        print("\n【カーネルパラメータ】")
        print(f"  length_scale:   {self.info['length_scale']:.4f}")
        print(f"  constant_value: {self.info['constant_value']:.4f}")
        print(f"  noise_level:    {self.info['noise_level']:.4f}")
        print(f"  評価: {self.info['kernel_status']}")

        recommendations = self._generate_recommendations()
        print("\n【推奨事項】")
        for rec in recommendations:
            print(f"  {rec}")

        print("\n" + "=" * 80)


# デモ実行
print("=" * 80)
print("診断ツール（model.diagnose()）のデモ")
print("=" * 80)

# テストデータ生成
def true_function(X):
    return (10.0 + 3.0*X[:, 0] - 2.0*X[:, 1] +
            2.0*X[:, 0]**2 - 1.5*X[:, 1]**2 +
            1.5*X[:, 0]**3 + 0.8*X[:, 1]**3)  # 3次項あり

rng = np.random.default_rng(42)

# ケース1: 少数データ（n=30）
print("\n" + "=" * 80)
print("ケース1: 少数データ（n=30）")
print("=" * 80)

X_train1 = rng.uniform(-1, 1, size=(30, 2))
y_train1 = true_function(X_train1) + rng.normal(0, 0.3, size=30)
X_test1 = rng.uniform(-1, 1, size=(50, 2))
y_test1 = true_function(X_test1)

model1 = rsm.RSMPlusGP_Production(
    gp_kernel=C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0)),
    gp_n_restarts_optimizer=5,
    random_state=42
)
model1.fit(X_train1, y_train1)

diagnosis1 = DiagnosticReport(model1, X_test1, y_test1)
diagnosis1.display()

# ケース2: 中規模データ（n=100）
print("\n" + "=" * 80)
print("ケース2: 中規模データ（n=100）")
print("=" * 80)

X_train2 = rng.uniform(-1, 1, size=(100, 2))
y_train2 = true_function(X_train2) + rng.normal(0, 0.3, size=100)
X_test2 = rng.uniform(-1, 1, size=(50, 2))
y_test2 = true_function(X_test2)

model2 = rsm.RSMPlusGP_Production(
    gp_kernel=C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0)),
    gp_n_restarts_optimizer=5,
    random_state=42
)
model2.fit(X_train2, y_train2)

diagnosis2 = DiagnosticReport(model2, X_test2, y_test2)
diagnosis2.display()

# ケース3: 大規模データ（n=200）
print("\n" + "=" * 80)
print("ケース3: 大規模データ（n=200）")
print("=" * 80)

X_train3 = rng.uniform(-1, 1, size=(200, 2))
y_train3 = true_function(X_train3) + rng.normal(0, 0.3, size=200)
X_test3 = rng.uniform(-1, 1, size=(50, 2))
y_test3 = true_function(X_test3)

model3 = rsm.RSMPlusGP_Production(
    gp_kernel=C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0)),
    gp_n_restarts_optimizer=5,
    random_state=42
)
model3.fit(X_train3, y_train3)

diagnosis3 = DiagnosticReport(model3, X_test3, y_test3)
diagnosis3.display()

print("\n" + "=" * 80)
print("まとめ")
print("=" * 80)
print("\n診断ツールを使うことで:")
print("  ✓ モデルの状態を一目で把握")
print("  ✓ RSMとGPの寄与度が分かる")
print("  ✓ データ量によるGP改善率の変化を確認")
print("    - n=30:  GP改善率 低い")
print("    - n=100: GP改善率 中程度")
print("    - n=200: GP改善率 高い")
print("  ✓ 次に何をすべきか推奨事項を提示")
print("\n" + "=" * 80)
