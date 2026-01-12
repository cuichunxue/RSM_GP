#!/usr/bin/env python3
"""
次元の呪いを克服する革新的アプローチ

認識を超えた3つの方法を実装：
1. 標準GP（ベースライン）
2. ARD (Automatic Relevance Determination) カーネル - 重要な次元を自動選択
3. Additive GP - 各次元を独立に処理して合計

テスト: 10次元、n=300（従来は失敗したケース）
"""

import sys
import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 警告を抑制
warnings.filterwarnings('ignore')

# rsm.py をインポート
sys.path.insert(0, '/home/user/RSM_GP')
from rsm import RSMPlusGP_Production


def generate_10d_data(n_samples, random_state=42):
    """
    10次元データ生成（最初の2次元のみ非線形）
    """
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1, 1, size=(n_samples, 10))

    # 線形項（全次元）
    linear = 10.0
    for i in range(10):
        linear += (2.0 - 0.3*i) * X[:, i]

    # 二次項（全次元）
    quadratic = 0.0
    for i in range(10):
        quadratic += (1.5 - 0.2*i) * X[:, i]**2

    # 非線形項（最初の2次元のみ）- これがGPの検知対象
    nonlinear = 5.0*np.sin(3*X[:, 0]) + 4.0*np.cos(2*X[:, 1])

    y = linear + quadratic + nonlinear
    noise = rng.normal(0, 0.1, size=n_samples)
    y += noise

    return X, y


class RSMPlusGP_ARD(RSMPlusGP_Production):
    """
    ARD (Automatic Relevance Determination) カーネルを使用したRSM+GP

    各次元に異なる length_scale を持たせることで、
    重要な次元を自動的に選択し、次元の呪いを緩和
    """

    def __init__(self, n_features=10, **kwargs):
        """ARDカーネルを設定"""
        # ARD: 各次元に独立した length_scale
        ard_kernel = (C(1.0, (1e-3, 1e3)) *
                     RBF(length_scale=[1.0]*n_features,  # 各次元に独立
                         length_scale_bounds=(0.1, 10.0)) +
                     WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

        # 親クラスのgp_kernelを設定
        kwargs['gp_kernel'] = ard_kernel
        super().__init__(**kwargs)


class AdditiveGP:
    """
    Additive GP: 各次元を独立に処理して合計

    f(x₁, x₂, ..., xₐ) = f₁(x₁) + f₂(x₂) + ... + fₐ(xₐ)

    利点：
    - 次元の呪いを回避（各GPは1次元）
    - 解釈性が高い（各次元の寄与が明確）
    - 計算効率が良い
    """

    def __init__(self, rsm_model, random_state=42):
        self.rsm_model = rsm_model
        self.random_state = random_state
        self.gps_ = []  # 各次元のGP
        self.fitted_ = False

    def fit(self, X_train, y_train):
        """各次元で独立にGPを学習"""
        # RSM予測を取得
        y_rsm = self.rsm_model.predict(X_train, return_std=False)
        residuals = y_train - y_rsm

        n_features = X_train.shape[1]
        self.gps_ = []

        # 各次元で独立にGPを学習
        for i in range(n_features):
            kernel = (C(1.0, (1e-3, 1e3)) *
                     RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
                     WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)))

            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                random_state=self.random_state + i,
                normalize_y=False
            )

            # i番目の次元のみを使用
            X_i = X_train[:, i:i+1]

            # 残差の1/n_features を各次元に割り当て（初期仮定）
            gp.fit(X_i, residuals / n_features)
            self.gps_.append(gp)

        self.fitted_ = True
        return self

    def predict(self, X_test):
        """各次元のGP予測を合計"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet")

        # RSM予測
        y_rsm = self.rsm_model.predict(X_test, return_std=False)

        # 各次元のGP予測を合計
        gp_correction = np.zeros(X_test.shape[0])
        for i, gp in enumerate(self.gps_):
            X_i = X_test[:, i:i+1]
            gp_correction += gp.predict(X_i)

        return y_rsm + gp_correction


def evaluate_model(model, X_test, y_test, model_name):
    """モデルを評価"""
    # RSMPlusGP系はreturn_stdパラメータを持つ
    if isinstance(model, (RSMPlusGP_Production, RSMPlusGP_ARD)):
        y_pred = model.predict(X_test, return_std=False)
    else:
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # RSM単独との比較
    if isinstance(model, (RSMPlusGP_Production, RSMPlusGP_ARD)):
        # RSM部分のみを計算（GPなし）
        Z = model.x_scaler.transform(X_test)
        Zpoly_all = model.poly.transform(Z)
        Zpoly = Zpoly_all[:, model._rsm_selected_cols]
        t_rsm = model.lin.predict(Zpoly)
        y_rsm = model.y_scaler.inverse_transform(t_rsm.reshape(-1, 1)).ravel()
    elif hasattr(model, 'rsm_model'):
        # Additive GPの場合
        Z = model.rsm_model.x_scaler.transform(X_test)
        Zpoly_all = model.rsm_model.poly.transform(Z)
        Zpoly = Zpoly_all[:, model.rsm_model._rsm_selected_cols]
        t_rsm = model.rsm_model.lin.predict(Zpoly)
        y_rsm = model.rsm_model.y_scaler.inverse_transform(t_rsm.reshape(-1, 1)).ravel()
    else:
        y_rsm = None

    if y_rsm is not None:

        r2_rsm = r2_score(y_test, y_rsm)
        improvement = (r2 - r2_rsm) / (1 - r2_rsm) * 100 if r2_rsm < 1 else 0

        # GP寄与度計算
        gp_correction = y_pred - y_rsm
        rsm_var = np.var(y_rsm - np.mean(y_test))
        gp_var = np.var(gp_correction)
        total_var = rsm_var + gp_var

        gp_contrib = (gp_var / total_var * 100) if total_var > 1e-12 else 0
    else:
        r2_rsm = None
        improvement = None
        gp_contrib = None

    return {
        'model': model_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'r2_rsm': r2_rsm,
        'improvement': improvement,
        'gp_contrib': gp_contrib
    }


def print_results(results):
    """結果を整形して表示"""
    print(f"\n{'='*80}")
    print(f"{results['model']}")
    print(f"{'='*80}")
    print(f"R²:              {results['r2']:.4f}")
    print(f"RMSE:            {results['rmse']:.4f}")
    print(f"MAE:             {results['mae']:.4f}")

    if results['r2_rsm'] is not None:
        print(f"\nRSM単独のR²:     {results['r2_rsm']:.4f}")
        print(f"GP改善率:        {results['improvement']:.2f}%")
        print(f"GP寄与度:        {results['gp_contrib']:.2f}%")

        # 検知判定
        if results['gp_contrib'] > 30 or results['improvement'] > 15:
            detection = "強い"
            icon = "⚠"
        elif results['gp_contrib'] > 15 or results['improvement'] > 8:
            detection = "中程度"
            icon = "○"
        elif results['gp_contrib'] > 5 or results['improvement'] > 3:
            detection = "弱い"
            icon = "△"
        else:
            detection = "なし"
            icon = "✓"

        print(f"高次項検知:      {icon} {detection}")


def main():
    print("="*80)
    print("次元の呪いを克服する革新的アプローチ")
    print("="*80)
    print("\n【問題設定】")
    print("- 10次元データ、n=300（n/d=30）")
    print("- 従来の標準GPでは失敗（GP寄与0.2%）")
    print("- 最初の2次元のみに非線形性（sin/cos）")
    print("\n【3つのアプローチを比較】")
    print("1. 標準GP（ベースライン）- すべての次元を等しく扱う")
    print("2. ARD GP - 各次元の重要度を自動学習")
    print("3. Additive GP - 各次元を独立に処理")

    # データ生成
    print("\n" + "="*80)
    print("データ生成")
    print("="*80)
    X_train, y_train = generate_10d_data(300, random_state=42)
    X_test, y_test = generate_10d_data(100, random_state=123)

    feature_names = [f'x{i+1}' for i in range(10)]

    print(f"訓練データ: n={X_train.shape[0]}, d={X_train.shape[1]}")
    print(f"テストデータ: n={X_test.shape[0]}, d={X_test.shape[1]}")
    print(f"n/d比: {X_train.shape[0]/X_train.shape[1]:.1f}")

    results_all = []

    # ========================================
    # 方法1: 標準GP（ベースライン）
    # ========================================
    print("\n" + "="*80)
    print("方法1: 標準GP（ベースライン）")
    print("="*80)
    print("すべての次元を等しく扱う（従来の方法）")

    model_standard = RSMPlusGP_Production(
        F_enter=2.0,
        F_remove=2.0,
        random_state=42
    )
    model_standard.fit(X_train, y_train, feature_names=feature_names)

    results_standard = evaluate_model(model_standard, X_test, y_test, "標準GP")
    print_results(results_standard)
    results_all.append(results_standard)

    # ========================================
    # 方法2: ARD GP
    # ========================================
    print("\n" + "="*80)
    print("方法2: ARD (Automatic Relevance Determination) GP")
    print("="*80)
    print("各次元の重要度を自動学習 → 重要な次元に集中")

    model_ard = RSMPlusGP_ARD(
        n_features=X_train.shape[1],  # 10次元
        F_enter=2.0,
        F_remove=2.0,
        random_state=42
    )
    model_ard.fit(X_train, y_train, feature_names=feature_names)

    results_ard = evaluate_model(model_ard, X_test, y_test, "ARD GP")
    print_results(results_ard)
    results_all.append(results_ard)

    # ARD length scales を表示
    if hasattr(model_ard, 'gp') and hasattr(model_ard.gp, 'kernel_'):
        kernel = model_ard.gp.kernel_
        # Extract length scales
        try:
            length_scales = kernel.k1.k2.length_scale
            print(f"\n【ARD Length Scales】各次元の重要度（小さいほど重要）")
            for i, ls in enumerate(length_scales):
                importance = 1.0 / ls
                bar = "█" * int(importance * 10)
                print(f"  x{i+1}: {ls:.3f}  {bar}")
        except Exception as e:
            print(f"\n【ARD Length Scales】取得できませんでした: {e}")

    # ========================================
    # 方法3: Additive GP
    # ========================================
    print("\n" + "="*80)
    print("方法3: Additive GP")
    print("="*80)
    print("各次元を独立に処理して合計 → 次元の呪いを回避")

    # まずRSMのみのモデルを作成（Additive GPのベース）
    model_rsm_only = RSMPlusGP_Production(
        F_enter=2.0,
        F_remove=2.0,
        random_state=42
    )
    # RSMのみを学習（GPなし）
    model_rsm_only.fit(X_train, y_train, feature_names=feature_names)

    # Additive GP を構築
    model_additive = AdditiveGP(model_rsm_only, random_state=42)
    model_additive.fit(X_train, y_train)

    results_additive = evaluate_model(model_additive, X_test, y_test, "Additive GP")
    print_results(results_additive)
    results_all.append(results_additive)

    # ========================================
    # 総合比較
    # ========================================
    print("\n" + "="*80)
    print("総合比較: 次元の呪いを克服できたか？")
    print("="*80)

    print(f"\n{'方法':<20} {'R²':>8} {'GP寄与':>10} {'GP改善':>10} {'検知':>10}")
    print("-" * 60)

    for res in results_all:
        detection = ""
        if res['gp_contrib'] is not None:
            if res['gp_contrib'] > 30 or res['improvement'] > 15:
                detection = "⚠ 強い"
            elif res['gp_contrib'] > 15 or res['improvement'] > 8:
                detection = "○ 中程度"
            elif res['gp_contrib'] > 5 or res['improvement'] > 3:
                detection = "△ 弱い"
            else:
                detection = "✓ なし"

        gp_contrib_str = f"{res['gp_contrib']:.1f}%" if res['gp_contrib'] is not None else "N/A"
        improvement_str = f"{res['improvement']:.1f}%" if res['improvement'] is not None else "N/A"

        print(f"{res['model']:<20} {res['r2']:>8.4f} {gp_contrib_str:>10} {improvement_str:>10} {detection:>10}")

    print("\n" + "="*80)
    print("結論")
    print("="*80)

    # 最良の方法を特定（R²で比較）
    best_idx = max(range(len(results_all)),
                   key=lambda i: results_all[i]['r2'])
    best = results_all[best_idx]

    print(f"\n最良の方法: {best['model']}")
    print(f"  R²:       {best['r2']:.4f}")
    if best['gp_contrib'] is not None:
        print(f"  GP寄与度: {best['gp_contrib']:.1f}%")
        print(f"  GP改善率: {best['improvement']:.1f}%")

    # 標準GPとの比較
    standard_r2 = results_all[0]['r2']
    standard_contrib = results_all[0]['gp_contrib']
    if standard_contrib is not None and best['gp_contrib'] is not None:
        if standard_contrib > 0 and best['gp_contrib'] > standard_contrib * 2:
            improvement_ratio = best['gp_contrib'] / standard_contrib
            print(f"\n✨ 標準GPと比較して {improvement_ratio:.1f}x の改善！")
            print(f"   次元の呪いを克服しました！")
        elif standard_contrib <= 1e-6 and best['gp_contrib'] > 5:
            print(f"\n✨ 標準GPが完全に失敗したケースで、{best['gp_contrib']:.1f}% のGP寄与を達成！")
            print(f"   次元の呪いを克服しました！")
    elif best['r2'] > standard_r2 * 1.1:
        r2_improvement = (best['r2'] - standard_r2) / standard_r2 * 100
        print(f"\n✓ R²が {r2_improvement:.1f}% 改善しました")
        print(f"  次元の呪いを部分的に緩和")
    elif best['gp_contrib'] is not None and best['gp_contrib'] > 5:
        print(f"\n✓ 一定の改善が見られました")
        print(f"  次元の呪いを部分的に緩和")
    else:
        print(f"\n✗ 次元の呪いは依然として課題")
        print(f"  さらなる工夫が必要")

    print("\n" + "="*80)
    print("技術的洞察")
    print("="*80)
    print("""
1. ARD (Automatic Relevance Determination):
   - 各次元に独立した length_scale を学習
   - 重要な次元（x1, x2）には小さい length_scale
   - 重要でない次元には大きい length_scale
   → 実質的に重要な次元のみを使用

2. Additive GP:
   - f(x₁,...,xₐ) = f₁(x₁) + ... + fₐ(xₐ)
   - 各GPは1次元のみを処理
   - 次元の呪いを構造的に回避
   → 高次元でも安定

3. 次元の呪いの本質:
   - 高次元空間はスパース（疎）
   - データ点間の距離が均一化
   - GPのカーネルが機能しない
   → 構造を仮定することで回避可能
""")


if __name__ == '__main__':
    main()
