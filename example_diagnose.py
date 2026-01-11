"""
診断ツール（model.diagnose()）の実装例
ユーザーがモデルの妥当性を簡単に確認できる機能
"""
import numpy as np
from typing import Dict, Any, Optional

class DiagnosticReport:
    """診断レポートのクラス"""

    def __init__(self, model, X_test=None, y_test_true=None):
        """
        診断を実行

        Parameters:
        -----------
        model : RSMPlusGP_Production
            学習済みモデル
        X_test : np.ndarray, optional
            テストデータ（ない場合は訓練データで評価）
        y_test_true : np.ndarray, optional
            テストデータの真値
        """
        self.model = model
        self.X_test = X_test
        self.y_test_true = y_test_true

        # 診断情報を収集
        self.info = self._collect_diagnostics()

    def _collect_diagnostics(self) -> Dict[str, Any]:
        """診断情報を収集"""
        info = {}

        # 1. 基本情報
        info['n_train'] = self.model._X_train.shape[0]
        info['n_features'] = self.model._X_train.shape[1]
        info['n_selected_terms'] = len(self.model._rsm_selected_cols)
        info['n_total_terms'] = len(self.model.poly.powers_)

        # 2. RSM vs GP 寄与度
        if self.X_test is not None and self.y_test_true is not None:
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
            info['r2'] = 1 - np.sum((self.y_test_true - y_full)**2) / np.sum((self.y_test_true - np.mean(self.y_test_true))**2)
            info['rmse'] = np.sqrt(np.mean((self.y_test_true - y_full)**2))
            info['mae'] = np.mean(np.abs(self.y_test_true - y_full))

            # RSMのみとの比較
            info['rsm_only_r2'] = 1 - np.sum((self.y_test_true - y_rsm_only)**2) / np.sum((self.y_test_true - np.mean(self.y_test_true))**2)
            info['improvement'] = (info['r2'] - info['rsm_only_r2']) * 100

        # 3. カーネルパラメータ
        kernel_params = self.model.gp.kernel_
        info['length_scale'] = kernel_params.k1.k2.length_scale
        info['constant_value'] = kernel_params.k1.k1.constant_value
        info['noise_level'] = kernel_params.k2.noise_level

        # 4. カーネルパラメータの妥当性評価
        info['kernel_status'] = self._evaluate_kernel_params(info)

        return info

    def _evaluate_kernel_params(self, info: Dict) -> str:
        """カーネルパラメータの妥当性を評価"""
        ls = info['length_scale']

        # 長さスケールのチェック
        if ls < 0.05:
            return "短すぎ（過学習の可能性）"
        elif ls > 10.0:
            return "長すぎ（過平滑化の可能性）"
        else:
            return "適切"

    def _generate_recommendations(self, info: Dict) -> list:
        """推奨事項を生成"""
        recommendations = []

        # R²ベースの推奨
        if self.y_test_true is not None:
            if info['r2'] > 0.99:
                recommendations.append("✓ モデルは非常に良好に機能しています")
            elif info['r2'] > 0.95:
                recommendations.append("✓ モデルは良好に機能しています")
                if info['n_train'] < 100:
                    recommendations.append("- さらなる精度向上にはデータ追加推奨（n→100-150）")
            elif info['r2'] > 0.90:
                recommendations.append("⚠ モデルの精度は実用レベルですが改善余地があります")
                recommendations.append("- データ数を増やすことを推奨（現在n={}）".format(info['n_train']))
                recommendations.append("- gp_n_restarts_optimizerを増やす（3→10）")
            else:
                recommendations.append("⚠ モデルの精度が低いです")
                recommendations.append("- データ数を大幅に増やしてください")
                recommendations.append("- データの品質を確認してください（外れ値、ノイズ）")

        # GP改善度ベースの推奨
        if self.y_test_true is not None and info['improvement'] < 5:
            recommendations.append("- GPの改善が小さいです（{}%）".format(round(info['improvement'], 1)))
            if info['n_train'] < 100:
                recommendations.append("  → データを増やすとGPが高次項を捕捉しやすくなります（n≥100推奨）")

        # カーネルパラメータの推奨
        if info['kernel_status'] != "適切":
            recommendations.append("⚠ カーネルパラメータ: {}".format(info['kernel_status']))
            recommendations.append("  → カーネルの範囲を調整することを検討してください")

        # 変数選択の推奨
        selection_rate = info['n_selected_terms'] / info['n_total_terms']
        if selection_rate < 0.3:
            recommendations.append("- 選択項が少なめです（{}/{}）".format(info['n_selected_terms'], info['n_total_terms']))
            recommendations.append("  → F_enterを下げると項が増える可能性があります")
        elif selection_rate > 0.8:
            recommendations.append("- 選択項が多めです（{}/{}）".format(info['n_selected_terms'], info['n_total_terms']))
            recommendations.append("  → F_enterを上げると過学習を防げる可能性があります")

        if not recommendations:
            recommendations.append("✓ 特に問題はありません")

        return recommendations

    def display(self):
        """診断レポートを表示"""
        print("=" * 80)
        print("モデル診断レポート")
        print("=" * 80)

        # 基本統計
        print("\n【基本統計】")
        print(f"  訓練データ: n={self.info['n_train']}, d={self.info['n_features']}")
        print(f"  選択項数: {self.info['n_selected_terms']}/{self.info['n_total_terms']}")

        # 予測性能
        if self.y_test_true is not None:
            print("\n【予測性能】")
            print(f"  R²:   {self.info['r2']:.4f}")
            print(f"  RMSE: {self.info['rmse']:.4f}")
            print(f"  MAE:  {self.info['mae']:.4f}")

            # RSM vs GP
            print("\n【RSM vs GP 寄与度】")
            print(f"  RSM寄与: {self.info['rsm_contribution']:.1f}% (主に線形・二次構造)")
            print(f"  GP寄与:  {self.info['gp_contribution']:.1f}% (残差・高次項補正)")

            # GP改善
            print(f"\n  RSMのみのR²: {self.info['rsm_only_r2']:.4f}")
            print(f"  GP改善:      {self.info['improvement']:.2f}%")

            # 評価
            if self.info['rsm_contribution'] > 80:
                print("  → RSM主導型（二次多項式で十分説明）")
            elif self.info['gp_contribution'] > 50:
                print("  → GP主導型（高次・非線形性が支配的）")
            else:
                print("  → バランス良好 ✓")

        # カーネルパラメータ
        print("\n【カーネルパラメータ】")
        print(f"  length_scale:   {self.info['length_scale']:.4f}")
        print(f"  constant_value: {self.info['constant_value']:.4f}")
        print(f"  noise_level:    {self.info['noise_level']:.4f}")
        print(f"  評価: {self.info['kernel_status']}")

        # 推奨事項
        recommendations = self._generate_recommendations(self.info)
        print("\n【推奨事項】")
        for rec in recommendations:
            print(f"  {rec}")

        print("\n" + "=" * 80)

    def get_dict(self) -> Dict[str, Any]:
        """診断情報を辞書で返す"""
        return self.info


# RSMPlusGP_Productionクラスに追加するメソッド
def diagnose(self, X_test=None, y_test_true=None):
    """
    モデルの診断を実行

    Parameters:
    -----------
    X_test : np.ndarray, optional
        テストデータ（ない場合は訓練データで評価）
    y_test_true : np.ndarray, optional
        テストデータの真値

    Returns:
    --------
    DiagnosticReport
        診断レポートオブジェクト

    Examples:
    ---------
    >>> model.fit(X_train, y_train)
    >>> diagnosis = model.diagnose(X_test, y_test_true)
    >>> diagnosis.display()  # レポートを表示
    """
    self._check_fitted()

    if X_test is None:
        X_test = self._X_train
        y_test_true = self._y_train

    return DiagnosticReport(self, X_test, y_test_true)


# 使用例のデモ
if __name__ == "__main__":
    print("=" * 80)
    print("診断ツール（model.diagnose()）の実装例")
    print("=" * 80)

    print("\n【使用方法】")
    print("""
# モデルを学習
model = rsm.RSMPlusGP_Production(...)
model.fit(X_train, y_train)

# 診断を実行（テストデータがある場合）
diagnosis = model.diagnose(X_test, y_test_true)
diagnosis.display()

# または訓練データで診断（テストデータがない場合）
diagnosis = model.diagnose()
diagnosis.display()

# 診断情報を辞書で取得
info = diagnosis.get_dict()
print(info['r2'])  # R²を取得
print(info['rsm_contribution'])  # RSM寄与度を取得
""")

    print("\n【出力される診断情報】")
    print("1. 基本統計")
    print("   - 訓練データサイズ（n, d）")
    print("   - 選択項数（ステップワイズ結果）")

    print("\n2. 予測性能")
    print("   - R², RMSE, MAE")
    print("   - RSMのみとの比較")

    print("\n3. RSM vs GP 寄与度")
    print("   - RSMの寄与率（線形・二次構造）")
    print("   - GPの寄与率（高次項・非線形補正）")
    print("   - バランス評価")

    print("\n4. カーネルパラメータ")
    print("   - length_scale, constant_value, noise_level")
    print("   - 妥当性評価（適切/短すぎ/長すぎ）")

    print("\n5. 推奨事項")
    print("   - データ追加の推奨")
    print("   - パラメータ調整の提案")
    print("   - 潜在的な問題の警告")

    print("\n【メリット】")
    print("✓ 初心者でもモデルの状態を簡単に理解")
    print("✓ RSMとGPのどちらが主に寄与しているか一目瞭然")
    print("✓ カーネルパラメータの妥当性を自動チェック")
    print("✓ 次に何をすべきか推奨事項を提示")
    print("✓ 問題の早期発見（過学習、データ不足など）")

    print("\n" + "=" * 80)
