# 診断ツール（model.diagnose()）の使い方

## 概要

`model.diagnose()` メソッドは、RSM+GPモデルの性能を包括的に診断し、改善のための推奨事項を提供する機能です。

## 主な機能

### 1. 基本統計の表示
- 訓練データ数（n）と特徴量数（d）
- 選択された項数 vs 全候補項数

### 2. 予測性能の評価
- **R²スコア**: モデルの説明力
- **RMSE**: 二乗平均平方根誤差
- **MAE**: 平均絶対誤差

### 3. RSM vs GP 寄与度分析
- **RSM寄与度**: 線形・二次構造の寄与（%）
- **GP寄与度**: 残差・高次項補正の寄与（%）
- **GP改善率**: GPによる性能向上（%）

### 4. カーネルパラメータの検証
- **length_scale**: RBFカーネルの長さスケール
- **constant_value**: 定数カーネルの値
- **noise_level**: ノイズレベル
- **評価**: パラメータの妥当性判定

### 5. 推奨事項の自動生成
- データ量による推奨
- GP改善率に基づく提案
- カーネルパラメータの警告

---

## 使い方

### 基本的な使用法

```python
import numpy as np
import rsm
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# データ準備
X_train = np.random.uniform(-1, 1, size=(100, 3))
y_train = some_function(X_train)
X_test = np.random.uniform(-1, 1, size=(50, 3))
y_test = some_function(X_test)

# モデル訓練
kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))
model = rsm.RSMPlusGP_Production(
    gp_kernel=kernel,
    gp_n_restarts_optimizer=5,
    random_state=42
)
model.fit(X_train, y_train)

# 診断実行（レポート表示）
info = model.diagnose(X_test, y_test, display=True)
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|----------|-----|----------|------|
| `X_test` | array-like, optional | `None` | テストデータの入力。Noneの場合は訓練データを使用 |
| `y_test_true` | array-like, optional | `None` | テストデータの真値。Noneの場合は訓練データを使用 |
| `display` | bool | `True` | 診断レポートを表示するかどうか |

### 返り値

辞書型で以下の情報を含む：

```python
{
    'n_train': 訓練データ数,
    'n_features': 特徴量数,
    'n_selected_terms': 選択された項数,
    'n_total_terms': 全候補項数,
    'r2': RSM+GPのR²スコア,
    'rmse': RMSE,
    'mae': MAE,
    'rsm_only_r2': RSMのみのR²スコア,
    'improvement': GP改善率 (%),
    'rsm_contribution': RSM寄与度 (%),
    'gp_contribution': GP寄与度 (%),
    'length_scale': カーネルのlength_scale,
    'constant_value': カーネルの定数値,
    'noise_level': ノイズレベル,
    'kernel_status': カーネルパラメータの評価,
    'recommendations': 推奨事項のリスト
}
```

---

## 出力例

### 少数データ（n=30）の場合

```
================================================================================
モデル診断レポート
================================================================================

【基本統計】
  訓練データ: n=30, d=2
  選択項数: 6/6

【予測性能】
  R²:   0.9887
  RMSE: 0.2474
  MAE:  0.1843

【RSM vs GP 寄与度】
  RSM寄与: 99.5% (主に線形・二次構造)
  GP寄与:  0.5% (残差・高次項補正)
  RSMのみのR²: 0.9871
  GP改善:      0.16%
  → RSM主導型（二次多項式で十分説明）✓

【カーネルパラメータ】
  length_scale:   0.1620
  constant_value: 0.0174
  noise_level:    0.0009
  評価: 適切 ✓

【推奨事項】
  ✓ モデルは良好に機能しています
    さらなる精度向上にはデータ追加推奨（n→100-150）
    GPの改善が小さいです（0.2%）
    → データを増やすとGPが高次項を捕捉（n≥100推奨）

================================================================================
```

### 大規模データ（n=200）の場合

```
================================================================================
モデル診断レポート
================================================================================

【基本統計】
  訓練データ: n=200, d=2
  選択項数: 6/6

【予測性能】
  R²:   0.9987
  RMSE: 0.0799
  MAE:  0.0580

【RSM vs GP 寄与度】
  RSM寄与: 99.1% (主に線形・二次構造)
  GP寄与:  0.9% (残差・高次項補正)
  RSMのみのR²: 0.9886
  GP改善:      1.01%
  → RSM主導型（二次多項式で十分説明）✓

【カーネルパラメータ】
  length_scale:   1.2181
  constant_value: 0.0781
  noise_level:    0.0111
  評価: 適切 ✓

【推奨事項】
  ✓ モデルは非常に良好に機能しています
    GPの改善が小さいです（1.0%）

================================================================================
```

---

## 使用シーン

### 1. モデルの妥当性確認

```python
# モデル訓練後、すぐに診断
model.fit(X_train, y_train)
info = model.diagnose(X_test, y_test)

# R²が低い場合は警告が表示される
if info['r2'] < 0.95:
    print("注意: モデルの精度が低いです")
```

### 2. データ量の適切性評価

```python
# GPの改善が小さい場合、データ追加を推奨
if info['improvement'] < 5:
    print(f"GP改善率が低い（{info['improvement']:.1f}%）")
    print("→ データを n≥100 に増やすことを推奨")
```

### 3. RSM vs GP の寄与度分析

```python
# どちらが支配的かを確認
if info['rsm_contribution'] > 80:
    print("RSM主導型: 二次多項式で十分説明")
elif info['gp_contribution'] > 50:
    print("GP主導型: 高次・非線形性が支配的")
    print("→ データを増やすとさらに精度向上")
else:
    print("バランス良好")
```

### 4. カーネルパラメータの診断

```python
# カーネルパラメータの妥当性確認
if info['kernel_status'] != "適切 ✓":
    print(f"警告: {info['kernel_status']}")
    print("カーネルパラメータ範囲を調整してください")
```

### 5. プログラムによる自動判定

```python
# display=Falseで辞書のみ取得
info = model.diagnose(X_test, y_test, display=False)

# 自動判定ロジック
if info['r2'] > 0.99:
    status = "非常に良好"
elif info['r2'] > 0.95:
    status = "良好"
elif info['r2'] > 0.90:
    status = "実用レベル"
else:
    status = "改善必要"

print(f"モデル状態: {status} (R²={info['r2']:.4f})")
```

---

## 推奨事項の解釈

### R²スコアによる評価

| R² | 評価 | 推奨アクション |
|----|------|--------------|
| > 0.99 | 非常に良好 | そのまま使用可能 |
| 0.95 - 0.99 | 良好 | n<100ならデータ追加で精度向上可能 |
| 0.90 - 0.95 | 実用レベル | データ数を増やすことを推奨 |
| < 0.90 | 改善必要 | データを大幅に増やす |

### GP改善率による評価

| GP改善率 | 解釈 | 推奨アクション |
|---------|------|--------------|
| < 5% | GPの効果が限定的 | データを n≥100 に増やす |
| 5-20% | GPが一定の効果 | 良好 |
| 20-50% | GPが効果的 | 非常に良好 |
| > 50% | GPが非常に効果的 | 高次項を十分捕捉 |

### カーネルパラメータの評価

| length_scale | 評価 | 推奨アクション |
|-------------|------|--------------|
| < 0.05 | 短すぎ | 過学習の可能性、データを増やす |
| 0.05 - 10.0 | 適切 ✓ | 問題なし |
| > 10.0 | 長すぎ | 過平滑化の可能性、範囲を調整 |

---

## ベストプラクティス

### 1. モデル訓練直後に診断

```python
model.fit(X_train, y_train)
info = model.diagnose(X_test, y_test)  # すぐに診断
```

### 2. データ量を変えて比較

```python
for n in [30, 50, 100, 150]:
    model.fit(X_train[:n], y_train[:n])
    info = model.diagnose(X_test, y_test, display=False)
    print(f"n={n}: R²={info['r2']:.4f}, GP改善={info['improvement']:.1f}%")
```

### 3. カーネル設定の比較

```python
kernels = [
    ("デフォルト", C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))),
    ("広範囲", C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-6, 1e0))),
]

for name, kernel in kernels:
    model = rsm.RSMPlusGP_Production(gp_kernel=kernel)
    model.fit(X_train, y_train)
    info = model.diagnose(X_test, y_test, display=False)
    print(f"{name}: R²={info['r2']:.4f}")
```

### 4. 訓練データでの診断（過学習チェック）

```python
# テストデータ
info_test = model.diagnose(X_test, y_test, display=False)
# 訓練データ
info_train = model.diagnose(display=False)  # X_test=None, y_test_true=None

if info_train['r2'] - info_test['r2'] > 0.05:
    print("警告: 過学習の可能性（訓練データとテストデータのR²差が大きい）")
```

---

## まとめ

### ✅ 診断ツールでできること

1. **性能の即座な確認**: R², RMSE, MAEを一目で把握
2. **RSM vs GPの寄与度分析**: どちらが主導的かを理解
3. **カーネルパラメータの検証**: 過学習・過平滑化のチェック
4. **自動推奨事項**: データ量、パラメータ調整の提案
5. **プログラム連携**: 辞書形式で自動判定可能

### 🎯 主な用途

- モデルの妥当性確認
- データ量の適切性評価
- ハイパーパラメータ調整の指針
- 実験計画の最適化
- レポート生成

### 💡 推奨ワークフロー

```python
# 1. モデル訓練
model.fit(X_train, y_train)

# 2. 診断実行
info = model.diagnose(X_test, y_test)

# 3. 推奨事項に従って改善
if info['improvement'] < 5 and info['n_train'] < 100:
    # データを増やして再訓練
    X_train_more, y_train_more = collect_more_data()
    model.fit(X_train_more, y_train_more)
    info = model.diagnose(X_test, y_test)

# 4. 最終確認
if info['r2'] > 0.95:
    print("モデルは本番使用可能です")
```

---

**実装日**: 2026-01-11
**バージョン**: RSM+GP最適化版
**ドキュメント**: DIAGNOSE_TOOL.md
