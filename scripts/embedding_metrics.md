# Embedding 比較指標：LR accuracy、Procrustes、線性回歸 R² / CCA、CKA

用來比較 CAVACHON 的 embedding（`z_*`、`z_hat_*`、concat、joint model 的 z）。
三類指標回答的問題不同，對座標轉換的不變性也不同，選指標前先確認要比較的是
「資訊量」還是「幾何形狀」。

## 1. LR accuracy（logistic regression accuracy）

**問題：** embedding 能不能用線性分類器分出已知標籤（例如 cell type）？

**計算：** embedding 當特徵、標籤當目標，擬合 logistic regression，計算準確率。
`evaluate_embeddings.py` 目前的作法是 `StandardScaler` + `LogisticRegression`，
在同一份資料上擬合和預測，所以是**訓練準確率**：衡量線性可分程度，不是泛化能力
（`logreg_accuracy`、`logreg_balanced_accuracy`）。

```python
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
y_fit = clf.fit(embedding, y_true).predict(embedding)
accuracy_score(y_true, y_fit)
```

**不變性：**
- 平移、各軸縮放：不變（標準化處理掉）。
- 旋轉、一般可逆線性轉換：**近似不變**（線性可分性不變，但 L2 正則化會讓數值略有差異）。

**限制：** 需要標籤；只看和標籤有關的資訊，兩個 LR 都高的 embedding 其他結構可能完全不同。

## 2. Procrustes

**問題：** 兩個 embedding 能不能只用旋轉、平移、**等比例**縮放疊在一起？

**計算：** 同一批 cell 的 X、Y，找正交矩陣 R、縮放 s、平移 t 使 ‖Y − (sXR + t)‖ 最小，
回報殘差（disparity，越小越像）或換算成 R²。`scipy.spatial.procrustes`
會先把兩邊標準化，結果是對稱的。

```python
from scipy.spatial import procrustes
_, _, disparity = procrustes(X, Y)  # X, Y: (n_cells, n_dims)，維度須相同
```

**不變性：** 旋轉、平移、等比例縮放不變；**各軸不同比例縮放與 shear 會影響結果**。

**限制：** cell 須一一對應，維度通常須相同。

## 3. 線性回歸 R² / CCA

**問題：** 兩個 embedding 是否只差一個（任意）線性轉換？

**計算：**
- 線性回歸 R²：用 X 線性預測 Y（含截距），計算 R²。不對稱，兩個方向都算。
- CCA：找兩邊各自的線性投影使相關最大，回報各 canonical correlation（或平均）。

```python
from sklearn.linear_model import LinearRegression
r2_xy = LinearRegression().fit(X, Y).score(X, Y)  # X 能解釋 Y 的比例
r2_yx = LinearRegression().fit(Y, X).score(Y, X)
```

**不變性：** 對任意可逆線性（仿射）轉換不變；**維度可以不同**。

**用途：** 兩個 20 維 latent 本來就只能在線性轉換的意義下比較，這是比較
「是不是同一個表示」最公平的指標。

## 4. CKA（Centered Kernel Alignment；Kornblith et al., 2019）

**問題：** 兩個 embedding 對「哪些 cell 彼此相似」的整體看法是否一致？

**計算（linear CKA）：** X、Y 先置中（減去平均），

```
CKA(X, Y) = ‖Yᵀ X‖²_F / ( ‖Xᵀ X‖_F · ‖Yᵀ Y‖_F )
```

值介於 0–1，1 表示相似度結構完全一致。

```python
def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return np.linalg.norm(Y.T @ X) ** 2 / (
        np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    )
```

**不變性：** 正交轉換、等比例縮放不變；**各軸不同比例縮放會影響結果**（作者刻意設計：
各軸的變異大小也視為表示的一部分）。

**優點：** 不需要標籤，維度可以不同（例如 20 維對 60 維）。

## 比較表

| 指標 | 需要標籤 | 維度可不同 | 旋轉 | 各軸不同比例 / shear |
|---|---|---|---|---|
| LR accuracy | 要 | — | 近似不變 | 近似不變 |
| Procrustes | 不要 | 通常須相同 | 不變 | **會受影響** |
| 線性回歸 R² / CCA | 不要 | 可以 | 不變 | 不變 |
| linear CKA | 不要 | 可以 | 不變 | **會受影響** |

## 在 CAVACHON 中的用法

1. **ẑ 有沒有變好：** LR 比較 `z_hat_*`、`z_*`、concat(z) 與 joint z。
2. **`z_hat_Shared` 是否等同 joint 20 維 z：**
   - 線性回歸 R²（兩個方向）或 CCA：差一個線性轉換能否對上（資訊層面）。
   - CKA、Procrustes：幾何形狀（包含各軸變異分佈）是否也接近。
3. **殘差是否混入 parent 資訊：** CKA(z_ATAC_Only, z_hat_RNA_All)，或用
   z_ATAC_Only 線性回歸 z_hat_RNA_All 的 R²，都應該很低。
4. **判讀分解 vs 整合：** concat(z_ATAC_Only, z_RNA_Only, z_Shared) 是上限。
   上限高但 `z_hat_Shared` 低 → 整合 head 的問題；連上限都低 → 分解階段就遺失資訊。

**注意：** 根節點（沒有 parent）的 z_hat = A z + b，A 不一定正交，所以 Procrustes 與 CKA
算出的 `z_hat_RNA_All` 與 `z_RNA_All` 可能不同；這是座標系差異而非資訊量差異。
只關心資訊量時看 LR 或線性回歸 R²。

## 參考文獻

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network
  Representations Revisited. *ICML*.
- Gower, J. C. (1975). Generalized Procrustes analysis. *Psychometrika*, 40, 33–51.
- Hotelling, H. (1936). Relations between two sets of variates. *Biometrika*, 28, 321–377.
