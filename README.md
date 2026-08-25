# openclaw-peptide-admet

**peptide ADMET 預測管線(v4.0 真實數據版,2026-08-25)**

本 repo 是一個**可重現、誠實標註**的 peptide ADMET 預測管線。v4.0 起改用
**真實實驗數據集** [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset),
預測 **4 個端點**:**Hemolysis(溶血)**、**Half-life(半衰期)**、**Caco-2 透性**、
**PAMPA/MDCK 透性**。

> ⚠️ **v4.0 與 v3.0 的斷裂式變更**:v3.0 的 30,000 行 `synthetic_demo` 合成數據
> 與 9 端點(含 BBB/Ames/hERG/毒性/HC50)管線**已刪除**。v4.0 只保留
> 4 個有真實數據的端點,特徵改為**雙模態**(序列端點用氨基酸序列,
> 分子端點用 RDKit SMILES 描述子)。舊的 `prepare_data.py`、
> `train_peptide_admet_model.py`、`peptide_admet_model/` 權重、合成數據
> 皆已從 repo 移除。

---

## 為什麼從合成數據改為真實數據

v3.0 的 30,000 行 `synthetic_demo` 是用 Dirichlet 家族結構 + 潛在理化分數
**生成**的示範數據,標籤非實驗量測。它適合驗證管線端對端可跑通,但
**不能**宣稱任何「真實肽類性能」。v4.0 改用公開真實數據集,4 個端點的
標籤皆來自實驗/文獻,指標才有意義。

---

## 4 個端點與雙模態

| 端點 | 類型 | 模態 | 特徵維度 | 訓練行數 | 測試集 | 主指標(實測) |
|------|------|------|---------|---------|--------|--------------|
| **Hemolysis** | 二分類 | 序列 | 428 | 8,719 | 1,745 | AUC **0.7755** |
| **Half_life** | 回歸(log10) | 序列 | 428 | 1,763 | 428 | R² **0.5883** |
| **Caco2** | 回歸(logPapp) | 分子 | 2,265 | 7,429 | 1,490 | R² **0.3861** |
| **PAMPA_MDCK** | 回歸(logPapp) | 分子 | 2,265 | 7,283 | 1,457 | R² **0.4573** |

**兩種模態**(這是 v4.0 的核心架構決策):
- **序列模態**(Hemolysis、Half_life):4 個數據集中有乾淨的 20 標準氨基酸
  one-letter 序列 → 用 428 維序列特徵(AAC 20 + DPC 400 + 理化 8)。
- **分子模態**(Caco2、PAMPA_MDCK):這兩個數據集的「序列」是
  CycPeptMPDB 的**非標準殘基名稱清單**(MEL、DP、DL、ME_DL…),
  20-氨基酸編碼不了;但都有**有效 SMILES** → 用 2,265 維 RDKit 分子
  特徵(217 個 2D 描述子 + 2,048 位 Morgan fingerprint,半徑 2)。

4 個端點是**互斥的分子**(不共享同一批化合物),且分屬兩種模態,
因此採用 **4 個獨立單任務模型**(各自匹配模態),而非一個共享 trunk
的多頭模型(後者對互斥+零填充數據只是學到兩個不相交的子空間)。
`MixedADMETMLP` 本身是通用的(由 `input_dim` + `endpoints` 參數化),
每個端點建一個單頭模型。

---

## 📁 專案文件

### 管線腳本

| 文件 | 說明 |
|------|------|
| `endpoint_config.py` | **4 端點的單一事實來源**(kind、模態、來源 CSV、特徵欄、標籤轉換、單位) |
| `feature_extractor.py` | 雙模態特徵:序列 428 維(AAC/DPC/理化)+ RDKit 分子 2,265 維(2D 描述子 + Morgan) |
| `prepare_pepadmet_data.py` | 載入 4 個真實 CSV → 清洗(去 X/非 20-AA/無效 SMILES/缺失標籤)→ 每端點輸出準備好的 CSV + meta |
| `homology_split.py` | 3-mer Jaccard 家族 70/10/20 同源性不相交分割 + 量測洩漏(最大 pairwise Jaccard、標籤率差) |
| `admet_model.py` | `MixedADMETMLP`(binary/regression heads,通用 input_dim)+ 訓練/預測共用,架構永不漂移 |
| `train_pepadmet_model.py` | 每端點:特徵 → (同源性分割/分子唯一 SMILES 分割)→ 標準化 → 訓練 → 雙分割評估 → `metrics.json` + 權重 |
| `peptide_admet_predictor.py` | 推論 CLI:輸入 sequence / SMILES → 自動路由到對應模態端點 → 4 端點預測 + 單位 |

### 訓練產物(`models_v4/`,已 commit)

| 路徑 | 說明 |
|------|------|
| `models_v4/<endpoint>/admet_mlp.pt` | PyTorch 權重 + 架構 metadata(`model_version: v4_endpoint`) |
| `models_v4/<endpoint>/scaler.pt` | StandardScaler |
| `models_v4/<endpoint>/metrics.json` | 實測指標(同源性/隨機雙分割)+ 分割統計 + 洩漏稽核 |
| `models_v4/summary.json` | 4 端點彙總 |

`<endpoint>` ∈ {`hemolysis`, `half_life`, `caco2`, `pampa_mdck`}。
4 個模型權重共 ~6 MB,已 commit 到 repo(可直用,無需重訓)。

### 數據(`data/`,已 commit)

| 文件 | 說明 |
|------|------|
| `data/pepadmet_hemolysis.csv` | 準備好的 Hemolysis 序列 + 二值標籤 |
| `data/pepadmet_half_life.csv` | 準備好的 Half_life 序列 + log10 半衰期 |
| `data/pepadmet_caco2.csv` | 準備好的 Caco2 SMILES + logPapp |
| `data/pepadmet_pampa_mdck.csv` | 準備好的 PAMPA SMILES + logPapp |
| `data/pepadmet_data.meta.json` | 來源、清洗統計、行數聲明 |

這些 CSV 是**確定性的準備輸出**(由 `prepare_pepadmet_data.py` 從原始
PepADMET-Dataset 產生),已 commit 以讓整個管線可從 repo 重現。
原始未清洗 CSV 不進 repo(體積大、且可由公開 repo 重現)。

---

## 🚀 如何使用

```bash
# 0. 環境(需 rdkit)
uv pip install --python .venv/Scripts/python.exe rdkit

# 1. 準備數據(從 Chemit797/PepADMET-Dataset 載入 + 清洗)
python prepare_pepadmet_data.py
#    → data/pepadmet_*.csv + data/pepadmet_data.meta.json

# 2. 訓練 4 端點(同源性/分子控制分割 + 雙分割評估)
python train_pepadmet_model.py --epochs 80 --seed 42
#    → models_v4/<endpoint>/{admet_mlp.pt, scaler.pt, metrics.json} + summary.json

# 3. 預測(自動路由模態:序列端點吃 sequence,分子端點吃 SMILES)
python peptide_admet_predictor.py \
  --sequence "ACDEFGHIKLMNPQRSTVWY" \
  --smiles "CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O"
```

---

## 📊 實測性能(同源性/分子控制測試分割)

| 端點 | 類型 | 主指標 | 其他(同源性測試) | 隨機分割對照 |
|------|------|--------|------------------|------------|
| Hemolysis | binary | AUC **0.7755** | MCC 0.3782, Acc 0.7009 | 0.7746(差 −0.0009) |
| Half_life | regression(log10 s) | R² **0.5883** | RMSE 1.2502, MAE 0.8714 | 0.8650(隨機被近重複洩漏推高) |
| Caco2 | regression(logPapp) | R² **0.3861** | RMSE 0.7879, MAE 0.4896 | —(無序列,無同源性對照) |
| PAMPA_MDCK | regression(logPapp) | R² **0.4573** | RMSE 0.8043, MAE 0.5070 | —(無序列,無同源性對照) |

### 洩漏控制(誠實說明)

- **序列端點**(Hemolysis、Half_life):用 3-mer Jaccard 家族做同源性
  控制分割。分割前先將**同 3-mer 多重集的 anagram 合併為同一家族**
  (canonical signature),**保證 jaccard-1.0 的精確複製不跨 train/test
  界線**。稽核量測的最大跨界 Jaccard ≈ 0.97(這是同源性控制下近重複
  的合理上限,非洩漏)。**Half_life 隨機分割 R² 0.865 遠高於同源性的
  0.5883,正是近重複洩漏被控管掉後誠實指標的體現**——這才是有意義的
  測試。
- **分子端點**(Caco2、PAMPA_MDCK):無序列,無法做 3-mer 同源性控制;
  改為按**唯一 SMILES 分組**(精確重複 SMILES 進同一分割)。但
  **近異構物**(不同 SMILES、相同化學)仍可能跨界——這是 SMILES-only
  真實數據的已知限制,比序列端的同源性控制弱,已如實標註。

> ⚠️ **這些是真實數據上的實測指標,不是合成示範。** R² 0.38–0.46 的
> 透性回歸是真實數據上合理(偏低但誠實)的結果;換更大/更乾淨的數據
> 會改變數字。

---

## 🎯 特徵工程

**序列模態(428 維)**:
1. 氨基酸組成 AAC — 20
2. 二肽組成 DPC — 400
3. 理化性質 — 8(MW 估、Kyte-Doolittle 疏水性、淨電荷、pI 估、GRAVY、
   疏水/帶電殘基比)

**分子模態(2,265 維)**:
1. RDKit 2D 描述子 — 217(`rdkit.Chem.Descriptors.CalcMolDescriptors`)
2. Morgan fingerprint(半徑 2)— 2,048 位

無效 SMILES(RDKit 解析失敗)→ 該行分子特徵全零並計入清洗統計,
如實標註(不造假)。

---

## 📦 模型架構

- 共享 trunk:`input_dim → 256 → 128`(ReLU + BatchNorm + Dropout 0.2)
- 單任務 head:
  - binary:`Linear(128,1)` + sigmoid
  - regression:`Linear(128,1)`
- 訓練:Adam(lr=3e-4)+ ReduceLROnPlateau + early stopping(val loss)
- 參數數:序列端點 **143,617**;分子端點 **613,889**

---

## 🔄 完整工作流程

```
Chemit797/PepADMET-Dataset (整理/*.csv, 真實數據)
       ↓
prepare_pepadmet_data.py (載入 + 清洗 + 來源標記 → data/pepadmet_*.csv)
       ↓
train_pepadmet_model.py (每端點:
   序列端點 → 428 維 + 同源性分割(3-mer signature 合併)
   分子端點 → 2,265 維 + 唯一 SMILES 分割
   → 標準化 → MixedADMETMLP 單頭 → 訓練 → 雙分割評估
   → models_v4/<endpoint>/{權重, metrics.json})
       ↓
peptide_admet_predictor.py (輸入 sequence/SMILES → 自動路由 → 4 端點預測 + 單位)
```

---

## ⚠️ 誠實聲明

- 4 端點來自**真實實驗/文獻數據**(PepADMET-Dataset),非合成。
- 分子端點無序列,同源性控制受限於唯一 SMILES 分組(近異構物可跨界,
  已標註)。
- 清洗會**丟棄**無效行(非 20-AA 序列、無效 SMILES、缺失標籤);
  丟棄統計寫入 `data/pepadmet_data.meta.json`,不隱瞞。
- 指標只在**同源性/分子控制測試分割**上報告為主指標;隨機分割僅作
  洩漏對照(Half_life 的 0.865 是「被洩漏推高」的反例,非主指標)。

---

## 版本

**v4.0**(真實數據 4 端點,雙模態)· **日期**:2026-08-25

**數據來源**:[Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset)
`整理/` 子目錄(hemolysis / half_life / caco2 / pampa_mdck)。
