# 肽類 ADMET 預測工具 — 完整總結(v4.0 真實數據版,2026-08-25)

> **v4.0 斷裂式變更**(取代 v3.0):
> 1. **改用真實數據集** [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset),
>    **刪除** v3.0 的 30,000 行 `synthetic_demo` 合成數據與 9 端點管線。
> 2. **4 端點**:Hemolysis、Half-life、Caco-2、PAMPA/MDCK(其餘 5 個
>    v3.0 端點 BBB/Ames/hERG/毒性/HC50 不再保留)。
> 3. **雙模態特徵**:序列端點用 428 維氨基酸序列特徵;分子端點用
>    2,265 維 RDKit SMILES 描述子(217 2D 描述子 + 2,048 位 Morgan)。
> 4. **4 個獨立單任務模型**(互斥分子 + 兩種模態,共享 trunk 多頭模型
>    在此只是學兩個不相交子空間)。
>
> v2.0/v3.0 的誠實修訂(移除硬編碼指標、phantom 數據、RF 偽裝 NN)全部保留。

## ✅ 完成狀態:100%(端對端已跑通驗證,4 端點全部訓練 + 預測驗證)

---

## 📁 專案文件

### 1. 管線腳本

| 文件 | 說明 |
|------|------|
| `endpoint_config.py` | 4 端點的單一事實來源(kind、模態、來源 CSV、特徵欄、標籤轉換、單位) |
| `feature_extractor.py` | 雙模態特徵:序列 428 維(AAC 20 + DPC 400 + 理化 8)+ RDKit 分子 2,265 維(217 2D 描述子 + 2,048 位 Morgan) |
| `prepare_pepadmet_data.py` | 載入 4 個真實 CSV → 清洗(去 X/非 20-AA/無效 SMILES/缺失標籤)→ 每端點輸出準備 CSV + meta |
| `homology_split.py` | 3-mer Jaccard 家族 70/10/20 同源性不相交分割 + 量測洩漏 |
| `admet_model.py` | `MixedADMETMLP`(binary/regression heads,通用 input_dim)+ 訓練/預測共用 |
| `train_pepadmet_model.py` | 每端點:特徵 → 分割 → 標準化 → 訓練 → 雙分割評估 → `metrics.json` + 權重 |
| `peptide_admet_predictor.py` | 推論 CLI:輸入 sequence / SMILES → 自動路由模態 → 4 端點預測 + 單位 |

### 2. 訓練產物(`models_v4/`,已 commit,~6 MB)

| 路徑 | 說明 |
|------|------|
| `models_v4/<endpoint>/admet_mlp.pt` | PyTorch 權重 + 架構 metadata(`model_version: v4_endpoint`) |
| `models_v4/<endpoint>/scaler.pt` | StandardScaler |
| `models_v4/<endpoint>/metrics.json` | 實測指標(雙分割)+ 分割統計 + 洩漏稽核 |
| `models_v4/summary.json` | 4 端點彙總 |

`<endpoint>` ∈ {`hemolysis`, `half_life`, `caco2`, `pampa_mdck`}。

### 3. 數據(`data/`,已 commit)

| 文件 | 說明 |
|------|------|
| `data/pepadmet_hemolysis.csv` | 準備好的 Hemolysis 序列 + 二值標籤(8,719 行) |
| `data/pepadmet_half_life.csv` | 準備好的 Half_life 序列 + log10 半衰期(1,763 行) |
| `data/pepadmet_caco2.csv` | 準備好的 Caco2 SMILES + logPapp(7,429 行) |
| `data/pepadmet_pampa_mdck.csv` | 準備好的 PAMPA SMILES + logPapp(7,283 行) |
| `data/pepadmet_data.meta.json` | 來源、清洗統計、行數聲明 |

---

## 🚀 如何使用

```bash
# 0. 環境(需 rdkit)
uv pip install --python .venv/Scripts/python.exe rdkit

# 1. 準備數據(從 Chemit797/PepADMET-Dataset 載入 + 清洗)
python prepare_pepadmet_data.py

# 2. 訓練 4 端點
python train_pepadmet_model.py --epochs 80 --seed 42

# 3. 預測(自動路由模態)
python peptide_admet_predictor.py \
  --sequence "ACDEFGHIKLMNPQRSTVWY" \
  --smiles "CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O"
```

---

## 📊 實測性能(同源性/分子控制測試分割)

| 端點 | 類型 | 模態 | 主指標 | 其他(同源性測試) | 隨機分割對照 |
|------|------|------|--------|------------------|------------|
| Hemolysis | binary | 序列 | AUC **0.7755** | MCC 0.3782, Acc 0.7009 | 0.7746(差 −0.0009) |
| Half_life | regression(log10 s) | 序列 | R² **0.5883** | RMSE 1.2502, MAE 0.8714 | 0.8650(近重複洩漏推高) |
| Caco2 | regression(logPapp) | 分子 | R² **0.3861** | RMSE 0.7879, MAE 0.4896 | —(無序列對照) |
| PAMPA_MDCK | regression(logPapp) | 分子 | R² **0.4573** | RMSE 0.8043, MAE 0.5070 | —(無序列對照) |

### 洩漏控制

- **序列端點**:3-mer Jaccard 家族同源性分割;分割前合併同 3-mer 多重集
  anagram(canonical signature),**保證 jaccard-1.0 精確複製不跨界**。
  最大跨界 Jaccard ≈ 0.97(近重複合理上限,非洩漏)。**Half_life 隨機
  R² 0.865 遠高於同源性 0.5883,是近重複洩漏被控管掉的誠實體現。**
- **分子端點**:無序列 → 按唯一 SMILES 分組(精確重複 SMILES 同分割);
  近異構物可跨界(SMILES-only 限制,已標註)。

---

## 🎯 特徵工程

**序列模態(428 維)**:AAC 20 + DPC 400 + 理化 8。

**分子模態(2,265 維)**:RDKit 2D 描述子 217(`CalcMolDescriptors`)+
Morgan fingerprint(半徑 2)2,048 位。無效 SMILES → 該行特徵全零並計入
清洗統計(不造假)。

---

## 📦 模型架構

- 共享 trunk:`input_dim → 256 → 128`(ReLU + BatchNorm + Dropout 0.2)
- 單任務 head:binary `Linear(128,1)` + sigmoid;regression `Linear(128,1)`
- 訓練:Adam(lr=1e-3, wd=1e-5)+ ReduceLROnPlateau + early stopping(val loss, patience 10)
- 參數數:序列端點 **143,617**;分子端點 **613,889**

---

## 🔄 完整工作流程

```
Chemit797/PepADMET-Dataset (整理/*.csv, 真實數據)
       ↓
prepare_pepadmet_data.py (載入 + 清洗 + 來源標記 → data/pepadmet_*.csv)
       ↓
train_pepadmet_model.py (每端點:
   序列端點 → 428 維 + 同源性分割(signature 合併, jaccard-1.0 不跨界)
   分子端點 → 2,265 維 + 唯一 SMILES 分割
   → 標準化 → MixedADMETMLP 單頭 → 訓練 → 雙分割評估
   → models_v4/<endpoint>/{權重, metrics.json})
       ↓
peptide_admet_predictor.py (輸入 sequence/SMILES → 自動路由 → 4 端點預測 + 單位)
```

---

## ⚠️ 誠實聲明

- 4 端點來自**真實實驗/文獻數據**(PepADMET-Dataset),非合成。
- 分子端點無序列,同源性控制受限於唯一 SMILES 分組(近異構物可跨界)。
- 清洗**丟棄**無效行(非 20-AA 序列、無效 SMILES、缺失標籤);
  丟棄統計寫入 `data/pepadmet_data.meta.json`,不隱瞞。
- 主指標報告在**同源性/分子控制測試分割**;隨機分割僅作洩漏對照。

---

## 🎉 總結

✅ 端對端管線跑通:prepare_pepadmet_data → train_pepadmet_model → peptide_admet_predictor
✅ 4 端點全部訓練 + 預測驗證(真實數據)
✅ 雙模態特徵(序列 428 維 + 分子 2,265 維)
✅ 洩漏控制(序列端 signature 合併保證 jaccard-1.0 不跨界;分子端唯一 SMILES)
✅ 實測指標:Hemolysis AUC 0.7755、Half_life R² 0.5883、Caco2 R² 0.3861、PAMPA R² 0.4573
✅ 4 個模型權重 + 準備數據 + 指標全部 commit 到 repo(可重現)

**版本**:4.0(真實數據 4 端點,雙模態)· **日期**:2026-08-25
