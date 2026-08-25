# 肽類 ADMET 預測工具 — 完整總結(v4.1 真實數據 + ESMC 版,2026-08-26)

> **v4.1 增量變更**(建立在 v4.0 上):
> 1. **序列端點(Hemolysis、Half_life)導入 ESMC-600M 凍結嵌入**:
>    Biohub 蛋白質語言模型產出 1,152 維 mean-pooled 向量,與 428 維經典
>    序列特徵拼接成 **1,580 維**輸入(凍結、不 fine-tune;`.venv-esmc`
>    算一次 → commit `data/esmc/*.npz`)。Hemolysis AUC **0.7755 → 0.8348**、
>    Half_life R² **0.5883 → 0.6973**(實測重訓)。
> 2. **Caco2 / PAMPA_MDCK 維持 RDKit 分子路徑**(非標準殘基,ESMC 不適用),
>    決定性重訓精確重現 v4.0(0.3861 / 0.4573)。
> 3. 新增 `esmc_embed.py`(嵌入生成器)+ `data/esmc/*.npz`(凍結嵌入快取)。
>
> **v4.0 斷裂式變更**(取代 v3.0):
> 1. **改用真實數據集** [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset),
>    **刪除** v3.0 的 30,000 行 `synthetic_demo` 合成數據與 9 端點管線。
> 2. **4 端點**:Hemolysis、Half-life、Caco-2、PAMPA/MDCK(其餘 5 個
>    v3.0 端點 BBB/Ames/hERG/毒性/HC50 不再保留)。
> 3. **雙模態特徵**:序列端點用 428 維氨基酸序列特徵(v4.1 起 +1,152 維
>    ESMC = 1,580 維);分子端點用 2,265 維 RDKit SMILES 描述子
>    (217 2D 描述子 + 2,048 位 Morgan)。
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
| `endpoint_config.py` | 4 端點的單一事實來源(kind、模態、來源 CSV、特徵欄、標籤轉換、單位、**ESMC 旗標**) |
| `feature_extractor.py` | 雙模態特徵:序列 428 維(AAC 20 + DPC 400 + 理化 8)+ RDKit 分子 2,265 維(217 2D 描述子 + 2,048 位 Morgan) |
| `esmc_embed.py` | **v4.1** ESMC-600M 凍結嵌入生成器(`.venv-esmc` 執行;批次 / ad-hoc 雙模式) |
| `prepare_pepadmet_data.py` | 載入 4 個真實 CSV → 清洗(去 X/非 20-AA/無效 SMILES/缺失標籤)→ 每端點輸出準備 CSV + meta |
| `homology_split.py` | 3-mer Jaccard 家族 70/10/20 同源性不相交分割 + 量測洩漏 |
| `admet_model.py` | `MixedADMETMLP`(binary/regression heads,通用 input_dim)+ 訓練/預測共用 |
| `train_pepadmet_model.py` | 每端點:特徵 →(v4.1 序列端點 +ESMC 拼接)→ 分割 → 標準化 → 訓練 → 雙分割評估 → `metrics.json` + 權重 |
| `peptide_admet_predictor.py` | 推論 CLI:輸入 sequence / SMILES → 自動路由模態 →(v4.1 序列端點快取/子程序嵌入)→ 4 端點預測 + 單位 |

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
| `data/esmc/esmc_emb_hemolysis.npz` | **v4.1** 凍結 ESMC-600M 嵌入(8,719 × 1,152 float32) |
| `data/esmc/esmc_emb_half_life.npz` | **v4.1** 凍結 ESMC-600M 嵌入(1,763 × 1,152 float32) |
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
| Hemolysis | binary | 序列+ESMC | AUC **0.8348** | MCC 0.4557, Acc 0.7479 | 0.8112(差 −0.0236) |
| Half_life | regression(log10 s) | 序列+ESMC | R² **0.6973** | RMSE 1.072, MAE 0.7705 | 0.8733(近重複洩漏推高) |
| Caco2 | regression(logPapp) | 分子 | R² **0.3861** | RMSE 0.7879, MAE 0.4896 | —(無序列對照) |
| PAMPA_MDCK | regression(logPapp) | 分子 | R² **0.4573** | RMSE 0.8043, MAE 0.5070 | —(無序列對照) |

> **v4.1 ESMC 增益**(同源性控制分割,seed 42):
> Hemolysis AUC 0.7755 → **0.8348**(Δ +0.0593)、
> Half_life R² 0.5883 → **0.6973**(Δ +0.1090)。
> 序列端點 428 → 1,580 維(428 + 1,152 ESMC);分子端點 2,265 維不變,
> 決定性重訓精確重現 v4.0。

### 洩漏控制

- **序列端點**:3-mer Jaccard 家族同源性分割;分割前合併同 3-mer 多重集
  anagram(canonical signature),**保證 jaccard-1.0 精確複製不跨界**。
  最大跨界 Jaccard ≈ 0.97(近重複合理上限,非洩漏)。**Half_life 隨機
  R² 0.873 遠高於同源性 0.6973,是近重複洩漏被控管掉的誠實體現**
  (v4.1 拼 ESMC 後同源性從 0.5883 升到 0.6973,隨機對照仍保持 0.87 量級)。
- **分子端點**:無序列 → 按唯一 SMILES 分組(精確重複 SMILES 同分割);
  近異構物可跨界(SMILES-only 限制,已標註)。

---

## 🎯 特徵工程

**序列模態(428 維經典 + 1,152 維 ESMC = 1,580 維,v4.1)**:
- 經典 428 維:AAC 20 + DPC 400 + 理化 8。
- **ESMC-600M 凍結嵌入 1,152 維**:Biohub ESM Cambrian 蛋白質語言模型,
  attention-mask 加權 mean-pooling;`esmc_embed.py` 在 `.venv-esmc`
  (Python ≥ 3.12)算一次 → `data/esmc/*.npz`(已 commit)。凍結、不
  fine-tune;訓練時拼接、推論時快取/子程序回退。

**分子模態(2,265 維,不變)**:RDKit 2D 描述子 217(`CalcMolDescriptors`)+
Morgan fingerprint(半徑 2)2,048 位。無效 SMILES → 該行特徵全零並計入
清洗統計(不造假)。

---

## 📦 模型架構

- 共享 trunk:`input_dim → 256 → 128`(ReLU + BatchNorm + Dropout 0.2)
- 單任務 head:binary `Linear(128,1)` + sigmoid;regression `Linear(128,1)`
- 訓練:Adam(lr=1e-3, wd=1e-5)+ ReduceLROnPlateau + early stopping(val loss, patience 10)
- 參數數:序列端點(1,580 維)**438,529**;分子端點(2,265 維)**613,889**
  (v4.0 序列端點 143,617 為 428 維;v4.1 拼 ESMC 後輸入變 1,580 維)

---

## 🔄 完整工作流程

```
Chemit797/PepADMET-Dataset (整理/*.csv, 真實數據)
       ↓
prepare_pepadmet_data.py (載入 + 清洗 + 來源標記 → data/pepadmet_*.csv)
       ↓
esmc_embed.py (v4.1,選做:ESMC-600M 凍結嵌入 → data/esmc/*.npz,
   首次或序列集變動時才需;已 commit 的 npz 可直接用)
       ↓
train_pepadmet_model.py (每端點:
   序列端點 → 428 維 + 1,152 維 ESMC = 1,580 維
              + 同源性分割(signature 合併, jaccard-1.0 不跨界)
   分子端點 → 2,265 維 + 唯一 SMILES 分割
   → 標準化 → MixedADMETMLP 單頭 → 訓練 → 雙分割評估
   → models_v4/<endpoint>/{權重, metrics.json})
       ↓
peptide_admet_predictor.py (輸入 sequence/SMILES → 自動路由 →
   序列端點:快取嵌入 / 新序列走 .venv-esmc 子程序 → 4 端點預測 + 單位)
```

---

## ⚠️ 誠實聲明

- 4 端點來自**真實實驗/文獻數據**(PepADMET-Dataset),非合成。
- **ESMC-600M 嵌入是凍結的、外部預訓練的**(Biohub ESM Cambrian),
  本管線**不 fine-tune**——只把 mean-pooled 向量當附加特徵。npz 已
  commit,重訓/推論不需重算。
- 分子端點無序列,同源性控制受限於唯一 SMILES 分組(近異構物可跨界);
  ESMC 對分子端點**不適用**(非標準殘基,標準 20-AA < 0.5%),未拼 ESMC。
- 清洗**丟棄**無效行(非 20-AA 序列、無效 SMILES、缺失標籤);
  丟棄統計寫入 `data/pepadmet_data.meta.json`,不隱瞞。
- 主指標報告在**同源性/分子控制測試分割**;隨機分割僅作洩漏對照。

---

## 🎉 總結

✅ 端對端管線跑通:prepare_pepadmet_data → esmc_embed(選做)→ train_pepadmet_model → peptide_admet_predictor
✅ 4 端點全部訓練 + 預測驗證(真實數據)
✅ 雙模態特徵(序列 428 + 1,152 ESMC = 1,580 維;分子 2,265 維)
✅ v4.1 ESMC 增益:Hemolysis AUC 0.7755 → **0.8348**、Half_life R² 0.5883 → **0.6973**(同源性分割)
✅ 洩漏控制(序列端 signature 合併保證 jaccard-1.0 不跨界;分子端唯一 SMILES)
✅ 實測指標:Hemolysis AUC 0.8348、Half_life R² 0.6973、Caco2 R² 0.3861、PAMPA R² 0.4573
✅ 4 個模型權重 + 準備數據 + ESMC 嵌入 npz + 指標全部 commit 到 repo(可重現)
✅ E2E 驗證:predictor 路徑精確重現全部 shipped 指標;CLI 快取/新序列雙路徑通過

**版本**:4.1(真實數據 4 端點 + ESMC-600M 凍結嵌入,雙模態)· **日期**:2026-08-26
