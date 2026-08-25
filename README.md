# openclaw-peptide-admet

**Peptide ADMET 預測管線 — v4.1 真實數據 + ESMC 版(4 端點、雙模態、洩漏稽核)**

> **v4.1 增量變更(2026-08-26,建立在 v4.0 上):**
>
> 1. **序列端點(Hemolysis、Half-life)導入 ESMC-600M 凍結嵌入**:
>    Biohub 蛋白質語言模型對每條 20-AA 序列產出 1,152 維 mean-pooled
>    向量,與 428 維經典序列特徵**拼接成 1,580 維**輸入(嵌入是**凍結**的,
>    不 fine-tune,只用 `.venv-esmc`(Python ≥ 3.12)算一次並 commit npz 快取)。
>    同源性分割下 Hemolysis AUC **0.7755 → 0.8348**、Half-life R²
>    **0.5883 → 0.6973**(均實測重訓,非硬編)。
> 2. **Caco-2 / PAMPA/MDCK 維持 RDKit 分子路徑不變**(其「序列」欄是
>    非標準殘基名稱,20-AA 編碼器無法消耗),決定性重訓精確重現 v4.0。
> 3. 新增 `esmc_embed.py`(嵌入生成器)+ `data/esmc/*.npz`(凍結嵌入快取,
>    已 commit,免 3.12 環境即可重訓/推論)。
>
> **v4.0 斷裂式變更(2026-08-25,取代 v3.0):**
>
> 1. **改用真實數據集** [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset),
>    **刪除** v3.0 的 30,000 行 `synthetic_demo` 合成數據與 9 端點管線。
> 2. **只保留 4 端點**:Hemolysis、Half-life、Caco-2、PAMPA/MDCK
>    (v3.0 的 BBB / Ames / hERG / 毒性 / HC50 等不再保留)。
> 3. **雙模態特徵**:序列端點用 428 維氨基酸序列特徵(v4.1 起 +1,152 維
>    ESMC 嵌入 = 1,580 維);分子端點用 2,265 維 RDKit SMILES 描述子
>    (217 個 2D 描述子 + 2,048 位 Morgan)。
> 4. **4 個獨立單任務模型**——4 張表是**互斥分子**、分屬兩種模態,
>    共享 trunk 的多頭模型在此只會學兩個不相交子空間,故拆成 4 個聚焦模型。
>
> v2.0/v3.0 的誠實修訂(移除硬編碼指標、phantom 數據、RF 偽裝 NN)**全部保留**。

---

## 為什麼是真實數據、為什麼 4 端點

v3.0 的 30,000 行 `synthetic_demo` 是 `prepare_data.py` 用
Dirichlet 家族結構 + 潛在理化分數**合成**的示範數據——標籤不是實驗量測,
只能證明管線能跑,不能聲稱任何真實肽類準確度。v4.0 把它整個移除,改餵
**真實實驗/文獻數據**(PepADMET-Dataset),並收斂到你點名的 4 個端點:

| 端點 | 來源表(整理/) | 輸入 | 標籤 | 準備後行數 |
|------|------|------|------|:---:|
| **Hemolysis** | `hemolysis_unified/hemolysis_unified.csv` | `sequence_std`(20-AA 序列) | `label`(0/1 二值) | 8,719 |
| **Half-life** | `half_life_*` | `sequence`(20-AA 序列) | `half_life_seconds`(連續) | 1,763 |
| **Caco-2** | `caco2_*` | `SMILES`(RDKit 可解析) | `Permeability`(logPapp) | 7,429 |
| **PAMPA/MDCK** | `pampa_mdck_*` | `SMILES`(RDKit 可解析) | `PAMPA`(logPapp) | 7,283 |

> 關鍵:兩張通透性表(Caco-2 / PAMPA)的「序列」欄其實是 **CycPeptMPDB 的
> 非標準殘基名稱清單**(`MEL`、`DP`、`DL`、`ME_DL`…),20 標準氨基酸編碼器
> 無法消耗,因此這兩個端點改用**有效 SMILES 欄**做分子描述子特徵,並
> 逐端點在 `endpoint_config.py` 如實標註。

---

## 專案結構

| 文件 | 說明 |
|------|------|
| `endpoint_config.py` | 4 端點單一事實來源(kind、模態、來源 CSV、特徵欄、標籤轉換、單位、**ESMC 旗標**) |
| `feature_extractor.py` | 雙模態特徵(訓練/推論共用同一份):序列 428 維 + RDKit 分子 2,265 維 |
| `esmc_embed.py` | **v4.1** ESMC-600M 凍結嵌入生成器(`.venv-esmc` 執行;預設批次模式或 `--sequences-file` ad-hoc 模式) |
| `prepare_pepadmet_data.py` | 載入 4 個真實 CSV → 清洗(去無效序列/無效 SMILES/缺失標籤)→ 每端點準備 CSV + `meta.json` |
| `homology_split.py` | 3-mer Jaccard 家族 70/10/20 同源性分割 + 量測洩漏 |
| `admet_model.py` | `MixedADMETMLP`(binary/regression heads,通用 input_dim)+ 存檔 |
| `train_pepadmet_model.py` | 每端點:特徵 →(v4.1 序列端點 +ESMC 拼接)→ 分割 → 標準化 → 訓練 → 雙分割評估 → `metrics.json` + 權重 |
| `peptide_admet_predictor.py` | 推論 CLI:輸入 sequence / SMILES → 自動路由模態 →(v4.1 序列端點快取/子程序嵌入)→ 預測 + 單位 |

**訓練產物 `models_v4/`**(已 commit):

| 路徑 | 說明 |
|------|------|
| `models_v4/<endpoint>/admet_mlp.pt` | PyTorch 權重 + 架構 metadata |
| `models_v4/<endpoint>/scaler.pt` | StandardScaler(只在 train 上 fit) |
| `models_v4/<endpoint>/metrics.json` | **實測**指標(雙分割)+ 分割統計 + 洩漏稽核 |
| `models_v4/summary.json` | 4 端點彙總 |

`<endpoint>` ∈ {`hemolysis`, `half_life`, `caco2`, `pampa_mdck`}。

---

## 快速開始

```bash
# 0. 環境(需要 rdkit;CPU 版 torch 即可)
uv pip install --python .venv/Scripts/python.exe rdkit

# 0b. (v4.1) ESMC 環境:Python >= 3.12 + Biohub esm(git main)
#     只需在「重新生成嵌入」或「預測全新序列」時用到;
#     重訓與預測已快取的序列不需要它(npz 已 commit)。
#     建立方式見 esmc_embed.py 檔頭,或用 uv venv --python 3.12 手動建。

# 1. 準備數據(從本機已 clone 的 PepADMET-Dataset 載入 + 清洗)
python prepare_pepadmet_data.py

# 2. (v4.1,選做)重新生成 ESMC 嵌入 → data/esmc/*.npz
#    首次或序列集變動時才需要;已 commit 的 npz 可直接用。
.venv-esmc/Scripts/python.exe esmc_embed.py

# 3. 訓練 4 端點(CPU,約 5–6 分鐘;ESMC 端點讀 npz 快取,分子端點 RDKit 為主要開銷)
python train_pepadmet_model.py --epochs 80 --seed 42

# 4. 預測(自動路由模態:sequence → 序列端點,SMILES → 分子端點)
python peptide_admet_predictor.py --sequence "ACDEFGHIKLMNPQRSTVWY"
python peptide_admet_predictor.py --smiles "CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O"
```

---

## 實測性能(洩漏控制測試分割,seed 42)

全部數字取自 `models_v4/<endpoint>/metrics.json`,**實測、可重導**,非硬編碼。

| 端點 | 類型 | 模態 | 主指標 | 其他(控制分割) | 隨機分割對照 |
|------|------|------|--------|----------------|------------|
| Hemolysis | binary | 序列+ESMC | AUC **0.8348** | MCC 0.4557, Acc 0.7479 | 0.8112(差 −0.0236) |
| Half-life | regression(log10 s) | 序列+ESMC | R² **0.6973** | RMSE 1.072, MAE 0.7705 | 0.8733(近重複洩漏推高) |
| Caco-2 | regression(logPapp) | 分子 | R² **0.3861** | RMSE 0.7879, MAE 0.4896 | —(無序列對照) |
| PAMPA/MDCK | regression(logPapp) | 分子 | R² **0.4573** | RMSE 0.8043, MAE 0.5070 | —(無序列對照) |

> **v4.1 ESMC 增益(同源性控制分割,seed 42)**:
> Hemolysis AUC 0.7755 → **0.8348**(Δ +0.0593)、
> Half-life R² 0.5883 → **0.6973**(Δ +0.1090)。
> 兩個序列端點都從 428 維經典特徵升級為 428 + 1,152(ESMC-600M)
> = 1,580 維;分子端點不變(2,265 維),決定性重訓精確重現 v4.0。

### 洩漏控制(這是 v4.0 的核心方法貢獻)

- **序列端點(Hemolysis、Half-life)**:AMPBench-MT 風格同源性分割
  (arXiv:2607.25518)。分割前先把**同 3-mer 多重集的 anagram 合併為同一
  家族**(canonical 3-mer-multiset signature)——**保證 jaccard-1.0 的精確
  複製(含保長 anagram)絕不跨界**。最大跨界 Jaccard ≈ 0.97(近重複的合理
  上限,非洩漏)。**Half-life 隨機 R² 0.865 遠高於同源性 0.5883,正是
  近重複/近 anagram 洩漏被控管掉的誠實體現**(delta +0.277)。
- **分子端點(Caco-2、PAMPA/MDCK)**:無序列 → 按**唯一 SMILES** 分組
  (精確重複 SMILES 同分割);**近異構物**(不同 SMILES、同化學)可跨界,
  這是 SMILES-only 數據的限制,已如實寫入每個端點的 `metrics.json`。

---

## 特徵工程(訓練/推論共用 `feature_extractor.py`)

**序列模態(428 維經典 + 1,152 維 ESMC = 1,580 維,v4.1)**:
- 經典 428 維:AAC 20(氨基酸組成)+ DPC 400(二肽組成)+ 8 理化
  (估算 MW、平均 Kyte–Doolittle 親水性、pH7 淨電荷、pI 估計、GRAVY、
  疏水/帶電殘基比)。
- **ESMC-600M 凍結嵌入 1,152 維(v4.1)**:Biohub ESM Cambrian 蛋白質
  語言模型對每條 20-AA 序列做 attention-mask 加權 mean-pooling。嵌入是
  **凍結**的(不 fine-tune、不梯度),由 `esmc_embed.py` 在 `.venv-esmc`
  (Python ≥ 3.12)算一次,存 `data/esmc/*.npz`(已 commit)。訓練時與經典
  特徵**拼接**;推論時快取序列直接取、全新序列走子程序回退。

**分子模態(2,265 維,不變)**:RDKit 2D 描述子 217(`CalcMolDescriptors`,
固定確定性註冊表)+ Morgan fingerprint(半徑 2)2,048 位。無效 SMILES →
該行特徵全零並計入清洗統計(不造假)。

---

## 模型

- 共享 trunk:`input_dim → 256 → 128`(ReLU + BatchNorm + Dropout 0.2)
- 單任務 head:binary `Linear(128,1)` + sigmoid;regression `Linear(128,1)`
- 訓練:Adam(lr=1e-3, wd=1e-5)+ ReduceLROnPlateau(factor 0.5, patience 4)
  + early stopping(val loss, patience 10)
- 參數數:序列端點(1,580 維)**438,529**;分子端點(2,265 維)**613,889**
  (v4.0 序列端點 143,617 為 428 維;v4.1 拼 ESMC 後輸入變 1,580 維)

---

## 誠實聲明

- 4 端點來自**真實實驗/文獻數據**(PepADMET-Dataset),非合成。
- **ESMC-600M 嵌入是凍結的、外部預訓練的**(Biohub ESM Cambrian),
  本管線**不 fine-tune** ESMC——只把它的 mean-pooled 向量當附加特徵。
  嵌入 npz 已 commit,重訓/推論不需重算;`data/esmc/*.npz` 的 meta
  記錄模型、維度、池化方式。
- 分子端點無序列,同源性控制受限於唯一 SMILES 分組(近異構物可跨界);
  ESMC 對分子端點**不適用**(其「序列」欄是非標準殘基,標準 20-AA 佔比
  < 0.5%),故分子端點維持 RDKit 路徑、未拼 ESMC。
- 清洗**丟棄**無效行(非 20-AA 序列、無效 SMILES、缺失標籤);
  丟棄統計寫入 `data/pepadmet_data.meta.json`,不隱瞞。
- 主指標報告在**同源性/分子控制測試分割**;隨機分割僅作洩漏對照。
- 所有性能數字是**實測值**(見 `metrics.json`),可用
  `prepare_pepadmet_data.py → train_pepadmet_model.py` 在 CPU 上重導
  (ESMC 端點直接讀已 commit 的 npz)。

---

## 版本

**v4.1**(真實數據 4 端點 + ESMC-600M 凍結嵌入,雙模態)· 2026-08-26
建立在 v4.0(2026-08-25,真實數據 4 端點、雙模態)之上;
v4.0 取代 v3.0(30k `synthetic_demo` 合成數據 9 端點)。
完整技術說明與方法論見 [peptide_admet_manuscript_jcim.md](peptide_admet_manuscript_jcim.md);
端點摘要見 [PREDICTOR_SUMMARY.md](PREDICTOR_SUMMARY.md)。
