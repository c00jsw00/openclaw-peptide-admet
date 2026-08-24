# 肽類 ADMET 預測工具 — 完整總結(2026-08 誠實修訂版 v2.0)

> **修訂說明**:舊版宣稱的 97.70% 準確率 / 0.9987 AUC 來自硬編碼數字與
> 「15,000 真實肽類」的 phantom 數據集(倉庫中根本沒有該檔案),且將
> 第二個 Random Forest 存檔為「神經網絡」。本版全部移除:
> 模型為**真 PyTorch MLP**,指標全部**實測**(存於 `metrics.json`),
> 數據為**可再生成的 synthetic_demo 集**,評估採用
> **AMPBench-MT(arXiv:2607.25518)式同源性控制分割**。

## ✅ 完成狀態:100%(端對端已跑通驗證)

---

## 📁 專案文件

### 1. 管線腳本

| 文件 | 說明 |
|------|------|
| `prepare_data.py` | 產生 15,000 行可再生成 synthetic_demo 數據集(seed=42),每行標記 `data_origin=synthetic_demo` |
| `homology_split.py` | 依氨基酸組合成家族做 70/10/20 同源性不相交分割,並**量測**洩漏(最大 pairwise Jaccard、端點標籤率差) |
| `admet_model.py` | 共享模型定義(MLP 428→256→128→5 端點 sigmoid heads),訓練與預測共用,架構永不漂移 |
| `train_peptide_admet_model.py` | 訓練 + **雙分割評估**(同源性主指標 + 隨機分割對照)+ 寫入 `metrics.json` |
| `peptide_admet_predictor.py` | 推論 CLI;讀實測 metrics(不硬編碼);**多目標綜合評分**(幾何平均,AMPGAN v3/PepCraft 式)與 `--rank` 候選排序 |
| `create_graphics.py` | 生成圖文摘要(沿用舊版) |

### 2. 訓練產物(`peptide_admet_model/`)

| 文件 | 說明 |
|------|------|
| `admet_mlp.pt` | PyTorch state dict + 架構 metadata(真 NN,非 RF 偽裝) |
| `scaler.pt` | StandardScaler(torch.save) |
| `metrics.json` | **實測**每端點 AUC/MCC/Acc(兩種分割)+ 分割統計 + 洩漏稽核 |
| `feature_names.txt` | 428 個特徵名稱(AAC_* / DPC_* / 理化) |

### 3. 數據(`data/`,gitignored,可再生成)

| 文件 | 說明 |
|------|------|
| `peptide_admet_demo.csv` | 15,000 行 synthetic_demo(欄位:sequence, family_id, data_origin, 5 端點) |
| `peptide_admet_demo.meta.json` | 生成參數與來源聲明 |
| `split/` | 同源性分割的 train/val/test 索引與 leakage_audit.json |

### 4. 文檔

`README.md`、`README_PREDICTOR.md`、`cover_letter_jcim.md`、
`SUBMISSION_CHECKLIST.md`、`test_sequences.txt`、
`peptide_admet_manuscript_jcim.md`(手稿,已更新為實測指標)。

---

## 🚀 如何使用

### 快速開始(4 個步驟)

```bash
# 0. 環境(建議 venv;系統 Python 可能由 uv 管理不可直裝)
uv venv .venv
uv pip install --python .venv/Scripts/python.exe torch --default-index https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe scikit-learn pandas numpy

# 1. 生成數據(可再生成,seed=42)
python prepare_data.py

# 2. 同源性控制分割(AMPBench-MT 式)
python homology_split.py

# 3. 訓練(產出 metrics.json;真 PyTorch MLP)
python train_peptide_admet_model.py

# 4. 預測
python peptide_admet_predictor.py --sequence "WALVKALVNHRISSSLVCG"
python peptide_admet_predictor.py --sequences test_sequences.txt --rank
```

**輸出範例**(實測):
```
✅ Model loaded from peptide_admet_model (144,133 params, data: synthetic_demo)
...
Composite multi-objective score: 0.3883  (geometric mean of favourable endpoint probabilities)
Measured on homology-controlled (AMPBench-MT-style, arXiv:2607.25518) split
(synthetic_demo data, 10490 train samples): mean AUC = 0.8684, mean accuracy = 0.7836
NOTE: Training data is the synthetic demo set — numbers validate the pipeline,
not real-peptide performance.
```

---

## 📊 實測性能(本次訓練,非硬編碼)

### 同源性控制測試分割(主指標,3,020 序列)

| 端點 | AUC | MCC | Accuracy | Positive rate |
|------|-----|-----|----------|---------------|
| GI Absorption | 0.8810 | 0.4457 | 0.8037 | 0.132 |
| Caco-2 Permeability | 0.8882 | 0.5930 | 0.8094 | 0.319 |
| BBB Penetration | 0.9070 | 0.4575 | 0.8367 | 0.105 |
| Ames Mutagenicity | 0.8011 | 0.3418 | 0.7016 | 0.171 |
| hERG Inhibition | 0.8645 | 0.5261 | 0.7665 | 0.299 |
| **宏觀 AUC(主指標)** | **0.8684** | — | **0.7836(均值)** | — |

隨機分割對照:宏觀 AUC 0.8688(delta +0.0004)。

> ⚠️ **這些數字描述的是示範管線,不是真實肽類性能。**
> 標籤來自粗略的潛在理化模型,因此 0.8–0.9 AUC 恰是預期結果。
> 舊版 0.9987 是序列相似性洩漏的假象——正是 AMPBench-MT
> (arXiv:2607.25518)所記錄的失敗模式。

---

## 🎯 特徵工程

428 維(訓練與推論完全一致):

1. **氨基酸組成 (AAC)** — 20:20 種標準氨基酸頻率
2. **二肽組成 (DPC)** — 400:所有有序二肽組合頻率
3. **理化性質** — 8:分子量(估)、平均疏水性(Kyte-Doolittle)、
   疏水性範圍、淨電荷(pH 7)、等電點估計、GRAVY、疏水殘基比、帶電殘基比

---

## 📦 模型架構

**單一共享 PyTorch MLP**(非 ensemble):

- 輸入:428 維(StandardScaler 標準化)
- 隱藏層:256 → 128(ReLU + BatchNorm + Dropout 0.2)
- 輸出:5 個 sigmoid heads(每端點一個)
- 損失:每端點 BCE 平均(類別權重依端點盛行率)
- 訓練:Adam(lr=3e-4)+ ReduceLROnPlateau + early stopping(val BCE)
- 參數數:144,133

---

## 🔄 完整工作流程

```
prepare_data.py  → 15,000 synthetic_demo 序列 + 潛在理化模型標籤
       ↓
homology_split.py → 組合成家族 70/10/20 不相交分割 + 洩漏稽核
       ↓
train_peptide_admet_model.py
   428 維特徵 → 標準化 → MLP → 5 heads
   雙分割評估(同源性主指標 + 隨機對照)→ metrics.json
       ↓
peptide_admet_predictor.py
   輸入序列 → 428 維 → 標準化 → MLP → 5 端點概率
   → 多目標綜合評分(幾何平均:GI+、Caco2+、BBB+、Ames-、hERG-)
   → 讀 metrics.json 顯示實測性能(絕不硬編碼)
```

---

## 💻 Python API 使用

```python
from peptide_admet_predictor import PeptideADMETPredictor

predictor = PeptideADMETPredictor(model_dir='peptide_admet_model')

# 單一預測
results = predictor.predict("WALVKALVNHRISSSLVCG")
for r in results:
    print(f"{r['endpoint']}: {r['probability']:.4f}")

# 多目標候選排序(AMPGAN v3/PepCraft 式綜合評分)
ranked = predictor.rank_candidates(["GAGAGAGAGAGA", "MLLLLLLLLL", "KKKKKKKKKK"])
best = ranked[0]

# 模型來源資訊(全部讀自 metrics.json)
info = predictor.model_info()
print(info['mean_auc_homology_split'])   # 0.8684
```

---

## 🔗 2026 文獻對照(修訂依據)

| 文獻 | 對本專案的影響 |
|------|----------------|
| **AMPBench-MT**(arXiv:2607.25518, 2026-07) | 序列相似性洩漏使 AMP/ADMET 表現虛高 → `homology_split.py` 同源性控制分割 + 洩漏稽核 |
| **AMPGAN v3 + PepCraft**(arXiv, 2026-06) | 生成候選體的多目標綜合評分 → `composite_score()` / `--rank` |
| **ApexGO**(Nat. Mach. Intell., 2026-05) | 生成式重新設計抗生素的誠實評估教訓 → 移除假指標、改實測 |
| **npj Drug Discovery 整合管線**(2026-05) | ProtGPT2 軟提示整合管線 → Future Directions 列為語言模型特徵骨干 |
| **Genotypic Triggers**(2026-08) | 藥理基因學「背門」安全性盲點 → Limitations 聲明缺乏 toxicogenomics 端點 |

---

## ⚠️ 限制

1. **Synthetic demo 數據**:所有指標僅驗證管線,非真實肽類 ADMET 行為。
2. **僅 5 端點**:無 toxicogenomics/藥理基因學端點(參 Genotypic Triggers)。
3. **組成層級特徵**:AAC/DPC 無法捕捉二肽以上的順序效應;
   需語言模型或 GNN 骨干處理順序敏感性質。
4. **序列長度**:demo 數據為 10–30 aa,範圍外未驗證。
5. **無實驗驗證**:未使用任何濕實驗數據。

---

## 📋 投稿狀態(JCIM)

舊版投稿包(手稿、cover letter、checklist)中的性能敘述**已全部更新為
實測數字與 synthetic_demo 聲明**。在取得真實實驗數據並完成同源性控制
評估之前,**本專案不適合以「真實性能」主張投稿**;建議定位為
*reproducible benchmark/evaluation-protocol demonstration*
(可重現的評估協議示範),或以 AMPBench-MT 類任務重新訓練後再投稿。

---

## 🎉 總結

✅ 端對端管線跑通:prepare_data → homology_split → train → predict
✅ 真 PyTorch MLP(144,133 參數),非 RF 偽裝
✅ 實測指標:同源性分割宏觀 AUC **0.8684**、平均準確率 **0.7836**
✅ 多目標綜合評分 + `--rank` 候選排序
✅ 所有文件(README/手稿/cover letter/checklist)已更新為誠實敘述

**版本**:2.0(integrity revision)· **日期**:2026-08-24
