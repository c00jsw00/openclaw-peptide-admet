# 肽類 ADMET 預測工具 — 完整總結(v3.0 擴展版,2026-08-25)

> **v3.0 變更**(在 v2.0 誠實修訂基礎之上):
> 1. **訓練集可擴展**:`prepare_data.py --n <任意>` + `ingest_external.py`
>    外部數據攝入(驗證/去重/來源標記)+ `--merge` 併入。已實測 15k → 30k
>    規模,平均指標 0.6929 → 0.7189。
> 2. **預測項目擴展到 9 端點**(新增 pepADMET 的 4 個毒性端點):
>    `toxicity_binary`(二分類)、`toxicity_type`(6 類)、
>    `neurotoxicity_type`(4 類)、`HC50`(回歸),並採用 pepADMET 的
>    **partial-label mask 機制**(NaN = 該列未測,不參與該端點訓練)。
> 3. 模型改為 `MixedADMETMLP`(共享 trunk + per-task heads,145,681 參數),
>    保留舊版 `ADMETMLP` 以相容舊 checkpoint。
>
> v2.0 的誠實修訂(移除硬編碼 97.70%/0.9987、phantom 15k 數據、RF 偽裝
> NN)全部保留。

## ✅ 完成狀態:100%(端對端已跑通驗證,15k 與 30k 兩輪)

---

## 📁 專案文件

### 1. 管線腳本

| 文件 | 說明 |
|------|------|
| `endpoint_config.py` | 9 端點的單一事實來源(kind、類別數、pepADMET 欄位映射、合成標籤模型參數) |
| `prepare_data.py` | 產生 `--n` 行可再生成 synthetic_demo 數據集(9 端點 + partial labels),`--merge` 併入外部 CSV |
| `ingest_external.py` | 外部數據攝入:序列正規化、長度/字母表驗證、去重、`data_origin`/`sequence_provenance` 標記;SMILES-only 輸入可嘗試 RDKit 序列還原(低信任標記) |
| `smiles_to_sequence.py` | RDKit 結構式 α-碳偵測 → one-letter 序列(含置信度) |
| `homology_split.py` | 依氨基酸組合成家族做 70/10/20 同源性不相交分割,並**量測**洩漏(最大 pairwise Jaccard、端點標籤率差) |
| `admet_model.py` | `MixedADMETMLP`(binary/multiclass/regression heads)+ 舊版 `ADMETMLP`,訓練與預測共用,架構永不漂移 |
| `train_peptide_admet_model.py` | mask 感知混合損失 + **雙分割評估** + 每端點按 kind 的指標(binary AUC/MCC、多類 macro-F1/acc、回歸 R²/RMSE)→ `metrics.json` |
| `peptide_admet_predictor.py` | 推論 CLI;9 端點(機率/類別/回歸值)+ 風險標籤;**多目標綜合評分**與 `--rank` 候選排序;自動偵測 v2/v3 checkpoint |

### 2. 訓練產物(`peptide_admet_model/`)

| 文件 | 說明 |
|------|------|
| `admet_mlp.pt` | PyTorch state dict + 架構 metadata(含 `model_version: v3_mixed`) |
| `scaler.pt` | StandardScaler(torch.save) |
| `metrics.json` | **實測**每端點指標(兩種分割)+ 分割統計 + 洩漏稽核 |
| `feature_names.txt` | 428 個特徵名稱 |

### 3. 數據(`data/`,gitignored,可再生成)

| 文件 | 說明 |
|------|------|
| `peptide_admet_demo.csv` | `--n` 行 synthetic_demo(欄位:sequence, family_id, data_origin, 9 端點;NaN=未測) |
| `peptide_admet_demo.meta.json` | 生成參數與來源聲明 |
| `external_pepadmet.csv` | pepADMET sample 經 `ingest_external.py` 攝入的存活行(14/135) |
| `split/` | 同源性分割的 train/val/test 索引與 leakage_audit.json |

---

## 🚀 如何使用

```bash
# 1. 生成數據(可再生成,任意規模)
python prepare_data.py --n 30000

# 2. (選)攝入外部數據
python ingest_external.py --input real.csv --source mydata --output data/real.csv
python prepare_data.py --n 30000 --merge data/real.csv

# 3. 同源性控制分割(AMPBench-MT 式)
python homology_split.py

# 4. 訓練(混合損失 + mask;產出 metrics.json)
python train_peptide_admet_model.py --epochs 40

# 5. 預測(9 端點)
python peptide_admet_predictor.py --sequence "WALVKALVNHRISSSLVCG"
python peptide_admet_predictor.py --sequences test_sequences.txt --rank
```

---

## 📊 實測性能(30k 訓練,同源性控制測試分割 5,992 序列)

| 端點 | 類型 | 主指標 | 其他 | 已標註(%) |
|------|------|--------|------|-----------|
| GI_absorption | binary | AUC **0.8857** | MCC 0.4529, Acc 0.8092 | 100% |
| Caco2_permeability | binary | AUC **0.8831** | MCC 0.6135, Acc 0.8176 | 100% |
| BBB_penetration | binary | AUC **0.9042** | MCC 0.4640, Acc 0.8402 | 100% |
| Ames_mutagenicity | binary | AUC **0.8052** | MCC 0.3482, Acc 0.7067 | 100% |
| hERG_inhibition | binary | AUC **0.8602** | MCC 0.5375, Acc 0.7744 | 100% |
| toxicity_binary | binary | AUC **0.8268** | MCC 0.1225, Acc 0.7522 | 100% |
| toxicity_type | multiclass(6) | macro-F1 **0.3701** | acc 0.7402 | 100% |
| neurotoxicity_type | multiclass(4) | macro-F1 **0.3898** | acc 0.7477 | 12.6% |
| HC50 | regression | R² **0.5937** | RMSE 0.5058 | 30% |
| **平均主指標** | — | **0.7189** | 隨機分割對照 0.7227(delta −0.0038) | — |

對照:15k 訓練時平均主指標 0.6929、HC50 R² 0.4610 → **加大訓練集實測有效**。

> ⚠️ **這些數字描述的是示範管線,不是真實肽類性能。**
> `toxicity_type` 的 macro-F1 偏低是類別不平衡(class 0 佔多數)的誠實結果,
> 非 bug;換入平衡的真實毒性數據會改變此數字。
> `neurotoxicity_type`/`HC50` 只有部分行有標籤(partial labels),
> 指標僅在已標註子集上計算。

---

## 🎯 特徵工程(與 v2.0 相同,428 維)

1. **氨基酸組成 (AAC)** — 20
2. **二肽組成 (DPC)** — 400
3. **理化性質** — 8(MW 估、Kyte-Doolittle 疏水性、淨電荷、pI 估、GRAVY、疏水/帶電殘基比)

## 📦 模型架構(v3.0)

- 共享 trunk:428 → 256 → 128(ReLU + BatchNorm + Dropout 0.2)
- Per-task heads:
  - binary × 6:`Linear(128,1)` + sigmoid
  - multiclass:`Linear(128,6)`(toxicity_type)、`Linear(128,4)`(neurotoxicity_type)+ softmax
  - regression × 1:`Linear(128,1)`(HC50)
- 損失:mask 感知混合(BCE-with-logits + pos_weight / CrossEntropy / MSE,`reduction='none'` × mask,每端點平均再加權)
- 訓練:Adam(lr=3e-4)+ ReduceLROnPlateau + early stopping(val mixed loss)
- 參數數:145,681

## 🔄 完整工作流程

```
prepare_data.py (--n 任意, --merge 外部 CSV)
       ↓
ingest_external.py (外部數據: 驗證/去重/來源標記/partial labels)
       ↓
homology_split.py → 組合成家族 70/10/20 不相交分割 + 洩漏稽核
       ↓
train_peptide_admet_model.py
   428 維特徵 → 標準化 → MixedADMETMLP → 9 heads
   mask 感知混合損失;雙分割評估 → metrics.json
       ↓
peptide_admet_predictor.py
   輸入序列 → 428 維 → 標準化 → 9 端點預測
   → 多目標綜合評分(幾何平均)→ --rank 排序
```

## 📌 pepADMET 端點映射

| pepADMET 欄位 | openclaw 端點 | 類型 | 本管線合成標籤來源 |
|---|---|---|---|
| Toxicity | toxicity_binary | binary | 疏水性/長度潛在模型 |
| Toxicity_Type(6 類) | toxicity_type | multiclass | toxicity_binary 陰性→class 0,陽性→class 1-5 隨機 |
| Neurotoxicity_Type(4 類) | neurotoxicity_type | multiclass | 隨機 4 類 + 少量結構訊號(12.5% 行有標籤) |
| HC50(回歸) | HC50 | regression | 毒性潛變數的線性映射 + 雜訊(30% 行有標籤) |

## ⚠️ 誠實聲明:SMILES→序列

pepADMET 隨 repo 的 `Toxicity.csv`(135 行)**只有 SMILES 沒有序列欄**,
且其自带的氨基酸組成參考欄與 SMILES 結構不自洽(組成總和 ~100 vs
~10 殘基結構)。`ingest_external.py` 嘗試 RDKit 還原後,僅 **14/135**
行通過長度/組成健全性檢查,並標記 `sequence_provenance=smiles_inferred`
(低信任)。**我們不把这些行當作乾淨的真實數據**;管線「能」攝入真實
序列,但 pepADMET 的 sample 檔不是可靠來源。

---

## 🎉 總結

✅ 端對端管線跑通(15k + 30k 兩輪):prepare_data → (ingest_external) → homology_split → train → predict
✅ 訓練集可擴展(`--n` 任意 + `--merge` 外部 CSV),實測 15k→30k 平均指標 +0.026
✅ 9 端點(6 binary + 6 類 + 4 類 + 回歸),含 pepADMET 毒性端點與 partial-label mask
✅ 實測指標:30k 平均主指標 **0.7189**(同源性分割)
✅ 多目標綜合評分 + `--rank` 候選排序
✅ 所有文件已更新為 v3.0 誠實敘述

**版本**:3.0(extensibility + pepADMET endpoint expansion)· **日期**:2026-08-25
