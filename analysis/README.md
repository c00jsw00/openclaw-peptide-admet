# PAMPA R² 天花板分析(2026-08-27)

本目錄保存 v4.2 PAMPA 端點 R² = 0.4642「為何無法達到 0.70」的完整調查。
**結論:0.70 在当前數據上不可達**——不是模型/特徵/架構問題,是目標變數的
**左側審查(left-censoring)地板 + 標籤噪音**的數學限制。以下每條結論都有
可重現的腳本與實測數字。

## 結論摘要

| 問題 | 答案(實測) | 證據腳本 |
|---|---|---|
| 目標變數結構 | 7,283 行只有 **648 個唯一值**(量化到 0.01);**269 行(3.7%)恰好 = -10.0000**,是 assay 偵測下限的**審查地板**;floor 佔全域 SS 的 **49.6%** | `r2_ceiling.py` |
| R² 損失集中在哪 | 地板點佔總平方誤差 **64%**(47 個 test 點,MAE 3.43 log 單位);非地板子集模型 R² 已達 **0.6317** | `round3_strong_2d.py` 段 A |
| 地板分子可否從結構預測? | **只能部分排序、無可用操作點**。最佳 LightGBM 地板分類 AUC_test = **0.8557**(val 0.8251)、MLP 預測值排序 AUC 0.7624;val 調閾下 precision 僅 0.121(recall 0.617),兩階段法 R² 崩潰 | `floor_predictability.py`、`soft_blend.py` |
| 理論天花板 | 完美非地板回歸 + 地板→全域均值 = **R² 0.5387**;oracle 完美識別地板 = 0.807(不可達) | `r2_ceiling.py` |
| 5 條提升路線 | 全部 ≤ 0.47(見下表),無一超過 baseline 0.4642 | round1–3 + blend + tobit |

## 五條路線的實測結果(同一分割:seed 42、unique-SMILES 70/10/20)

| 路線 | 最佳 test R² | 判定 |
|---|---:|---|
| v4.2 baseline(MLP 3,033→256→128→1, Huber) | **0.4642** | 基準 |
| 1. Rank-Gaussian 目標變換(train-only mapping,多 seed) | 0.434 ± 0.011 | ❌ 比基準差 |
| 2. 強 2D:LightGBM × 128 超參 × 4 特徵集 + top-5 ensemble | 0.4234 | ❌ 比基準差 |
| 3. 兩階段地板法(classifier→regressor) | AUC 0.86 但最佳閾 precision 0.121 → R² −1.21 | ❌ 誤報成本使地板不可用於修正 |
| 4. Soft posterior-mean blend(β 掃描,val 選參) | 0.4651(+0.0009,噪音內) | ❌ 無實質增益 |
| 5. Tobit 審查似然(統計上最正確) | 0.4056 ± 0.024 | ❌ 比基準差 |

輔助 ablation(round1):RDKit 217 描述子單獨 **−0.47**(有害,過拟合);
更多 Morgan 半徑 0.24(過拟合);更寬/更深 MLP 0.44(無幫助);信號在
Morgan 指紋,不在標量描述子或模型容量。

## 為何 0.7 不可達(數學)

R² = 1 − SSE/SST。PAMPA 目標 y = logPapp 的變異數結構:

| 區間 | 行數 | 全域 SS 占比 | 說明 |
|---|---:|---:|---|
| y = -10.0000(審查地板) | 269 | **49.6%** | assay 偵測下限,真實值未知(≤ -10);行值近常數,占比全部來自偏移 |
| (-10, -6] | 5,615 | 22.6% | 模型 R² 0.63 的 bulk |
| y > -6(高滲透尾) | 1,457 | 27.8% | 重尾 |

- 地板行佔**近一半**總變異數(全域平方偏差的 49.6%)。模型只能把地板
  分子**部分排序**(AUC 0.76–0.86)、無法精確標記(最佳閾 precision 0.12),
  所以這 49.6% 的變異數大部分無法解釋 → R² 上限被鎖在 ~0.54。
- 即便 oracle 完美標記地板分子,上限 0.807 也依賴不可達的條件。
- 現行 0.4642 距離 0.5387 天花板尚有 0.074 的差距,但**橋過天花板本身
  不可能**(它要求非地板回歸完美);天花板之上的 0.70 需要地板分子有
  可預測的真實值——即**無審查的重新量測**。

## 要真正達到 0.7 需要什麼

1. **無審查的重新量測**:對 y = -10.0000 的 269 個分子做更敏感 assay
   (或文獻值),取得真實 logPapp。這直接移除 49.6% 變異數的不可預測部分。
2. **降低重複量測噪音**:重複組 within-SD σ̂ ≈ 0.20(log 單位)已不低,
   但相對 var(y)=1.26 尚可控;主要瓶頸是審查,不是重複噪音。
3. **更大樣本**:7,283 行中只有 648 個唯一值(0.01 量化),有效樣本量遠
   低於 7,283;更多**獨立量測的分子**才能壓低標籤噪音。

## 重現方法

```bash
cd <repo root>
.venv/Scripts/python.exe analysis/tobit_sanity.py          # 秒級,驗證 NLL 函數
.venv/Scripts/python.exe analysis/r2_ceiling.py            # ~2 min,天花板分解(先跑,建快取)
.venv/Scripts/python.exe analysis/round1_feature_ablation.py   # ~25 min
.venv/Scripts/python.exe analysis/round2_rank_gaussian.py      # ~25 min(首次)
.venv/Scripts/python.exe analysis/round3_strong_2d.py          # ~30 min
.venv/Scripts/python.exe analysis/floor_predictability.py      # ~10 min
.venv/Scripts/python.exe analysis/soft_blend.py                # ~10 min
.venv/Scripts/python.exe analysis/tobit_censored.py            # ~15 min
```

- 所有腳本用**與主訓練管線逐字相同的分割**(`train_pepadmet_model.py`
  的 `split_molecular`,seed 42),所以 R² 與已提交的 0.4642 直接可比。
- 特徵快取 `_pampa_feat_cache.npz`(176MB,repo root)由 `r2_ceiling.py` /
  round2 / round3 首次運行建立(~8 min,CPU),其餘腳本直接讀取;
  `.gitignore` 已排除。
- `import common`(各腳本檔頭)會把 CWD 導回 repo root,故可從任意目錄啟動。
- 結果 log 本輪產生於:`_pampa_diag.log`、`_pampa_r2.log`、`_pampa_r3.log`、
  `_pampa_floor.log`、`_pampa_blend.log`、`_pampa_tobit.log`(均被 gitignore)。

## 各腳本內容

| 檔 | 做什麼 | 關鍵輸出 |
|---|---|---|
| `common.py` | 共用:CWD re-root、sys.path、`FEAT_CACHE` 路徑 | — |
| `r2_ceiling.py` | 天花板分解:變異數占比、非地板 R²、地板 ss 占比、天花板 R² 公式 | 0.4642 / 0.6317 / 64% / 0.5387 |
| `round1_feature_ablation.py` | 特徵/目標/架構 ablation(12 組) | 信號定位:信號在 Morgan |
| `round2_rank_gaussian.py` | Rank-Gaussian 目標變換,誠實 train-only mapping,多 seed | 0.434±0.011(否決) |
| `round3_strong_2d.py` | LightGBM 128 超參 × 4 特徵 + ensemble + 兩階段 | 最佳 0.4234;區間 R² 診斷 |
| `floor_predictability.py` | 地板分子分類器 AUC(全特徵/描述子)+ oracle 天花板 sweep | AUC_test 0.8557;oracle 0.807 |
| `soft_blend.py` | 後驗均值 soft blend,β 掃描,val 選參 | 0.4651(+0.0009) |
| `tobit_censored.py` | Tobit 審查似然模型(50 ep,多 seed) | 0.4056±0.024 |
| `tobit_sanity.py` | `_log_phi_cdf`/`nll_tobit` 數值驗證(vs scipy) | ALL PASSED |
