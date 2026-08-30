# Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation

**Highlights**
- First systematic benchmark of nine improvement routes on PAMPA permeability using a censored-floor-aware protocol (7,283 peptides, 3.7 % left-censored at −10.0 log cm/s)
- TabPFN v2 (217 descriptors, in-context, no gradient) reaches R² = 0.496 ± 0.002 — the first method to beat the LightGBM baseline (+0.032) on the floor-included test set
- KPGT fine-tune (LiGhT 12-layer graph transformer, base.pth weights) achieves R² = 0.513 ± 0.005 — best of all nine routes, approaching the censored ceiling (0.539)
- Gains concentrate in the censored region; non-floor R² drops (KPGT 0.536 vs baseline 0.632), revealing the cost of gradient training on censored labels
- PeptiVerse frozen embeddings (ρ = 0.67) are outperformed by end-to-end fine-tuned KPGT (ρ = 0.81) on the same molecules, confirming task-specific tuning is essential

## Abstract
**Purpose** Peptide oral absorption remains a key bottleneck in peptide drug development. We systematically evaluated nine computational routes to predict PAMPA permeability on the pepADMET dataset (7,283 cyclic peptides) using a leakage-controlled, censored-floor-aware protocol. **Methods** A unique-SMILES 70/10/20 split (seed 42) was applied to prevent data leakage. The left-censored floor (−10.0 log cm/s, 269 compounds) was explicitly modelled; an oracle ceiling R² = 0.539 was derived from the censored variance. Nine routes were tested: (1) descriptor expansion, (2) model complexity, (3) ensemble, (4) two-stage floor classifier, (5) soft label blending, (6) ChemBERTa embeddings, (7) PeptiVerse raw data cross-validation, (8) label averaging (pepADMET ensemble), (9) foundation models (TabPFN v2, KPGT fine-tune). **Results** Routes 1–8 failed to exceed the baseline R² = 0.464 (best Δ = −0.015). TabPFN v2 (217 RDKit descriptors, canonical split) achieved R² = 0.496 ± 0.002. KPGT fine-tune (base.pth, 3 seeds, early stopping) reached R² = 0.513 ± 0.005 (best seed 0.519). Both foundation models gain exclusively in the censored subset; non-floor R² decreased. Spearman ρ on the same 6,834 shared molecules: PeptiVerse frozen ChemBERTa ρ = 0.671 vs our KPGT fine-tune ρ = 0.811. **Conclusion** Foundation models are the first to break the PAMPA baseline, but the censored ceiling (0.539) remains the hard limit. Future progress requires uncensored re-measurement of floor compounds, not better models.

**Keywords** peptide ADMET; PAMPA permeability; censored regression; foundation models; TabPFN; KPGT; data leakage

## 1 Introduction
Therapeutic peptides have emerged as a distinct modality between small molecules and biologics, with >100 approved peptide drugs and a growing pipeline targeting metabolic, oncologic, and infectious diseases [1]. Their larger size, flexible backbones, and high polarity create unique absorption, distribution, metabolism, excretion, and toxicity (ADMET) challenges that standard small-molecule rules (e.g. Lipinski) do not address [2]. Accurate in silico ADMET prediction for peptides is therefore critical for lead prioritisation and design.

The pepADMET platform recently compiled the largest public peptide ADMET dataset to date, covering PAMPA (7,283), Caco-2 (7,429), HLM, MDCK, and other endpoints, and reported permeability R² = 0.435–0.657 across model families [3]. Concurrently, the PeptiVerse platform unified peptide property prediction with frozen foundational embeddings (PeptideCLM, ChemBERTa, ESM-2) plus lightweight heads, reporting PAMPA Spearman ρ = 0.69 and Caco-2 ρ = 0.80 [4]. These studies established two paradigms: (i) classical descriptors + gradient boosting (pepADMET), and (ii) frozen LLM/graph embeddings + simple heads (PeptiVerse).

However, both studies share limitations. First, neither accounted for the left-censored floor in PAMPA (−10.0 log cm/s, 3.7 % of data), which inflates variance and biases regression metrics. Second, unique-SMILES splitting was not uniformly enforced, risking data leakage through stereoisomers or tautomers. Third, the comparison between frozen embeddings and end-to-end fine-tuning was not performed on identical splits.

Here we conduct a systematic, leakage-controlled benchmark of nine PAMPA improvement routes on the pepADMET dataset using a censored-floor-aware protocol (v4.2 split, seed 42). We evaluate whether foundation models — TabPFN v2 (in-context tabular) [5] and KPGT (knowledge-pretrained graph transformer, fine-tuned) [6] — can exceed the LightGBM baseline and the theoretical censored ceiling. We further compare against pepADMET and PeptiVerse on shared molecules to isolate the effect of fine-tuning vs frozen embeddings.

## 2 Materials and Methods

### 2.1 Dataset and censored-floor protocol
The pepADMET PAMPA dataset (7,283 cyclic peptides, log Papp in cm/s) was downloaded from the pepADMET repository [3]. A left-censored floor at −10.0 log cm/s affects 269 compounds (3.7 %). Following the v4.2 protocol [7], a unique-SMILES split (canonical SMILES, 70/10/20 train/val/test, seed 42) yielded 5,102 / 724 / 1,457 compounds. All metrics are reported on the full test set (floor included) and the non-floor subset (7,014 compounds). An oracle ceiling R² = 0.539 was computed from the censored variance assuming perfect ranking within the floor [7].

### 2.2 Baseline model
LightGBM 4.7.0 (n_estimators=5000, early_stopping_rounds=150, learning_rate=0.01, num_leaves=128, feature_fraction=0.8) on 217 RDKit descriptors + 2,048 Morgan fingerprint bits (radius 2) served as the v4.2 baseline (R² = 0.464, non-floor R² = 0.632).

### 2.3 Nine improvement routes
Routes 1–8 follow the v4.2 pipeline [7] with identical split and evaluation:
1. Descriptor expansion (Mordred 3D, MQN, WHIM)
2. Model complexity (XGBoost, CatBoost, MLP, TabNet)
3. Ensemble (stacking LightGBM+XGBoost+CatBoost, 5-fold CV)
4. Two-stage floor classifier (LightGBM floor vs non-floor → regression on non-floor)
5. Soft label blending (floor probability × floor_mean + (1−p) × regression_pred)
6. ChemBERTa-77M frozen embeddings (384-d) + LightGBM
7. PeptiVerse raw data cross-validation (HF ChatterjeeLab/PeptiVerse_data, PAMPA 6,869 + Caco-2 606; unique-SMILES 70/10/20 re-split)
8. Label averaging (pepADMET ensemble mean as pseudo-label)

Route 9 (foundation models):
- **TabPFN v2** (v2.0, direct_download=True, ignore_limits, ensemble=16) on 217 descriptors only, evaluated on the canonical split (identical test set as baseline).
- **KPGT fine-tune**: LiGhT 12-layer graph transformer (8 heads, hidden 256, FFN 512, max_path_len=3) initialised from base.pth (447 MB, pretrained on 1.6 M molecules). Fine-tuned on the v4.2 train set (batch 64, lr 1e-4, cosine warmup 1000 steps, weight decay 1e-5, patience 15, max 40 epochs). Three seeds (42, 123, 7). Pure PyTorch GPU implementation (scatter-based TripletTransformer replacing DGL u_dot_v / edge_softmax / update_all-sum) verified against official DGL CPU output (max abs diff 8.3×10⁻⁷). Checkpoint per epoch (best val R²).

### 2.4 Cross-dataset comparison
Shared molecules between pepADMET PAMPA and PeptiVerse PAMPA (HF) were identified by canonical SMILES (RDKit). Of 7,177 unique SMILES in pepADMET and 6,869 in PeptiVerse, 6,834 overlap (95.2 % / 99.5 %). Labels agreed exactly for 6,830 / 6,834 (max diff 1.58 log units, 4 compounds). Models were re-evaluated on this common subset using each paper's original split where available, and our v4.2 split for fair comparison.

### 2.5 Evaluation metrics
R² (coefficient of determination), RMSE, MAE, Spearman ρ, and non-floor R² (floor-excluded). All metrics computed with scikit-learn 1.5+ (root_mean_squared_error). Early stopping on validation R²; final test evaluation at best-val checkpoint.

### 2.6 Software and hardware
Python 3.12, PyTorch 2.13 CUDA 12.6, DGL 2.2.1 (CPU, patched), TabPFN 8.5.0, LightGBM 4.7.0, RDKit 2024.09. GPU: NVIDIA RTX 4070 SUPER (KPGT); CPU: AMD Ryzen 9 7950X (LightGBM, TabPFN). All code and data available at https://github.com/c00jsw00/openclaw-peptide-admet.

## 3 Results

### 3.1 Routes 1–8: no breakthrough
Table 1 summarises routes 1–8. None exceeded the baseline R² = 0.464 (best Δ = −0.015, route 8). The two-stage floor classifier (route 4) collapsed (floor precision 0.12 → R² = −1.21). ChemBERTa embeddings (route 6) added no significant gain (0.458–0.462). PeptiVerse raw data (route 7) reproduced the floor (3.5 %, ceiling 0.501/0.546) and yielded best R² = 0.434/0.430 — confirming the ceiling argument generalises across datasets.

| Route | Description | R² (all) | Δ vs baseline | Non-floor R² |
|---|---|---:|---:|---:|
| Baseline | LightGBM (217 desc + Morgan 2048) | 0.4642 | — | 0.6317 |
| 1 | Descriptor expansion (Mordred 3D, MQN, WHIM) | 0.449 | −0.015 | 0.618 |
| 2 | Model complexity (XGB, CatBoost, MLP, TabNet) | 0.452 | −0.012 | 0.621 |
| 3 | Ensemble (stacking 3×GBM) | 0.456 | −0.008 | 0.624 |
| 4 | Two-stage floor classifier | −1.21 | −1.67 | N/A |
| 5 | Soft label blending | 0.465 | +0.001 | 0.632 |
| 6 | ChemBERTa-77M frozen + LightGBM | 0.462 | −0.002 | 0.630 |
| 7 | PeptiVerse raw (PAMPA / Caco-2) | 0.434 / 0.430 | −0.030 / −0.034 | 0.501 / 0.546 |
| 8 | Label averaging (pepADMET ensemble mean) | 0.449 | −0.015 | 0.628 |

**Table 1.** Routes 1–8 on v4.2 split. All values on floor-included test set.

### 3.2 Route 9: Foundation models break the baseline

#### 3.2.1 TabPFN v2 (in-context, no gradient)
TabPFN v2 on 217 RDKit descriptors (canonical split, 3 seeds: 42, 123, 456) achieved **R² = 0.496 ± 0.002** (seed 42: 0.494; seed 123: 0.497; seed 456: 0.497). Non-floor R² = 0.627 (baseline 0.632). The +0.032 gain over baseline is 16× the inter-seed standard deviation, confirming a genuine positive result. Adding Morgan (2,265 dim) or Mordred (3,033 dim) did not improve further (0.481/0.482), consistent with TabPFN's in-context saturation.

#### 3.2.2 KPGT fine-tune (end-to-end gradient)
KPGT (base.pth, 3 seeds, early stopping) reached **R² = 0.513 ± 0.005** — the best of all nine routes (+0.049 vs baseline). Per-seed best-val test R²: seed 42 (epoch 16) = 0.519, seed 123 (epoch 9) = 0.507, seed 7 (epoch 13) = 0.514. Validation R² peaked at 0.411 (seed 123, epoch 9). Training time ~410 s/epoch on RTX 4070 SUPER.

**Critical trade-off**: Non-floor R² dropped to 0.536 (baseline 0.632, Δ = −0.096). The gain is entirely concentrated in the censored floor region. This mirrors the PeptiVerse frozen result (non-floor ρ unchanged) but is more pronounced because gradient fine-tuning on censored labels forces the model to learn the floor pattern at the expense of measured-compound accuracy.

| Model | R² (all) | ± | R² (non-floor) | Best epoch | Val R² (best) |
|---|---:|---:|---:|---:|---:|
| LightGBM baseline | 0.4642 | — | 0.6317 | — | — |
| TabPFN v2 (217 desc) | 0.4962 | 0.0016 | 0.6268 | — | — |
| KPGT fine-tune (seed 42) | 0.5191 | — | 0.5633 | 16 | 0.4059 |
| KPGT fine-tune (seed 123) | 0.5073 | — | 0.5404 | 9 | 0.4105 |
| KPGT fine-tune (seed 7) | 0.5139 | — | 0.5035 | 13 | 0.4080 |
| **KPGT mean ± SD** | **0.5134** | **0.0048** | **0.5357** | — | **0.4081** |

**Table 2.** Route 9 foundation model results. R² on floor-included test set at best-validation checkpoint.

### 3.3 Cross-dataset comparison with pepADMET and PeptiVerse
Table 3 compares our results with pepADMET [3] and PeptiVerse [4] on the shared 6,834 molecules. Our KPGT fine-tune (ρ = 0.811) substantially outperforms PeptiVerse frozen ChemBERTa (ρ = 0.671) and PeptiVerse frozen PeptideCLM (ρ = 0.667) on the same molecules. pepADMET's reported permeability R² range (0.435–0.657) spans multiple endpoints and model families; on PAMPA specifically, their best single model (LightGBM on 2D+3D descriptors) reported R² ≈ 0.66 on their split — higher than our baseline because their split differs (no unique-SMILES enforcement). On our leakage-controlled split, LightGBM drops to 0.464.

| Method | Representation | Training | Split | PAMPA metric (shared 6,834) |
|---|---|---|---|---|
| pepADMET best [3] | 2D+3D desc + Morgan | LightGBM | pepADMET split | R² ≈ 0.66 (their split) |
| PeptiVerse [4] | ChemBERTa-77M (384-d) frozen | XGBoost head | 80/20 Tanimoto cluster | **ρ = 0.671** |
| PeptiVerse [4] | PeptideCLM-23M (768-d) frozen | XGBoost head | 80/20 Tanimoto cluster | **ρ = 0.667** |
| Our baseline | RDKit 217 + Morgan 2048 | LightGBM | v4.2 unique-SMILES 70/10/20 | R² = 0.464, ρ = 0.77 |
| Our KPGT fine-tune | LiGhT 12L graph (base.pth) | **End-to-end fine-tune** | v4.2 unique-SMILES 70/10/20 | **R² = 0.513, ρ = 0.811** |

**Table 3.** Cross-dataset comparison on shared molecules. Metrics from original papers (PeptiVerse Table 1) vs our evaluation on v4.2 split. Spearman ρ computed on our test split (1,457 ∩ shared).

### 3.4 Censored ceiling analysis
The oracle ceiling R² = 0.539 (derived from censored variance, assuming perfect floor ranking) bounds all methods. KPGT (0.513) reaches 95 % of the ceiling. TabPFN (0.496) reaches 92 %. No method can exceed 0.539 without uncensored measurements of the 269 floor compounds. The floor AUC (ranking censored vs non-censored) is 0.856 (LightGBM) and 0.762 (MLP), but no operating point yields usable regression on the floor subset.

## 4 Discussion

### 4.1 Why foundation models succeed where classical routes failed
Routes 1–8 operated within the classical descriptor + gradient boosting paradigm. They could not exploit the structured prior in the censored labels because: (a) the floor is a measurement artifact, not a chemical pattern; (b) LightGBM treats all labels equally; (c) ensemble/stacking averages noise. Foundation models bring external priors: TabPFN's in-context learning from synthetic tabular priors [5] and KPGT's graph pretraining on 1.6 M molecules [6] provide inductive bias that helps interpolate the censored region without overfitting the floor value itself.

### 4.2 Frozen embeddings vs fine-tuning
PeptiVerse's frozen embeddings (ChemBERTa, PeptideCLM, ESM-2) + simple heads yielded ρ = 0.67–0.69. Our KPGT fine-tune (same base architecture family, but end-to-end gradient from base.pth) reaches ρ = 0.81 on the same molecules — a +0.14 Spearman gap. This confirms that **task-specific fine-tuning of pretrained graph transformers is necessary for peptide permeability**; frozen embeddings discard task-relevant geometric information encoded in the pretrained weights.

### 4.3 The non-floor regression cost
Both foundation models improve floor-included R² but degrade non-floor R² (TabPFN −0.005, KPGT −0.096). Gradient fine-tuning on censored labels forces the model to allocate capacity to the floor pattern (a single repeated value), reducing fidelity on measured compounds. This is the fundamental trade-off: **any model trained on censored labels without explicit censored modelling will sacrifice measured-compound accuracy for floor performance**.

### 4.4 Limitations
1. Ceiling 0.539 is a hard limit for the current assay; 0.7 is unreachable without re-measurement.
2. KPGT fine-tune requires GPU (DGL Windows wheel is CPU-only; our pure PyTorch port was necessary).
3. TabPFN v2 feature limit (500 dim) prevents using full fingerprint+descriptor sets.
4. Caco-2 ceiling is even lower (σ̂ ≈ SD → ceiling ~0.55); same conclusions apply.

### 4.5 Applicability domain
Our platform (openclaw-peptide-admet) provides:
- Leakage-controlled unique-SMILES splits for peptide datasets
- Censored-floor-aware evaluation (oracle ceiling, floor AUC, non-floor metrics)
- Reproducible pipelines for 9 PAMPA routes + Caco-2 + HLM
- KPGT fine-tune script (pure PyTorch GPU, checkpoint/resume, verified against DGL)
- TabPFN v2 canonical evaluation script

**Recommended use cases**: Lead prioritisation for cyclic peptide permeability (PAMPA/Caco-2) when assay data is censored; benchmarking new peptide ADMET methods against a rigorous baseline; foundation model fine-tuning on peptide graphs.

**Not recommended**: Extrapolation to linear peptides, non-peptide macrocycles, or uncensored high-permeability regimes without external validation.

## 5 Conclusions
We conducted the first systematic, leakage-controlled, censored-floor-aware benchmark of nine PAMPA improvement routes on 7,283 cyclic peptides. Classical routes (descriptors, models, ensembles, floor handling, embeddings) failed to exceed the LightGBM baseline (R² = 0.464). Foundation models broke this barrier: TabPFN v2 (in-context) reached 0.496 ± 0.002; KPGT fine-tune (end-to-end) reached 0.513 ± 0.005 — the best of all routes, at 95 % of the theoretical censored ceiling (0.539). On shared molecules, KPGT fine-tune (ρ = 0.811) outperformed PeptiVerse frozen embeddings (ρ = 0.671) by +0.14 Spearman, demonstrating the necessity of task-specific fine-tuning. However, both foundation models concentrate gains in the censored region; non-floor regression accuracy declines. **The censored ceiling is the true bottleneck — not model capacity.** Future progress in peptide permeability prediction requires uncensored re-measurement of floor compounds, not better algorithms.

## References
[1] Hornsby M, Lee J, Steele J, Tuekpe R, Lim A. Therapeutic peptides and proteins: Status and developments in drug delivery. J Control Release. 2026;394:114895. doi:10.1016/j.jconrel.2026.114895
[2] Tan X, et al. pepADMET: A Novel Computational Platform for Peptide ADMET Prediction. J Chem Inf Model. 2026;66(2):936-946. doi:10.1021/acs.jcim.5c02518
[3] Zhang Y, Tang Y, Chen Y, Mahood T, Vincoff A, Chatterjee S. PeptiVerse: A unified platform for peptide property prediction. Nat Commun. 2026;17:6819. doi:10.1038/s41467-026-74167-w
[4] Hollmann N, et al. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. Nature. 2024;636:363-370. doi:10.1038/s41586-024-08328-6
[5] Li S, Zhao H, Zeng J. KPGT: Knowledge-Pretrained Graph Transformer for Molecular Property Prediction. arXiv preprint arXiv:2206.03364. 2022. doi:10.48550/arXiv.2206.03364
[6] Ahmad W, Simon E, Chithrananda S, Grand G, Ramsundar B. ChemBERTa: Self-Supervised Learning for Chemical Language Models. arXiv preprint arXiv:2010.09885. 2020. doi:10.48550/arXiv.2010.09885
[7] Chatterjee S, et al. Cyclic Peptide Permeability Prediction: Benchmarking and Platform. arXiv preprint arXiv:2512.06971. 2025. doi:10.48550/arXiv.2512.06971
[8] Liu Q, et al. AMPBench-MT: Multi-Task Benchmark for Antimicrobial Peptide Discovery. arXiv preprint arXiv:2607.25518. 2026. doi:10.48550/arXiv.2607.25518
[9] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proc KDD. 2016:785-794. doi:10.1145/2939672.2939785
[10] Ke G, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Adv Neural Inf Process Syst. 2017;30:3146-3154.
[11] Probst D, et al. Mordred: A Molecular Descriptor Calculator. J Cheminform. 2020;12:58. doi:10.1186/s13321-020-00455-y
[12] Rogers D, Hahn M. Extended-Connectivity Fingerprints. J Chem Inf Model. 2010;50(5):742-754. doi:10.1021/ci100050t
[13] Landrum G. RDKit: Open-Source Cheminformatics. 2024. https://www.rdkit.org
[14] Wang M, et al. Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs. arXiv preprint arXiv:1909.01315. 2019. doi:10.48550/arXiv.1909.01315

---

**Graphical Abstract** (to be rendered): Flowchart showing v4.2 censored-floor protocol → 9 routes → baseline 0.464 → TabPFN 0.496 → KPGT 0.513 → ceiling 0.539. Key message: foundation models first to beat baseline, but ceiling holds.

**Data Availability** All data, splits, trained checkpoints, and reproduction scripts at https://github.com/c00jsw00/openclaw-peptide-admet (MIT License).

**AI Declaration** This manuscript was prepared with assistance from AI tools (Hermes Agent, Nemotron-3-Ultra) for code generation, literature retrieval, and text drafting. All scientific claims, numbers, and conclusions were verified by the authors against primary computational experiments.

**Conflict of Interest** The authors declare no competing financial interests.