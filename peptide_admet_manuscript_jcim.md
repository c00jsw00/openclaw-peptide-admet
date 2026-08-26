# From Synthetic Demo to Real Data: A Reproducible, Leakage-Audited Benchmark for Four Peptide ADMET Endpoints on the PepADMET Dataset, with Frozen ESMC Sequence Embeddings and Frozen MoLFormer Molecular Embeddings

**Running Title:** Leakage-Audited Peptide ADMET Benchmark on Real Data, with Frozen ESMC + MoLFormer Embeddings

**Article Type:** Article (Full-Length Research Manuscript)

> **⚠️ v4.2 repository note (2026-08-26).** This manuscript was updated in
> place for the **v4.2 real-data + ESMC + MoLFormer edition**, which builds on
> the v4.1 real-data + ESMC edition, which builds on the v4.0 real-data
> edition. v4.0 had removed the v3.0 **30,000-row synthetic demo** and re-run
> the leakage-audited protocol on **real experimental data** from the
> [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset)
> release over **four endpoints** — Hemolysis, plasma Half-life, Caco-2
> permeability, and PAMPA/MDCK permeability — with a **dual-modality** design
> (428-dim amino-acid-sequence features for the sequence endpoints; 2,265-dim
> RDKit descriptors + Morgan fingerprints for the SMILES-only endpoints).
> v4.1 added a **frozen ESMC-600M (ESM Cambrian, Biohub) protein-language-model
> embedding** to the *sequence* endpoints only (1,152-dim mean-pooled vector
> concatenated to the 428-dim classical features → 1,580-dim input).
>
> **v4.2 makes three changes:**
> 1. **Molecular endpoints (Caco-2, PAMPA/MDCK) gain a frozen MoLFormer-XL
>    molecular-transformer embedding.** The IBM MoLFormer-XL model
>    (`ibm-research/MoLFormer-XL-both-10pct`, 60M parameters, hidden 768) is
>    run in inference-only mode over each SMILES and its **CLS token** (768-dim)
>    is concatenated to the 2,265-dim RDKit features → a **3,033-dim** model
>    input. As with ESMC, the embedding is **frozen** (no fine-tuning, no
>    gradient), precomputed once in the main Python environment, and shipped as
>    a committed `npz` cache so retraining and cached-SMILES inference need no
>    MoLFormer dependency.
> 2. **Half-life deduplication.** The 1,763 raw half-life rows contain only
>    **768 unique sequences** (995 rows are repeat measurements; the most
>    repeated sequence was measured 82 times). v4.2 aggregates the log10
>    half-life values of each repeated sequence to their **mean** before
>    splitting, so the headline metric is reported at the **sequence level**
>    (768 unique sequences) rather than the row level. This removes
>    irreducible experimental re-measurement noise that a row-level model could
>    never fit.
> 3. **Huber loss for all regression endpoints.** The three continuous
>    endpoints (half-life, Caco-2, PAMPA) now train with a Huber loss instead
>    of MSE — more robust to the 1–2 log-unit outliers present in real
>    permeability and half-life measurements.
>
> The two molecular endpoints' native "sequence" column is a non-standard
> peptidomimetic residue list (~0.2–0.3% standard amino acids) that a 20-AA
> protein language model cannot consume, so they remain on the molecular
> (SMILES) path — now enriched with MoLFormer rather than ESMC. All performance
> figures below are *measured* and re-derivable from the released code
> (`metrics.json`); the v2.0/v3.0 synthetic-demo figures are no longer the
> headline, and the v4.0 (pre-ESMC) and v4.1 (pre-MoLFormer) sequence numbers
> are retained as the baselines against which the v4.2 gains are stated.

**Version:** 4.2 (real-data + ESMC-600M + MoLFormer-XL edition, 2026-08-26).
This revision builds on the v4.1 real-data + ESMC edition (2026-08-26), which
builds on the v4.0 real-data edition (2026-08-25), which replaced the v3.0
manuscript — an integrity revision (replacing v1.0's hardcoded 97.70% / 0.9987
AUC) validated on a synthetic demo set. The evaluation protocol
(homology-controlled split + leakage audit + dual-split reporting) is
unchanged; v4.2 extends the molecular-endpoint feature design with a frozen
MoLFormer embedding, deduplicates the half-life target, and switches regression
to Huber loss.

---

## Abstract

Peptide-based therapeutics are among the fastest-growing pharmaceutical classes, yet computational ADMET prediction for peptides is plagued by an evaluation-integrity problem that 2026 benchmark work has begun to expose: sequence-similarity leakage across the train/test boundary inflates reported performance, and several public tools report metrics that cannot be reproduced from their released artifacts. Prior versions of this repository validated a reproducibility protocol — an AMPBench-MT-style homology-controlled split with a shipped leakage audit, dual-split reporting, and measured-only inference — but did so on a *synthetic* demo set, which can certify the pipeline yet cannot claim real-peptide accuracy. Here we close that gap: we re-run the identical protocol on **real experimental peptide data** (the Chemit797/PepADMET-Dataset release) over **four ADMET endpoints** (Hemolysis [binary], plasma half-life [regression], and Caco-2 and PAMPA/MDCK permeability [regression]). Because the four endpoint tables are disjoint molecule sets spanning two input modalities, we use a **dual-modality** design: 428-dim amino-acid-composition features (AAC 20 + DPC 400 + physicochemical 8) **concatenated to a frozen 1,152-dim ESMC-600M protein-language-model embedding** (a 1,580-dim input) for the two sequence endpoints, and 2,265-dim RDKit features (217 2D descriptors + 2,048-bit Morgan fingerprint) **concatenated to a frozen 768-dim MoLFormer-XL molecular-transformer CLS embedding** (a 3,033-dim input) for the two SMILES-only permeability endpoints, each with an independent single-task MLP trained with a Huber loss. Both language-model embeddings are **frozen** (no fine-tuning, no gradient), precomputed once, and shipped as committed `npz` caches. The half-life endpoint is additionally **deduplicated** from 1,763 rows to 768 unique sequences (repeat measurements averaged) so its headline metric is reported at the sequence level. On the leakage-controlled test split the models achieve **AUC 0.8348 (Hemolysis, +0.059 over the pre-ESMC v4.0 0.7755)**, **R² 0.7259 (half-life, log10-seconds, sequence-level; +0.029 over the row-level v4.1 0.6973 after dedup + Huber)**, **R² 0.3909 (Caco-2, logPapp; +0.005 over the RDKit-only v4.1 0.3861 after MoLFormer + Huber)**, and **R² 0.4642 (PAMPA, logPapp; +0.007 over 0.4573)**. The dual-split comparison on half-life now shows a **random-vs-controlled delta of +0.061** (0.7867 random vs 0.7259 controlled), down from +0.176 in v4.1 — a direct consequence of deduplication removing the near-duplicate rows that drove the inflation, not a loss of control. We state plainly that the molecular-endpoint MoLFormer gain (+0.005 to +0.007) is *within* the ±0.01 run-to-run retraining noise we measured, so the honest conclusion is that **the molecular-endpoint bottleneck is label noise (0.6–0.94 log-unit spread across repeat measurements of the same molecule), not the 2D-descriptor representation layer**; a frozen molecular transformer does not break that noise floor. Every number is measured and re-derivable from the released code and data.

**Keywords:** peptide ADMET prediction | benchmark | evaluation leakage | homology-controlled split | dual-modality features | RDKit descriptors | ESMC-600M | MoLFormer | frozen embeddings | Huber loss | target deduplication | reproducibility | Caco-2 permeability | PAMPA | hemolysis | half-life

---

## Graphical Abstract and Table of Contents Entry

**Table of Contents Graphic:** [real peptide table (sequence ∪ SMILES) → dual-modality features (1,580-dim sequence: 428-dim classical + 1,152-dim frozen ESMC-600M / 3,033-dim molecular: 2,265-dim RDKit + 768-dim frozen MoLFormer-XL CLS) → four single-task MLPs (Huber regression) → 4 ADMET predictions; inset: leakage audit — exact-anagram collapse + max cross-boundary Jaccard + half-life dedup 1,763 → 768]

**TOC Entry:** A leakage-audited protocol, first validated on a synthetic demo and re-run on real PepADMET data over four dual-modality endpoints — with frozen ESMC-600M protein embeddings boosting the sequence endpoints and frozen MoLFormer-XL molecular embeddings added to the permeability endpoints, plus half-life target deduplication — yields reproducible peptide ADMET numbers (AUC 0.835; R² 0.391–0.726) and a template for how the field should report them.

---

## 1. Introduction

### 1.1 Background

Peptide therapeutics have emerged as a promising class of drugs, with over 90 approved peptide drugs and hundreds more in clinical development. Their high specificity, potency, and favorable safety profiles have driven extensive research, while the development of peptide drugs faces unique challenges that differ from small-molecule drug development:

1. **Poor Oral Bioavailability**: Peptides typically exhibit low gastrointestinal absorption due to large molecular size (>500 Da), high polarity, and enzymatic degradation in the digestive tract.
2. **Membrane Permeability Limitations**: The polar nature of peptide bonds and side chains limits passive diffusion across biological membranes; Caco-2 and PAMPA are the two standard *in vitro* permeability assays used to model it.
3. **Metabolic Instability**: Peptides are rapidly degraded by proteases and peptidases, leading to short plasma half-lives.
4. **Hemolysis**: Many bioactive and antimicrobial peptides lyse erythrocytes, a dose-limiting toxicity that must be screened early.
5. **Rapid Renal Clearance**: Small peptides (<5 kDa) are efficiently filtered by the kidneys, reducing systemic exposure.

### 1.2 The Evaluation-Integrity Problem in Peptide ADMET

Beyond the biological challenges, computational peptide ADMET has a methodological problem that recent 2026 work has brought to the foreground:

- **Sequence-similarity leakage.** AMPBench-MT (arXiv:2607.25518), a 2026 multi-task benchmark for antimicrobial peptides, shows that when near-duplicate or compositionally similar sequences fall on both sides of the train/test boundary, reported accuracy and AUC can be inflated far beyond what the model could achieve on genuinely novel sequences. Homology-aware (or composition-family-aware) splitting is the currently recommended remedy, but few public peptide-ADMET repositories audit their splits or release the audit. This problem is *sharper* on real data than on a synthetic demo: real peptide libraries contain many near-duplicate and anagrammatic sequences that a naive random split will happily split across the boundary. **A second, related integrity problem** — one this work addresses directly in v4.2 — is *label multiplicity*: when the same molecule or sequence is measured several times with irreproducible results, a row-level model is forced to fit irreducible noise, and a naive split can place the repeat measurements on both sides of the boundary, inflating R². We address this by deduplicating the half-life target to unique sequences.
- **Non-reproducible metrics.** We observed, in our own v1.0 submission package, metrics that were hardcoded in the inference CLI and inconsistent with the shipped model artifacts, and a "training dataset" referenced in documentation that was not present in the repository.
- **Modality mismatch.** Real ADMET datasets are rarely uniform: some tables ship clean one-letter sequences, others ship only SMILES (or, as in the CycPeptMPDB-derived permeability tables here, *non-standard* residue-name lists such as `MEL`, `DP`, `DL` that a 20-amino-acid encoder cannot consume). A benchmark must state, per endpoint, what input it actually uses and why.

### 1.3 Prior Work (condensed)

Classical tools (AdmetSAR 2.0, SwissADME, ADMETlab 3.0, pepADMET) offer peptide ADMET predictions but were primarily optimized for small molecules or validated on small, non-external test sets. Deep approaches (LSTM, Transformer, GNN, protein language models) capture sequence order but demand large datasets and GPU resources. Handcrafted-composition QSAR and RDKit descriptor models remain competitive, particularly at modest data scale. Pretrained molecular transformers (MoLFormer and its successors) and protein language models (ESM-2, ESMC/ESM Cambrian) offer frozen or fine-tunable embeddings that can be concatenated to classical features as a static prior. 2026 has added generative-redesign pipelines for antibiotics (ApexGO, *Nature Machine Intelligence*, 2026-05) and integrated agentic pipelines for peptide campaigns (*npj Drug Discovery*, 2026-05), all of which converge on the same methodological point: **report what was measured, on which split, with which leakage controls.**

### 1.4 Study Objectives

1. Replace the synthetic demo set with a **real, permissively-licensed experimental dataset** (the Chemit797/PepADMET-Dataset release) and keep every shipped artifact present in the repository (prepared data, weights, scalers, metrics) so nothing referenced in documentation can be silently missing.
2. Implement a **leakage-audited split per modality**: an AMPBench-MT-style homology-controlled split with **exact-anagram collapse** for sequence endpoints (guaranteeing no jaccard-1.0 duplicate crosses the boundary), and a unique-SMILES grouping split for the SMILES-only endpoints (with the near-isomer limitation stated explicitly).
3. Train and release a **dual-modality** model family — the *same* architecture class in trainer and predictor — with the exact weights, scalers, and measured metrics in `metrics.json`.
4. **v4.1:** enrich the two sequence endpoints with a **frozen ESMC-600M protein-language-model embedding** (1,152-dim, attention-mask mean-pooled; no fine-tuning) and quantify — under the *same* leakage-controlled protocol — how much of the remaining gap to the v2.0 simulation targets a pretrained language-model prior recovers, without any gradient update to the embedding.
5. **v4.2:** (a) enrich the two molecular (permeability) endpoints with a **frozen MoLFormer-XL molecular-transformer CLS embedding** (768-dim; no fine-tuning) and quantify the gain under the *same* protocol; (b) **deduplicate the half-life target** from 1,763 rows to 768 unique sequences (repeat measurements averaged) so the headline metric is reported at the sequence level; (c) switch all regression endpoints to a **Huber loss** for robustness to log-space outliers.
6. Report **measured** per-endpoint AUC/MCC/accuracy (binary) and R²/RMSE/MAE (regression) on the leakage-controlled split (headline) and a random split (leakage comparison), and state plainly what the numbers do and do not claim.

### 1.5 Significance

**Methodological**: a reference implementation of the evaluation protocol (per-modality leakage audit + dual-split reporting + measured-metrics-only inference) applied to *real* data — the step the synthetic-demo version could not take. v4.2 adds a third integrity control: **target deduplication**, which addresses the label-multiplicity form of leakage that a pure sequence/SMILES homology split does not capture.
**Reproducibility**: every number in this manuscript is a function of released code and released data; `prepare_pepadmet_data.py → {esmc,molformer}_embed.py → train_pepadmet_model.py → peptide_admet_predictor.py` regenerates the pipeline end-to-end on CPU.
**Honesty**: we report AUC 0.8348 and R² 0.3909–0.7259, not 0.9987. The random-vs-controlled delta on half-life is the *finding*: in v4.1 (row-level) it was +0.176 (0.8733 random vs 0.6973 controlled); in v4.2 (sequence-level, after dedup) it is +0.061 (0.7867 vs 0.7259) — the shrinkage is itself the evidence that the inflation was driven by near-duplicate rows, and that deduplication removes it. The ESMC gain (v4.1) and the MoLFormer gain (v4.2) are both reported *within* the same controlled protocol, so neither improvement can be attributed to a split change. We state plainly that the MoLFormer molecular gain (+0.005 to +0.007) is within the ±0.01 retraining noise, so the molecular-endpoint bottleneck is label noise, not the representation layer.

---

## 2. Materials and Methods

### 2.1 Data (real experimental dataset, provenance stated)

We use the **Chemit797/PepADMET-Dataset** release (its cleaned `整理/` sub-directory), which provides four endpoint tables:

| Endpoint | Source table | Input | Label | Rows (prepared) |
|---|---|---|---|---|
| Hemolysis | `hemolysis_unified/hemolysis_unified.csv` | `sequence_std` (one-letter 20-AA) | `label` (0/1) | 8,719 |
| Half-life | `half_life_*/...` | `sequence` (one-letter 20-AA) | `half_life_seconds` (continuous) | 1,763 (→ 768 unique, v4.2) |
| Caco-2 | `caco2_*/...` | `SMILES` (valid) | `Permeability` (logPapp) | 7,429 |
| PAMPA/MDCK | `pampa_mdck_*/...` | `SMILES` (valid) | `PAMPA` (logPapp) | 7,283 |

The four tables are **disjoint molecule sets** (no compound appears in more than one), so no multi-task shared-label structure exists; each endpoint is an independent single-task problem. `prepare_pepadmet_data.py` loads each table, validates the input (sequence endpoints: rows must contain a clean 20-AA one-letter sequence; molecular endpoints: the SMILES must parse under RDKit), drops rows with a missing/non-finite label, and writes a per-endpoint prepared CSV plus a provenance/`meta.json` recording source path, row counts, and the exact dropped-row statistics. No synthetic labels are generated; every label is taken verbatim from the source table.

**Half-life target transform and deduplication (v4.2).** The raw half-life spans ~10⁻³ to ~10⁹ s (log10 range −3.1 to 9.1). We model **log10(seconds)** as the regression target and report R² in that space; the inference CLI inverts to seconds for display. The 1,763 prepared rows contain only **768 unique sequences**: 995 rows are repeat measurements of a sequence already seen (the most repeated sequence was measured 82 times), and the repeat measurements of the same sequence disagree by up to ~1.8 log10 units (mean ~0.74). Because a single deterministic model cannot fit irreproducible repeats, v4.2 aggregates the log10 values of each repeated sequence to their **mean**, yielding 768 unique (sequence, target) pairs; the headline metric is reported at this **sequence level**. The prepared CSV (`data/pepadmet_half_life.csv`) still holds all 1,763 rows for provenance; the deduplication happens inside `train_pepadmet_model.py` and is recorded in `metrics.json` (`dedup_info`). Caco-2 and PAMPA permeability are already log-scale (logPapp) and are modeled directly; their SMILES duplication rate is far lower (52 and 104 exact-duplicate SMILES respectively), so no target aggregation is applied there.

### 2.2 Feature Engineering (dual-modality, both modalities now embedding-augmented)

Identical in training and inference (single shared implementation in `feature_extractor.py` + the two embedding generators):

**Sequence modality (1,580-dim = 428-dim classical + 1,152-dim ESMC-600M)** — Hemolysis, Half-life:
1. **Amino Acid Composition (AAC)** — 20: frequency of each standard amino acid.
2. **Dipeptide Composition (DPC)** — 400: frequency of every ordered dipeptide.
3. **Physicochemical** — 8: estimated MW, average Kyte–Doolittle hydropathy, net charge at pH 7, pI estimate, GRAVY, hydrophobic/charged-residue ratios.
4. **Frozen ESMC-600M embedding (v4.1)** — 1,152: the ESM Cambrian-600M protein language model (Biohub; ~574M parameters) is run in **inference-only** mode — no fine-tuning, no gradient — over each 20-AA one-letter sequence, and its per-token hidden states are aggregated with an attention-mask-weighted mean pool to a single 1,152-dim vector. Embeddings are precomputed once per prepared dataset (`esmc_embed.py`, in a dedicated Python ≥ 3.12 environment) and cached to committed `npz` files (`data/esmc/esmc_emb_<endpoint>.npz`); retraining and cached-sequence inference never load the 574M-parameter model. The frozen vector is concatenated to the 428-dim classical features before Z-score standardization, giving a 1,580-dim model input. Because the embedding is frozen and precomputed, the added cost at training time is zero beyond one `npz` load; at inference, a cache miss for a genuinely new sequence triggers an on-demand ESMC subprocess (seconds to load, sub-second per short peptide).

**Molecular modality (3,033-dim = 2,265-dim RDKit + 768-dim MoLFormer-XL, v4.2)** — Caco-2, PAMPA/MDCK:
1. **RDKit 2D descriptors** — 217: `rdkit.Chem.Descriptors.CalcMolDescriptors(mol)` (a fixed, deterministic registry).
2. **Morgan fingerprint** — 2,048 bits, radius 2.
3. **Frozen MoLFormer-XL embedding (v4.2)** — 768: the IBM MoLFormer-XL molecular transformer (`ibm-research/MoLFormer-XL-both-10pct`; 60M parameters, hidden 768, 12 layers, 6 heads) is run in **inference-only** mode — no fine-tuning, no gradient — over each SMILES, and its **CLS token** is taken as the 768-dim embedding. Embeddings are precomputed once per prepared dataset (`molformer_embed.py`, in the main Python environment — no separate 3.12 env needed) and cached to committed `npz` files (`data/molformer/molformer_emb_<endpoint>.npz`); retraining and cached-SMILES inference never load the 60M-parameter model. The frozen vector is concatenated to the 2,265-dim RDKit features before Z-score standardization, giving a 3,033-dim model input. At inference, a cache miss for a genuinely new SMILES triggers an on-demand MoLFormer subprocess in the main env.

The two permeability tables' native "sequence" column is a non-standard residue-name list (`MEL`, `DP`, `DL`, `ME_DL`, …) from CycPeptMPDB that a 20-AA encoder cannot consume; we therefore use the (valid) SMILES column and state this per endpoint in `endpoint_config.py`. A SMILES that fails RDKit parsing yields an all-zero molecular row and is counted in the preparation statistics (not silently fabricated). All features are Z-score standardized with a scaler fit on the training split only.

> **Why a molecular transformer, not ESMC, for the molecular endpoints.** ESMC is a *protein* language model: it consumes 20-AA one-letter sequences and cannot parse SMILES or the non-standard peptidomimetic residue lists in these tables. MoLFormer, by contrast, is a *molecular* transformer pre-trained on SMILES, so it is the natural frozen embedding for the SMILES-only endpoints. Both are used in the same frozen, precomputed, cache-backed manner.

### 2.3 Model

A single MLP class (`MixedADMETMLP` in `admet_model.py`) is instantiated per endpoint — the *same* class in trainer and predictor, so the two can never drift apart:

- Input `d` → Linear 256 → BatchNorm → ReLU → Dropout(0.2) → Linear 128 → BatchNorm → ReLU → Dropout(0.2) → a single task head (`Linear(128,1)`; sigmoid for Hemolysis, identity for the three regressions).
- Parameter counts (v4.2): **438,529** (sequence endpoints, d = 1,580 = 428 classical + 1,152 ESMC; unchanged from v4.1) and **810,497** (molecular endpoints, d = 3,033 = 2,265 RDKit + 768 MoLFormer; up from v4.1's 613,889 at d = 2,265). Loss: BCE-with-logits (Hemolysis, pos-weight = train/neg ratio) or **Huber** (the three regressions; v4.2 — more robust to the 1–2 log-unit outliers in real permeability and half-life data; v4.1 and earlier used MSE). Optimizer: Adam (lr 1e-3, weight-decay 1e-5), `ReduceLROnPlateau` (factor 0.5, patience 4), early stopping on the validation objective (patience 10). Trained on CPU. No ensemble; no Random Forest; both the ESMC and MoLFormer embeddings carry no trainable parameters. The `hidden`/`dropout` architecture is persisted in each checkpoint so the predictor can rebuild the exact model (backward-compatible with v4.1 checkpoints that predate persistence).

### 2.4 Leakage-Controlled Splitting with Audit (per modality)

**Sequence endpoints (Hemolysis, Half-life).** Following AMPBench-MT (arXiv:2607.25518): (1) each sequence is reduced to a **canonical 3-mer-multiset signature** (a count vector over its 3-mers); two sequences with identical 3-mer multisets have 3-mer Jaccard = 1.0, so collapsing by signature *guarantees* no exact-jaccard-1.0 duplicate (including length-preserving anagrams) is ever placed on both sides of the boundary; (2) the unique signatures are clustered by greedy single-linkage 3-mer Jaccard (threshold 0.35); (3) **families** — not sequences — are allocated to train/val/test at 70/10/20; (4) a **leakage audit** is shipped with the split: maximum audited cross-boundary 3-mer Jaccard and the per-endpoint label-rate delta. In our v4.2 run the max cross-boundary Jaccard is ≈0.968 (Hemolysis) — the expected near-duplicate ceiling under a controlled split, with exact-multiset leakage **guaranteed 0**. For Half-life, v4.2 deduplicates to 768 unique sequences *before* the homology split (533/76/159 train/val/test), so both the controlled and random splits are at the sequence level and the anagram near-duplicate inflation of the row-level v4.1 protocol is removed.

**Molecular endpoints (Caco-2, PAMPA/MDCK).** No sequence is available, so a 3-mer homology control is impossible. We instead group by **unique SMILES** (exact-duplicate SMILES share one split) and draw a 70/10/20 split over unique SMILES. We state the limitation explicitly: **near-isomeric structures** (different SMILES strings, same chemistry) can cross the boundary — a real-data limitation of SMILES-only data, weaker than the sequence homology control. This is recorded in each endpoint's `metrics.json` audit.

As a **leakage comparison** (sequence endpoints only), the identical model is trained on a plain random 70/10/20 split (over the same deduplicated set for Half-life); the delta between the random and controlled test metrics quantifies the leakage the random protocol would have reported.

### 2.5 Evaluation

Per endpoint, computed on the held-out test split only and written to `metrics.json`: binary endpoints report AUC-ROC, MCC, and accuracy at threshold 0.5; regression endpoints report R², RMSE, and MAE (in the modeling space: log10-seconds for half-life, logPapp for permeability). Headline numbers are the leakage-controlled test metrics; the random-split numbers are the leakage comparison. Regression uses the Huber loss for training; evaluation metrics are unchanged (R²/RMSE/MAE in log-space).

### 2.6 Multi-Objective Composite Score (removed in v4.0)

The v3.0 composite score (geometric mean over favorable endpoint probabilities) assumed five *binary* endpoints of mixed favorability. With the v4.0 set — one binary plus three continuous regression endpoints on disjoint molecules — a single geometric-mean score is not meaningful, so it is **removed**; the predictor reports each endpoint's value in its own units.

---

## 3. Results

### 3.1 Measured Performance (leakage-controlled test split, headline)

All values below are taken from `models_v4/<endpoint>/metrics.json` produced by the released training script (seed 42, 80 epochs, early-stopped, single complete run).

| Endpoint | Kind | Modality | Test (n) | Primary | Other |
|---|---|---|---|---|---|
| Hemolysis | binary | sequence + ESMC | 1,745 | AUC **0.8348** | MCC 0.4557, Acc 0.7479 |
| Half-life | regression (log10 s) | sequence + ESMC | 159 | R² **0.7259** | RMSE 1.3651, MAE 0.8866 |
| Caco-2 | regression (logPapp) | molecular + MoLFormer | 1,490 | R² **0.3909** | RMSE 0.7848, MAE 0.4708 |
| PAMPA/MDCK | regression (logPapp) | molecular + MoLFormer | 1,457 | R² **0.4642** | RMSE 0.7991, MAE 0.4500 |

The Hemolysis row is bit-identical to v4.1 (same data, same split, same features, deterministic retraining reproduces 0.8348 exactly). The Half-life row reflects the v4.2 dedup (1,763 → 768 unique sequences, sequence-level test n = 159) + Huber. The two molecular rows reflect the v4.2 MoLFormer concatenation (2,265 → 3,033-dim) + Huber.

### 3.2 Dual-Split Comparison (the leakage question, now at the sequence level)

| Endpoint | Controlled test | Random test | Delta (random − controlled) |
|---|---|---|---|
| Hemolysis (AUC) | 0.8348 | 0.8112 | −0.0236 |
| Half-life (R², log10 s, sequence-level, 768 unique) | 0.7259 | 0.7867 | **+0.0608** |
| Caco-2 / PAMPA | — | — | (no sequence; unique-SMILES split only) |

The half-life delta is the central real-data demonstration, and it has *changed meaning* between v4.1 and v4.2. In v4.1 (row-level, 1,763 rows) a plain random split reported **R² 0.8733** versus the controlled **0.6973** — a **+0.1760** delta, the near-duplicate and near-anagram leakage a naive protocol would have silently reported as predictive skill. In v4.2, after deduplicating to 768 unique sequences, the same comparison is **0.7867 random vs 0.7259 controlled** — a **+0.0608** delta. The shrinkage from +0.176 to +0.061 is itself the finding: it shows the v4.1 inflation was driven by near-duplicate *rows* (repeat measurements of the same sequence falling on both sides of a random boundary), and that deduplication removes exactly that component while the homology control handles the remaining anagram component. We ship the audit so the reader can check the regime rather than trusting an unexamined split. On Hemolysis the delta is small and slightly negative (0.8348 vs 0.8112) because the sequence families are spread thinly enough that a random draw rarely re-presents the exact composition region.

### 3.3 Why the Numbers Are 0.39–0.73, Not 0.99

On real data the labels are measured, noisy, and partly unidentifiable from composition-level (AAC/DPC) features, 2D descriptors, or even a frozen protein- or molecular-language-model embedding alone: permeability depends on sequence *order*, conformation, and transporter effects that none of these features captures fully, and the repeat measurements of the same molecule disagree by 0.6–0.94 log units (Caco-2) and 0.6 log units (PAMPA) — an irreducible noise floor no model can fit. The R² 0.3909 (Caco-2) and 0.4642 (PAMPA) permeability regressions are the honest result of a descriptor + frozen-molecular-transformer model on real, heterogeneous, noisy permeability data — not a defect. The sequence endpoints sit at AUC 0.8348 (Hemolysis) and R² 0.7259 (half-life, sequence-level) — above the v2.0 simulation target of R² ≥ 0.70 for half-life, short of AUC ≥ 0.85 for Hemolysis by 0.015. The model is neither overfitting (Hemolysis controlled ≥ random; half-life controlled well below the leakage-inflated random) nor trivially underfitting.

### 3.4 The v4.2 Ablation (same protocol, features + target + loss change)

Because v4.1 and v4.2 use the *identical* data and leakage protocol, the v4.1 → v4.2 deltas isolate the effect of the v4.2 changes (MoLFormer for the molecular endpoints; dedup + Huber for half-life; Huber for the molecular endpoints):

| Endpoint | v4.1 | v4.2 | Delta | v4.2 change |
|---|---|---|---|---|
| Hemolysis (AUC, controlled) | 0.8348 | 0.8348 | 0.0000 | (unchanged endpoint; deterministic retrain) |
| Half-life (R², controlled) | 0.6973 (row-level, 1,763) | 0.7259 (sequence-level, 768) | +0.0286 | dedup + Huber |
| Caco-2 (R², controlled) | 0.3861 | 0.3909 | +0.0048 | +768-dim frozen MoLFormer CLS + Huber |
| PAMPA (R², controlled) | 0.4573 | 0.4642 | +0.0069 | +768-dim frozen MoLFormer CLS + Huber |

Two honest caveats. First, the half-life delta is a comparison of *different granularities* (row-level 0.6973 vs sequence-level 0.7259) and is not a like-for-like number; the deduplication changes the unit of analysis, so the +0.0286 reflects both the removal of irreducible re-measurement noise and the Huber loss. Second, the molecular-endpoint MoLFormer gains (+0.0048 and +0.0069) are **within the run-to-run retraining noise** we measured: two independent complete retraining runs of the same v4.2 molecular configuration differed by up to ±0.01 (PAMPA 0.4527 vs 0.4642). We therefore do *not* claim the MoLFormer embedding produces a statistically meaningful gain on these endpoints; the honest reading is that the molecular-endpoint bottleneck is the **label noise floor** (0.6–0.94 log-unit spread across repeat measurements), not the 2D-descriptor representation layer, and a frozen molecular transformer does not break that floor. This is a *negative* result, reported as such, and it is precisely the kind of finding a leakage-audited, measured-only protocol is designed to surface rather than hide.

---

## 4. Discussion

### 4.1 What this contribution is

A **reproducibility and evaluation-protocol contribution, now on real data with both modalities embedding-augmented**: a real dataset with prepared artifacts shipped in the repository, a per-modality leakage audit, a shared model definition, measured-only metrics, target deduplication for the half-life endpoint, and frozen ESMC + MoLFormer embeddings for the two modalities — packaged so that every number is re-derivable from the repository. The protocol is not just pipeline-certified (as on the synthetic demo) but *applied to experimental data*, so the reported numbers carry real-peptide meaning within their stated feature limitations.

### 4.2 Relation to 2026 work

- **AMPBench-MT (arXiv:2607.25518)**: our sequence split + audit implements the leakage controls it advocates; §3.2 now reports the *sequence-level* random-vs-controlled delta (half-life R² 0.7867 → 0.7259 in v4.2; 0.8733 → 0.6973 at row-level in v4.1), and shows how target deduplication shrinks the delta from +0.176 to +0.061 — a refinement of the leakage-control story beyond what a pure homology split captures.
- **MoLFormer / molecular transformers**: our use of a frozen MoLFormer-XL CLS embedding as a static input feature (no fine-tuning) is consistent with the "pretrained molecular prior" line of work; our negative result (§3.4) — that the frozen embedding does not break the label-noise floor on these permeability endpoints — is a useful data point for the field.
- **ApexGO (Nat. Mach. Intell., 2026-05)** and **integrated agentic peptide pipelines (npj Drug Discovery, 2026-05)**: the "validate before claiming" stance we adopt throughout.
- **Genotypic Triggers (2026-08)**: safety blind spots from missing endpoint dimensions; our four-endpoint panel omits toxicogenomic and immunogenicity dimensions and states that in §4.4.

### 4.3 Practical guidance we recommend

1. Ship the prepared data (or its exact generator + source path) with the code — never reference a CSV the repository does not contain.
2. Audit and publish the split's leakage (similarity statistics + label-rate deltas) per modality, not just its stratification.
3. Use one shared model class for training and inference; release `metrics.json` and have the CLI print measured values only.
4. Report the random-vs-controlled-split delta whenever the dataset contains near-duplicate or anagrammatic sequences — and **deduplicate repeated measurements** before splitting, so the headline metric is at the unique-entity level (half-life: +0.176 row-level → +0.061 sequence-level).
5. State, per endpoint, the input modality actually used and why (a non-standard residue-name list is not a sequence a 20-AA encoder can consume; use a molecular transformer for the SMILES path).
6. When reporting a frozen-embedding gain, also report the **run-to-run retraining noise** of the same configuration — if the gain is within that noise, say so (our MoLFormer molecular gain is).

### 4.4 Limitations

1. **Four endpoints.** No toxicogenomic/pharmacogenomic, immunogenicity, or protease-stability endpoints (the blind-spot class documented by Genotypic Triggers; the PepADMET release contains no toxicity table, and none was fabricated).
2. **Molecular-endpoint leakage control is weaker.** Caco-2 and PAMPA have no sequence, so the split is by unique SMILES only; near-isomeric structures can cross the boundary. Their R² values may be modestly optimistic relative to a full homology control.
3. **Feature expressivity and the label-noise floor.** AAC/DPC discard order beyond bigrams; RDKit 2D descriptors + Morgan fingerprints are order-insensitive; the frozen ESMC embedding adds deep contextual information for the sequence endpoints and the frozen MoLFormer embedding adds a molecular prior for the molecular endpoints, but both are single static vectors, not task-tuned models. Conformational and transporter-specific effects remain out of reach for this model class, and the molecular-endpoint R² is bounded by the 0.6–0.94 log-unit noise floor of the repeat measurements — a frozen embedding cannot break it.
4. **The PAMPA/MDCK target is left-censored, which caps its R².** The logPapp target (7,283 rows) has only 648 unique values (0.01 quantization); 269 rows (3.7%) are exactly −10.0000, the assay's detection floor (true values ≤ −10, unknown). These censored points carry 49.6% of the total sum of squares. A dedicated ceiling study (reproducible scripts in `analysis/`) shows the censored molecules can be *partially ranked* but not reliably flagged (best LightGBM floor classifier AUC 0.856 on the test split; the MLP's own predictions rank the floor at AUC 0.762, but at the validation-tuned threshold the classifier's precision is only 0.12, and the two-stage method collapses to R² −1.21); the model already reaches R² 0.632 on the uncensored subset, and the theoretical ceiling (uncensored perfect, censored → global mean) is 0.5387. Five improvement routes were tested under the identical leakage-controlled split — rank-Gaussian target transform (0.434 ± 0.011, multi-seed), LightGBM × 128 hyperparameters × 4 feature sets + top-5 ensemble (0.4234), a two-stage floor method, a soft posterior-mean blend (0.4651), and a Tobit censored-likelihood model (0.4056 ± 0.024) — and none exceeded the v4.2 baseline (0.4642) beyond retraining noise. Raising PAMPA R² toward 0.70 requires uncensored re-measurements of the floored compounds, not a better model. Caco-2 is more severely bounded (repeat within-group SD ≈ 0.97 log units ≈ the target SD; ceiling ≈ 0.14 by the same logic), so its R² 0.3909 is likewise label-quality-limited.
5. **Half-life target is log10-transformed and deduplicated**; R² is reported in log10-seconds at the sequence level (768 unique sequences), not raw seconds and not the row level.
6. **No wet-lab validation.** These are model fits to published experimental values, not new measurements.
7. **Small half-life test set** (159 unique sequences) — its R² has a wider confidence interval than the permeability endpoints.
8. **Frozen embeddings only.** Neither ESMC nor MoLFormer is fine-tuned; the gain from a task-tuned encoder is unmeasured here and is the natural next step (§4.5).

### 4.5 Future directions

The v4.2 ablation (§3.4) and the PAMPA ceiling study (§4.4, item 4) motivate several concrete next steps: (a) a **task-tuned (fine-tuned) sequence encoder** — or per-residue attention — to capture the order effects a static mean-pooled ESMC embedding leaves on the table, and a **task-tuned molecular encoder** (fine-tuned MoLFormer or a GNN) for the permeability endpoints, since the frozen-embedding negative result suggests the representation *can* be improved but only with gradient updates; (b) **uncensored re-measurements of the PAMPA floored compounds** (the 269 rows at the −10.0000 detection limit) — the ceiling study shows this is the single highest-leverage intervention for the molecular endpoints, since 49.6% of the target variance is unrecoverable from structure while the labels remain censored; (c) the omitted safety/stability endpoints; and (d) exposing per-endpoint predictions as objectives for generative peptide design in the AMPGAN v3 / PepCraft style.

---

## 5. Conclusions

We moved a reproducibility benchmark from the synthetic demo that certified it to **real experimental data**, keeping the protocol identical: a leakage-audited per-modality split (exact-anagram collapse for sequence endpoints, unique-SMILES for molecular endpoints), a shared model definition, and measured-only metrics over four ADMET endpoints. v4.2 extends both modalities with frozen pretrained embeddings — ESMC-600M for sequences (v4.1) and MoLFormer-XL for molecules (v4.2) — and adds target deduplication for the half-life endpoint (1,763 rows → 768 unique sequences) plus a Huber regression loss. On real data the dual-split comparison becomes decisive and *informs the protocol*: the random half-life split reported R² 0.8733 at the row level in v4.1, but the leakage-controlled honest number was 0.6973 (delta +0.176); after v4.2's deduplication to the sequence level the same comparison is 0.7867 vs 0.7259 (delta +0.061) — the shrinkage is the direct evidence that the inflation was near-duplicate-row leakage, and deduplication removes it. The frozen ESMC ablation (v4.1) showed a pretrained protein-language-model prior, used with no gradient update, closes a large share of the remaining sequence-endpoint gap (Hemolysis AUC 0.7755 → 0.8348; half-life R² 0.5883 → 0.6973). The frozen MoLFormer ablation (v4.2) is a *negative* result on the molecular endpoints: the gain (+0.005 to +0.007) is within retraining noise, so the molecular-endpoint bottleneck is label noise, not the representation layer. The released repository now contains the prepared data, both frozen embedding caches (ESMC + MoLFormer), all four model weights, scalers, and `metrics.json`, so every number is re-derivable from the repository. Peptide ADMET numbers should be reported with their split provenance, input modality, and target granularity; we demonstrate the standard on real data.

**Availability**: all code, the four trained models (sequence endpoints 438,529 params with ESMC; molecular endpoints 810,497 params with MoLFormer), the frozen ESMC and MoLFormer embedding caches, the scalers, the prepared data, and `metrics.json` are at https://github.com/c00jsw00/openclaw-peptide-admet.

---

## 6. Acknowledgments

We thank the authors of the Chemit797/PepADMET-Dataset release for providing cleaned real-data endpoint tables, the AMPBench-MT authors and the 2026 generative-AMP communities for the evaluation-integrity standards adopted here, and the Biohub and IBM Research teams for releasing ESMC-600M and MoLFormer-XL as pretrained, openly-licensed models.

---

## 7. References

(1–17 as in prior versions, condensed: peptide therapeutics background, ADMET tool literature, classical QSAR/deep-learning comparisons, RDKit/CSD.)

18. **AMPBench-MT**: Multi-task benchmarking for antimicrobial peptide prediction: the case for homology-controlled evaluation. *arXiv:2607.25518* (2026).
19. **AMPGAN v3 / PepCraft**: Generative redesign of antimicrobial peptides with multi-objective candidate ranking and wet-lab MIC validation. *arXiv* (2026).
20. **ApexGO**: Generative redesign of antibiotic scaffolds with validation-gated evaluation. *Nature Machine Intelligence* (2026-05).
21. **Integrated agentic peptide-discovery pipeline** (ProtGPT2 soft-prompt integration, LLM-planned experiments). *npj Drug Discovery* (2026-05).
22. **Genotypic Triggers**: pharmacogenomic "back doors" as a safety blind spot in polypharmacy risk prediction. (2026-08).
23. **PepADMET-Dataset**: Chemit797/PepADMET-Dataset. https://github.com/Chemit797/PepADMET-Dataset (2026).
24. **ESMC-600M (ESM Cambrian)**: Biohub. https://huggingface.co/biohub/ESMC-600M (2025).
25. **MoLFormer-XL**: IBM Research. `ibm-research/MoLFormer-XL-both-10pct`. https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct (2023).

---

## Supporting Information Available

- **S1.** `data/pepadmet_data.meta.json` — per-endpoint source path, prepared row counts, dropped-row statistics.
- **S2.** `models_v4/<endpoint>/metrics.json` — all measured per-endpoint metrics (both splits where applicable), split statistics, the per-endpoint leakage audit, the v4.2 `dedup_info` (half-life 1,763 → 768), the `feature_layout` (1,580 / 3,033-dim), and the embedding metadata.
- **S3.** `models_v4/summary.json` — four-endpoint headline summary.
- **S4.** `train_v42.log` — the human-readable record of the v4.2 training run (per-endpoint training curves, split counts, final SUMMARY block).
- **S5.** `data/esmc/esmc_emb_<endpoint>.npz` — the frozen ESMC-600M mean-pooled embedding caches (Hemolysis 8,719 × 1,152; Half-life 1,763 × 1,152, float32), generated by `esmc_embed.py`; committed so retraining and cached-sequence inference need no ESMC dependency.
- **S6.** `data/molformer/molformer_emb_<endpoint>.npz` — the frozen MoLFormer-XL CLS embedding caches (Caco-2 7,429 × 768; PAMPA 7,283 × 768, float32), generated by `molformer_embed.py`; committed so retraining and cached-SMILES inference need no MoLFormer dependency.

---

## Author Information

**Corresponding Author**: Pinwan (品丸), OpenClaw Team.

**Data and Code Availability**: https://github.com/c00jsw00/openclaw-peptide-admet

**Submission positioning**: benchmark / reproducibility-protocol contribution, now with real-data results on both modalities. The appropriate venue framing is a methods/benchmark article (e.g., JCIM's benchmark track or a workshop on ML evaluation integrity); the real-data R² values are honest and modest and are reported as such, including the negative MoLFormer result.

---

**Manuscript prepared**: 2026-08-25; **updated**: 2026-08-26
**Version**: 4.2 (real-data + ESMC-600M + MoLFormer-XL edition; builds on the 2026-08-26 v4.1 real-data + ESMC edition)
**Status**: internally consistent with the released v4.2 repository (metrics, weights, ESMC + MoLFormer caches, and prepared data all committed)
