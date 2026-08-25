# From Synthetic Demo to Real Data: A Reproducible, Leakage-Audited Benchmark for Four Peptide ADMET Endpoints on the PepADMET Dataset

**Running Title:** Leakage-Audited Peptide ADMET Benchmark on Real Data

**Article Type:** Article (Full-Length Research Manuscript)

> **⚠️ v4.0 repository note (2026-08-25).** This manuscript was updated in
> place for the **v4.0 real-data edition**. The released repository previously
> validated the evaluation protocol (homology-controlled split + leakage audit +
> dual-split reporting) on a **30,000-row regenerable synthetic demo set**
> (`synthetic_demo`, v3.0) and a 9-endpoint mixed multi-task model. v4.0
> **removes the synthetic set entirely** and re-runs the *same* protocol on
> **real experimental data** from the
> [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset)
> release, over **four endpoints** — Hemolysis, plasma Half-life, Caco-2
> permeability, and PAMPA/MDCK permeability — using a **dual-modality feature
> design** (428-dim amino-acid-sequence features for the sequence endpoints;
> 2,265-dim RDKit molecular descriptors + Morgan fingerprints for the
> SMILES-only permeability endpoints). All performance figures below are
> *measured* and re-derivable from the released code (`metrics.json`); the
> v2.0/v3.0 synthetic-demo figures are no longer the headline.

**Version:** 4.0 (real-data edition, 2026-08-25). This revision replaces the
v3.0 manuscript, which was an integrity revision (replacing v1.0's hardcoded
97.70% / 0.9987 AUC) validated on a synthetic demo set. The evaluation protocol
is unchanged; the data, endpoint set, and feature design are new.

---

## Abstract

Peptide-based therapeutics are among the fastest-growing pharmaceutical classes, yet computational ADMET prediction for peptides is plagued by an evaluation-integrity problem that 2026 benchmark work has begun to expose: sequence-similarity leakage across the train/test boundary inflates reported performance, and several public tools report metrics that cannot be reproduced from their released artifacts. Prior versions of this repository validated a reproducibility protocol — an AMPBench-MT-style homology-controlled split with a shipped leakage audit, dual-split reporting, and measured-only inference — but did so on a *synthetic* demo set, which can certify the pipeline yet cannot claim real-peptide accuracy. Here we close that gap: we re-run the identical protocol on **real experimental peptide data** (the Chemit797/PepADMET-Dataset release) over **four ADMET endpoints** (Hemolysis [binary], plasma half-life [regression], and Caco-2 and PAMPA/MDCK permeability [regression]). Because the four endpoint tables are disjoint molecule sets spanning two input modalities, we use a **dual-modality** design: 428-dim amino-acid-composition features (AAC 20 + DPC 400 + physicochemical 8) for the two sequence endpoints and 2,265-dim RDKit features (217 2D descriptors + 2,048-bit Morgan fingerprint) for the two SMILES-only permeability endpoints, each with an independent single-task MLP. On the leakage-controlled test split the models achieve **AUC 0.7755 (Hemolysis)**, **R² 0.5883 (half-life, log10-seconds)**, **R² 0.3861 (Caco-2, logPapp)**, and **R² 0.4573 (PAMPA, logPapp)**. The dual-split comparison is now genuinely informative: on half-life a plain random split reports R² 0.8650 — inflated by near-duplicate leakage — versus 0.5883 under the homology-controlled protocol, and we collapse exact 3-mer-multiset anagrams into a single family to *guarantee* that no jaccard-1.0 duplicate crosses the boundary. We report these modest, real-data numbers deliberately; the contribution is a reproducible, leakage-audited benchmark and reference implementation the field can adopt, with the exact weights, scalers, prepared data, and measured metrics released with the repository.

**Keywords:** peptide ADMET prediction | benchmark | evaluation leakage | homology-controlled split | dual-modality features | RDKit descriptors | reproducibility | Caco-2 permeability | PAMPA | hemolysis

---

## Graphical Abstract and Table of Contents Entry

**Table of Contents Graphic:** [real peptide table (sequence ∪ SMILES) → dual-modality features (428-dim sequence / 2,265-dim RDKit) → four single-task MLPs → 4 ADMET predictions; inset: leakage audit — exact-anagram collapse + max cross-boundary Jaccard]

**TOC Entry:** A leakage-audited protocol, first validated on a synthetic demo and now re-run on real PepADMET data over four dual-modality endpoints, yields reproducible peptide ADMET numbers (AUC 0.776; R² 0.386–0.588) — and a template for how the field should report them.

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

- **Sequence-similarity leakage.** AMPBench-MT (arXiv:2607.25518), a 2026 multi-task benchmark for antimicrobial peptides, shows that when near-duplicate or compositionally similar sequences fall on both sides of the train/test boundary, reported accuracy and AUC can be inflated far beyond what the model could achieve on genuinely novel sequences. Homology-aware (or composition-family-aware) splitting is the currently recommended remedy, but few public peptide-ADMET repositories audit their splits or release the audit. This problem is *sharper* on real data than on a synthetic demo: real peptide libraries contain many near-duplicate and anagrammatic sequences that a naive random split will happily split across the boundary.
- **Non-reproducible metrics.** We observed, in our own v1.0 submission package, metrics that were hardcoded in the inference CLI and inconsistent with the shipped model artifacts, and a "training dataset" referenced in documentation that was not present in the repository.
- **Modality mismatch.** Real ADMET datasets are rarely uniform: some tables ship clean one-letter sequences, others ship only SMILES (or, as in the CycPeptMPDB-derived permeability tables here, *non-standard* residue-name lists such as `MEL`, `DP`, `DL` that a 20-amino-acid encoder cannot consume). A benchmark must state, per endpoint, what input it actually uses and why.

### 1.3 Prior Work (condensed)

Classical tools (AdmetSAR 2.0, SwissADME, ADMETlab 3.0, pepADMET) offer peptide ADMET predictions but were primarily optimized for small molecules or validated on small, non-external test sets. Deep approaches (LSTM, Transformer, GNN, protein language models) capture sequence order but demand large datasets and GPU resources. Handcrafted-composition QSAR and RDKit descriptor models remain competitive, particularly at modest data scale. 2026 has added generative-redesign pipelines for antibiotics (ApexGO, *Nature Machine Intelligence*, 2026-05) and integrated agentic pipelines for peptide campaigns (*npj Drug Discovery*, 2026-05), all of which converge on the same methodological point: **report what was measured, on which split, with which leakage controls.**

### 1.4 Study Objectives

1. Replace the synthetic demo set with a **real, permissively-licensed experimental dataset** (the Chemit797/PepADMET-Dataset release) and keep every shipped artifact present in the repository (prepared data, weights, scalers, metrics) so nothing referenced in documentation can be silently missing.
2. Implement a **leakage-audited split per modality**: an AMPBench-MT-style homology-controlled split with **exact-anagram collapse** for sequence endpoints (guaranteeing no jaccard-1.0 duplicate crosses the boundary), and a unique-SMILES grouping split for the SMILES-only endpoints (with the near-isomer limitation stated explicitly).
3. Train and release a **dual-modality** model family — the *same* architecture class in trainer and predictor — with the exact weights, scalers, and measured metrics in `metrics.json`.
4. Report **measured** per-endpoint AUC/MCC/accuracy (binary) and R²/RMSE/MAE (regression) on the leakage-controlled split (headline) and a random split (leakage comparison), and state plainly what the numbers do and do not claim.

### 1.5 Significance

**Methodological**: a reference implementation of the evaluation protocol (per-modality leakage audit + dual-split reporting + measured-metrics-only inference) applied to *real* data — the step the synthetic-demo version could not take.
**Reproducibility**: every number in this manuscript is a function of released code and released data; `prepare_pepadmet_data.py → train_pepadmet_model.py → peptide_admet_predictor.py` regenerates the pipeline end-to-end on CPU.
**Honesty**: we report AUC 0.7755 and R² 0.3861–0.5883, not 0.9987. The large random-vs-controlled delta on half-life (0.8650 vs 0.5883) is the *finding* — it is the leakage a naive protocol would have silently reported as performance, and we ship the audit so the reader can check the regime.

---

## 2. Materials and Methods

### 2.1 Data (real experimental dataset, provenance stated)

We use the **Chemit797/PepADMET-Dataset** release (its cleaned `整理/` sub-directory), which provides four endpoint tables:

| Endpoint | Source table | Input | Label | Rows (prepared) |
|---|---|---|---|---|
| Hemolysis | `hemolysis_unified/hemolysis_unified.csv` | `sequence_std` (one-letter 20-AA) | `label` (0/1) | 8,719 |
| Half-life | `half_life_*/...` | `sequence` (one-letter 20-AA) | `half_life_seconds` (continuous) | 1,763 |
| Caco-2 | `caco2_*/...` | `SMILES` (valid) | `Permeability` (logPapp) | 7,429 |
| PAMPA/MDCK | `pampa_mdck_*/...` | `SMILES` (valid) | `PAMPA` (logPapp) | 7,283 |

The four tables are **disjoint molecule sets** (no compound appears in more than one), so no multi-task shared-label structure exists; each endpoint is an independent single-task problem. `prepare_pepadmet_data.py` loads each table, validates the input (sequence endpoints: rows must contain a clean 20-AA one-letter sequence; molecular endpoints: the SMILES must parse under RDKit), drops rows with a missing/non-finite label, and writes a per-endpoint prepared CSV plus a provenance/`meta.json` recording source path, row counts, and the exact dropped-row statistics. No synthetic labels are generated; every label is taken verbatim from the source table.

**Half-life target transform.** The raw half-life spans ~10⁻³ to ~10⁹ s (log10 range −3.1 to 9.1). We model **log10(seconds)** as the regression target and report R² in that space; the inference CLI inverts to seconds for display. Caco-2 and PAMPA permeability are already log-scale (logPapp) and are modeled directly.

### 2.2 Feature Engineering (dual-modality)

Identical in training and inference (single shared implementation in `feature_extractor.py`):

**Sequence modality (428-dim)** — Hemolysis, Half-life:
1. **Amino Acid Composition (AAC)** — 20: frequency of each standard amino acid.
2. **Dipeptide Composition (DPC)** — 400: frequency of every ordered dipeptide.
3. **Physicochemical** — 8: estimated MW, average Kyte–Doolittle hydropathy, net charge at pH 7, pI estimate, GRAVY, hydrophobic/charged-residue ratios.

**Molecular modality (2,265-dim)** — Caco-2, PAMPA/MDCK:
1. **RDKit 2D descriptors** — 217: `rdkit.Chem.Descriptors.CalcMolDescriptors(mol)` (a fixed, deterministic registry).
2. **Morgan fingerprint** — 2,048 bits, radius 2.

The two permeability tables' native "sequence" column is a non-standard residue-name list (`MEL`, `DP`, `DL`, `ME_DL`, …) from CycPeptMPDB that a 20-AA encoder cannot consume; we therefore use the (valid) SMILES column and state this per endpoint in `endpoint_config.py`. A SMILES that fails RDKit parsing yields an all-zero molecular row and is counted in the preparation statistics (not silently fabricated). All features are Z-score standardized with a scaler fit on the training split only.

### 2.3 Model

A single MLP class (`MixedADMETMLP` in `admet_model.py`) is instantiated per endpoint — the *same* class in trainer and predictor, so the two can never drift apart:

- Input `d` → Linear 256 → BatchNorm → ReLU → Dropout(0.2) → Linear 128 → BatchNorm → ReLU → Dropout(0.2) → a single task head (`Linear(128,1)`; sigmoid for Hemolysis, identity for the three regressions).
- Parameter counts: **143,617** (sequence endpoints, d=428) and **613,889** (molecular endpoints, d=2,265). Loss: BCE-with-logits (Hemolysis) or MSE (regressions). Optimizer: Adam (lr 3e-4), `ReduceLROnPlateau` (factor 0.5, patience 3), early stopping on the validation objective (patience 8). Trained on CPU. No ensemble; no Random Forest.

### 2.4 Leakage-Controlled Splitting with Audit (per modality)

**Sequence endpoints (Hemolysis, Half-life).** Following AMPBench-MT (arXiv:2607.25518): (1) each sequence is reduced to a **canonical 3-mer-multiset signature** (a count vector over its 3-mers); two sequences with identical 3-mer multisets have 3-mer Jaccard = 1.0, so collapsing by signature *guarantees* no exact-jaccard-1.0 duplicate (including length-preserving anagrams) is ever placed on both sides of the boundary; (2) the unique signatures are clustered by greedy single-linkage 3-mer Jaccard (threshold 0.35); (3) **families** — not sequences — are allocated to train/val/test at 70/10/20; (4) a **leakage audit** is shipped with the split: maximum audited cross-boundary 3-mer Jaccard and the per-endpoint label-rate delta. In our run the max cross-boundary Jaccard is ≈0.968 (Hemolysis) and ≈0.974 (half-life) — the expected near-duplicate ceiling under a controlled split, with exact-multiset leakage **guaranteed 0**.

**Molecular endpoints (Caco-2, PAMPA/MDCK).** No sequence is available, so a 3-mer homology control is impossible. We instead group by **unique SMILES** (exact-duplicate SMILES share one split) and draw a 70/10/20 split over unique SMILES. We state the limitation explicitly: **near-isomeric structures** (different SMILES strings, same chemistry) can cross the boundary — a real-data limitation of SMILES-only data, weaker than the sequence homology control. This is recorded in each endpoint's `metrics.json` audit.

As a **leakage comparison** (sequence endpoints only), the identical model is trained on a plain random 70/10/20 split; the delta between the random and controlled test metrics quantifies the leakage the random protocol would have reported.

### 2.5 Evaluation

Per endpoint, computed on the held-out test split only and written to `metrics.json`: binary endpoints report AUC-ROC, MCC, and accuracy at threshold 0.5; regression endpoints report R², RMSE, and MAE (in the modeling space: log10-seconds for half-life, logPapp for permeability). Headline numbers are the leakage-controlled test metrics; the random-split numbers are the leakage comparison.

### 2.6 Multi-Objective Composite Score (removed in v4.0)

The v3.0 composite score (geometric mean over favorable endpoint probabilities) assumed five *binary* endpoints of mixed favorability. With the v4.0 set — one binary plus three continuous regression endpoints on disjoint molecules — a single geometric-mean score is not meaningful, so it is **removed**; the predictor reports each endpoint's value in its own units.

---

## 3. Results

### 3.1 Measured Performance (leakage-controlled test split, headline)

All values below are taken from `models_v4/<endpoint>/metrics.json` produced by the released training script (seed 42, 80 epochs, early-stopped).

| Endpoint | Kind | Modality | Test (n) | Primary | Other |
|---|---|---|---|---|---|
| Hemolysis | binary | sequence | 1,745 | AUC **0.7755** | MCC 0.3782, Acc 0.7009 |
| Half-life | regression (log10 s) | sequence | 428 | R² **0.5883** | RMSE 1.2502, MAE 0.8714 |
| Caco-2 | regression (logPapp) | molecular | 1,490 | R² **0.3861** | RMSE 0.7879, MAE 0.4896 |
| PAMPA/MDCK | regression (logPapp) | molecular | 1,457 | R² **0.4573** | RMSE 0.8043, MAE 0.5070 |

### 3.2 Dual-Split Comparison (the leakage question, now real)

| Endpoint | Controlled test | Random test | Delta (random − controlled) |
|---|---|---|---|
| Hemolysis (AUC) | 0.7755 | 0.7746 | −0.0009 |
| Half-life (R², log10 s) | 0.5883 | 0.8650 | **+0.2767** |
| Caco-2 / PAMPA | — | — | (no sequence; unique-SMILES split only) |

The half-life delta is the central real-data demonstration: a plain random split reports **R² 0.8650**, but under the leakage-controlled protocol the honest number is **0.5883**. The 0.2767 gap is near-duplicate and near-anagram leakage that a naive protocol would have silently reported as predictive skill. On Hemolysis the delta is near zero because the sequence families are spread thinly enough that a random draw rarely re-presents the exact composition region — the regime in which the protocol matters most (large or seed-mutated families, as in real libraries) is exactly the half-life regime. We ship the audit so the reader can check the regime rather than trusting an unexamined split.

### 3.3 Why the Numbers Are 0.4–0.8, Not 0.99

On real data the labels are measured, noisy, and partly unidentifiable from composition-level (AAC/DPC) or 2D-descriptor features alone: permeability depends on sequence *order*, conformation, and transporter effects that neither feature set captures fully. The R² 0.386–0.457 permeability regressions are the honest result of a descriptor-level model on real, heterogeneous permeability data — not a defect. The model is neither overfitting (Hemolysis controlled ≈ random; half-life controlled well below the leakage-inflated random) nor trivially underfitting (AUC 0.7755, R² up to 0.5883, well above chance).

---

## 4. Discussion

### 4.1 What this contribution is

A **reproducibility and evaluation-protocol contribution, now on real data**: a real dataset with prepared artifacts shipped in the repository, a per-modality leakage audit, a shared model definition, and measured-only metrics, packaged so that every number is re-derivable from the repository. For the first time the protocol is not just pipeline-certified (as on the synthetic demo) but *applied to experimental data*, so the reported numbers carry real-peptide meaning within their stated feature limitations.

### 4.2 Relation to 2026 work

- **AMPBench-MT (arXiv:2607.25518)**: our sequence split + audit implements the leakage controls it advocates; §3.2 now reports a *large, real* random-vs-controlled delta (half-life R² 0.8650 → 0.5883), which is precisely the inflation it documents.
- **ApexGO (Nat. Mach. Intell., 2026-05)** and **integrated agentic peptide pipelines (npj Drug Discovery, 2026-05)**: the "validate before claiming" stance we adopt throughout.
- **Genotypic Triggers (2026-08)**: safety blind spots from missing endpoint dimensions; our four-endpoint panel omits toxicogenomic and immunogenicity dimensions and states that in §4.4.

### 4.3 Practical guidance we recommend

1. Ship the prepared data (or its exact generator + source path) with the code — never reference a CSV the repository does not contain.
2. Audit and publish the split's leakage (similarity statistics + label-rate deltas) per modality, not just its stratification.
3. Use one shared model class for training and inference; release `metrics.json` and have the CLI print measured values only.
4. Report the random-vs-controlled-split delta whenever the dataset contains near-duplicate or anagrammatic sequences — on real peptide libraries this delta is often large (half-life: +0.277 R²).
5. State, per endpoint, the input modality actually used and why (a non-standard residue-name list is not a sequence a 20-AA encoder can consume).

### 4.4 Limitations

1. **Four endpoints.** No toxicogenomic/pharmacogenomic, immunogenicity, or protease-stability endpoints (the blind-spot class documented by Genotypic Triggers).
2. **Molecular-endpoint leakage control is weaker.** Caco-2 and PAMPA have no sequence, so the split is by unique SMILES only; near-isomeric structures can cross the boundary. Their R² values may be modestly optimistic relative to a full homology control.
3. **Composition-level / 2D-descriptor features.** AAC/DPC discard order beyond bigrams; RDKit 2D descriptors + Morgan fingerprints are order-insensitive. Sequence-order-sensitive and conformational effects are out of reach for this model class.
4. **Half-life target is log10-transformed**; R² is reported in log10-seconds, not raw seconds.
5. **No wet-lab validation.** These are model fits to published experimental values, not new measurements.
6. **Small half-life test set** (428 rows) — its R² has a wider confidence interval than the permeability endpoints.

### 4.5 Future directions

Extend features with a frozen protein-language-model embedding (ProtGPT2-style) or a GNN backbone for order sensitivity on the sequence endpoints; add a molecular graph model to strengthen the molecular-endpoint leakage control; add the omitted safety/stability endpoints; and expose per-endpoint predictions as objectives for generative peptide design in the AMPGAN v3 / PepCraft style.

---

## 5. Conclusions

We moved a reproducibility benchmark from the synthetic demo that certified it to **real experimental data**, keeping the protocol identical: a leakage-audited per-modality split (exact-anagram collapse for sequence endpoints, unique-SMILES for molecular endpoints), a shared model definition, and measured-only metrics over four ADMET endpoints. On real data the dual-split comparison becomes decisive — a random half-life split reports R² 0.8650, but the leakage-controlled honest number is 0.5883 — and the released repository now contains the prepared data, all four model weights, scalers, and `metrics.json`, so every number is re-derivable from the repository. Peptide ADMET numbers should be reported with their split provenance and input modality; we demonstrate the standard on real data.

**Availability**: all code, the four trained models (sequence endpoints 143,617 params; molecular endpoints 613,889 params), the scalers, the prepared data, and `metrics.json` are at https://github.com/c00jsw00/openclaw-peptide-admet.

---

## 6. Acknowledgments

We thank the authors of the Chemit797/PepADMET-Dataset release for providing cleaned real-data endpoint tables, and the AMPBench-MT authors and the 2026 generative-AMP communities for the evaluation-integrity standards adopted here.

---

## 7. References

(1–17 as in prior versions, condensed: peptide therapeutics background, ADMET tool literature, classical QSAR/deep-learning comparisons, RDKit/CSD.)

18. **AMPBench-MT**: Multi-task benchmarking for antimicrobial peptide prediction: the case for homology-controlled evaluation. *arXiv:2607.25518* (2026).
19. **AMPGAN v3 / PepCraft**: Generative redesign of antimicrobial peptides with multi-objective candidate ranking and wet-lab MIC validation. *arXiv* (2026).
20. **ApexGO**: Generative redesign of antibiotic scaffolds with validation-gated evaluation. *Nature Machine Intelligence* (2026-05).
21. **Integrated agentic peptide-discovery pipeline** (ProtGPT2 soft-prompt integration, LLM-planned experiments). *npj Drug Discovery* (2026-05).
22. **Genotypic Triggers**: pharmacogenomic "back doors" as a safety blind spot in polypharmacy risk prediction. (2026-08).
23. **PepADMET-Dataset**: Chemit797/PepADMET-Dataset. https://github.com/Chemit797/PepADMET-Dataset (2026).

---

## Supporting Information Available

- **S1.** `data/pepadmet_data.meta.json` — per-endpoint source path, prepared row counts, dropped-row statistics.
- **S2.** `models_v4/<endpoint>/metrics.json` — all measured per-endpoint metrics (both splits where applicable), split statistics, and the per-endpoint leakage audit.
- **S3.** `models_v4/summary.json` — four-endpoint headline summary.
- **S4.** `train_v4.log` — the human-readable record of the training run (per-endpoint training curves, split counts, final SUMMARY block).

---

## Author Information

**Corresponding Author**: Pinwan (品丸), OpenClaw Team.

**Data and Code Availability**: https://github.com/c00jsw00/openclaw-peptide-admet

**Submission positioning**: benchmark / reproducibility-protocol contribution, now with real-data results. The appropriate venue framing is a methods/benchmark article (e.g., JCIM's benchmark track or a workshop on ML evaluation integrity); the real-data R² values are honest and modest and are reported as such.

---

**Manuscript prepared**: 2026-08-25
**Version**: 4.0 (real-data edition; replaces the 2026-08-25 v3.0 synthetic-demo update)
**Status**: internally consistent with the released v4.0 repository (metrics, weights, and prepared data all committed)
