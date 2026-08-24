# An Honest, Reproducible Benchmark for Peptide ADMET Prediction: Homology-Controlled Evaluation of a Multi-Task Neural Network on a Regenerable Synthetic Demo Set

**Running Title:** Homology-Controlled Peptide ADMET Benchmark

**Article Type:** Article (Full-Length Research Manuscript)

> **⚠️ v3.0 repository note (2026-08-25).** This manuscript describes the
> **v2.0, five-endpoint** configuration (144,133-param MLP, macro AUC 0.8684).
> The released repository has since been extended to **v3.0**: a 145,681-param
> **mixed multi-task** model covering **nine endpoints** (the five above plus
> pepADMET's `toxicity_binary`, `toxicity_type` [6-class],
> `neurotoxicity_type` [4-class], and `HC50` [regression]), with
> **partial-label masking** and an **extensible training set** (`prepare_data.py
> --n <any>` + `ingest_external.py --merge`). On a 30,000-row training set the
> homology-controlled mean primary metric is **0.7189** (per-endpoint values in
> `README.md` and `PREDICTOR_SUMMARY.md`). The evaluation protocol
> (homology-controlled split + leakage audit + dual-split reporting) is
> unchanged. **This manuscript must be updated to the v3.0 endpoint set and
> 30k metrics before any (re-)submission; the figures below reflect v2.0.**

**Version:** 2.0 (integrity revision, 2026-08-24). This revision replaces the v1.0 manuscript's
reported metrics (97.70% accuracy / 0.9987 AUC), which traced to hardcoded values and a
homology-uncontrolled split on a dataset that was never actually shipped with the
repository. All performance statements below are *measured* and reproducible from the
released code (see §2 and the repository's `metrics.json`).

---

## Abstract

Peptide-based therapeutics are one of the fastest-growing pharmaceutical classes, yet computational ADMET prediction for peptides is plagued by an evaluation-integrity problem that 2026 benchmark work has begun to expose: sequence-similarity leakage between train and test sets inflates reported performance, and several public tools report metrics that cannot be reproduced from their released artifacts. We contribute a reproducible benchmark and reference implementation for five peptide ADMET endpoints (GI absorption, Caco-2 permeability, BBB penetration, Ames mutagenicity, hERG inhibition). The pipeline consists of (i) a fully regenerable synthetic demo dataset of 15,000 sequences with explicitly labeled provenance, (ii) an AMPBench-MT-style homology-controlled train/val/test split that groups sequences by amino-acid-composition family and *measures* residual leakage, and (iii) a shared multi-task PyTorch MLP (428-dim AAC/DPC/physicochemical features; 144,133 parameters) whose exact weights, scaler, and measured per-endpoint metrics are released with the repository. On the homology-controlled split the model achieves a macro AUC of 0.8684 (mean accuracy 0.7836), compared with 0.8688 on a plain random split — a near-zero delta that we attribute to the limited sequence-level signal in composition features rather than to leakage. We deliberately report these *modest* numbers: on a synthetic latent-physicochemical label model, 0.8–0.9 AUC is the expected ceiling, and we treat the near-parity between split protocols as the demonstration the field needs — honest numbers, not 0.9987. We further add a multi-objective composite score (geometric mean over favorable endpoint probabilities, in the spirit of generative AMP campaigns such as AMPGAN v3 / PepCraft) for candidate prioritization. We position this work as a reproducibility and evaluation-protocol contribution, not as a claim of real-peptide predictive accuracy, and we make no submission-ready performance claim until real experimental data are used.

**Keywords:** peptide ADMET prediction | benchmark | evaluation leakage | homology-controlled split | multi-task learning | drug discovery | reproducibility | hERG inhibition | blood-brain barrier penetration

---

## Graphical Abstract and Table of Contents Entry

**Table of Contents Graphic:** [peptide sequence → 428-dim features (AAC + DPC + physchem) → multi-task MLP (5 heads) → 5 ADMET probabilities → multi-objective composite score; inset: leakage audit of the homology-controlled split]

**TOC Entry:** A regenerable synthetic demo set, an AMPBench-MT-style homology-controlled split with leakage auditing, and a released multi-task MLP yield reproducible peptide ADMET benchmark numbers (macro AUC 0.868) — and a template for how the field should report them.

---

## 1. Introduction

### 1.1 Background

Peptide therapeutics have emerged as a promising class of drugs, with over 90 approved peptide drugs and hundreds more in clinical development. Their high specificity, potency, and favorable safety profiles have driven extensive research, while the development of peptide drugs faces unique challenges that differ from small-molecule drug development:

1. **Poor Oral Bioavailability**: Peptides typically exhibit low gastrointestinal absorption due to large molecular size (>500 Da), high polarity, and enzymatic degradation in the digestive tract. Only approximately 1–2% of peptide drugs are administered orally.
2. **Membrane Permeability Limitations**: The polar nature of peptide bonds and side chains limits passive diffusion across biological membranes; the blood-brain barrier (BBB) imposes an even stricter barrier for CNS-targeted therapeutics.
3. **Metabolic Instability**: Peptides are rapidly degraded by proteases and peptidases, leading to short half-lives.
4. **Rapid Renal Clearance**: Small peptides (<5 kDa) are efficiently filtered by the kidneys, reducing systemic exposure.
5. **Potential Toxicity**: Some peptide sequences exhibit cytotoxicity, immunogenicity, or off-target effects, including hERG channel inhibition that can lead to QT prolongation and torsades de pointes.

### 1.2 The Evaluation-Integrity Problem in Peptide ADMET

Beyond the biological challenges, computational peptide ADMET has a methodological problem that recent 2026 work has brought to the foreground:

- **Sequence-similarity leakage.** AMPBench-MT (arXiv:2607.25518), a 2026 multi-task benchmark for antimicrobial peptides, shows that when near-duplicate or compositionally similar sequences fall on both sides of the train/test boundary, reported accuracy and AUC can be inflated far beyond what the model could achieve on genuinely novel sequences. Homology-aware (or composition-family-aware) splitting is the currently recommended remedy, but few public peptide-ADMET repositories audit their splits or release the audit.
- **Non-reproducible metrics.** We observed, in our own v1.0 submission package, metrics that were hardcoded in the inference CLI and inconsistent with the shipped model artifacts, and a "training dataset" referenced in documentation that was not present in the repository. Such artifacts are not isolated: generative AMP campaigns in 2026 (e.g., the AMPGAN v3 / PepCraft workflow, arXiv 2026) emphasize *measured* MIC-linked validation of generated candidates precisely because reported-in-silico-metrics-only studies have repeatedly failed wet-lab confirmation.
- **Blindness to pharmacogenomic "back doors".** The 2026 Genotypic Triggers work demonstrates that safety endpoints can be missed entirely when a model's endpoint list lacks pharmacogenomic dimensions; a 5-endpoint ADMET panel that omits toxicogenomic and immunogenicity endpoints should state that limitation explicitly, as we do in §4.4.

### 1.3 Prior Work (unchanged from v1.0, condensed)

Classical tools (AdmetSAR 2.0, SwissADME, ADMETlab 3.0, pepADMET) offer peptide ADMET predictions but were primarily optimized for small molecules or validated on small, non-external test sets. Deep approaches (LSTM, Transformer, GNN, protein language models) capture sequence order but demand large datasets and GPU resources. Handcrafted-composition QSAR remains competitive, particularly at modest data scale. 2026 has added generative-redesign pipelines for antibiotics (ApexGO, *Nature Machine Intelligence*, 2026-05) and integrated agentic pipelines for peptide campaigns (*npj Drug Discovery*, 2026-05), all of which converge on the same methodological point: **report what was measured, on which split, with which leakage controls.**

### 1.4 Study Objectives

1. Provide a **fully regenerable** demo dataset (fixed seed, explicit `data_origin` provenance column, metadata file) so that no shipped artifact can ever again be silently missing or fabricated.
2. Implement an **AMPBench-MT-style homology-controlled split** (composition-family grouping, 70/10/20) with an explicit **leakage audit** (maximum pairwise Jaccard across splits; per-endpoint label-rate deltas) released alongside the split.
3. Train and release a **shared multi-task PyTorch MLP** — the *same* architecture class in trainer and predictor — with the exact weights, scaler, and measured metrics in `metrics.json`.
4. Report **measured** per-endpoint AUC/MCC/accuracy on both the homology-controlled split (headline) and a random split (comparison), and state plainly what the numbers do and do not claim.
5. Add a **multi-objective composite score** for candidate prioritization, mirroring the multi-objective ranking used in generative AMP campaigns (AMPGAN v3 / PepCraft).

### 1.5 Significance

**Methodological**: a small, self-contained reference implementation of the evaluation protocol (leakage audit + dual-split reporting + measured-metrics-only inference) that the field can adopt.
**Reproducibility**: every number in this manuscript is a function of released code; `prepare_data.py → homology_split.py → train_peptide_admet_model.py → peptide_admet_predictor.py` regenerates the pipeline end-to-end on CPU in under an hour.
**Honesty**: we report macro AUC 0.8684, not 0.9987. The near-zero delta between homology-controlled and random splits (§3.2) is the *finding* — on composition-level features over this demo set, there is little leakage to control, and the modest absolute numbers are what a latent-physicochemical label model allows.

---

## 2. Materials and Methods

### 2.1 Data Generation (synthetic demo set, explicitly labeled)

Because no experimental peptide ADMET dataset of this size is openly available under a permissive license, and because the v1.0 "15,000 real peptides" CSV was never actually present in the repository, we generate a **synthetic demo dataset** and label it as such everywhere:

- **Sequences**: 15,000 sequences, lengths 10–30 aa. Sequences are drawn from 10,000 composition "families"; each family has an amino-acid composition profile sampled from a Dirichlet distribution (per-AAs α ∈ [0.8, 6.0]), giving controlled but diverse composition clusters. Seed 42 (`numpy.random.default_rng`).
- **Labels**: each of the 5 endpoints is the binary threshold (0.5) of a **latent physicochemical linear score** plus Gaussian noise. The scores use molecular-weight proxy, average Kyte–Doolittle hydropathy, net charge at pH 7, charged-residue fraction, and a per-endpoint composition tilt, with endpoint-specific weights (e.g., GI absorption penalizes long, charged, hydrophilic peptides; hERG inhibition is promoted by cationic + hydrophobic character).
- **Provenance**: every row carries `data_origin = synthetic_demo`; the metadata file (`peptide_admet_demo.meta.json`) records the seed, label model, and a plain-English statement that the set exists to validate the pipeline, not to model real peptide ADMET.

**Resulting positive rates** (measured): GI absorption ≈ 0.14, Caco-2 permeability ≈ 0.32, BBB penetration ≈ 0.10, Ames mutagenicity ≈ 0.17, hERG inhibition ≈ 0.29. These are *by construction*, and they are what the model must learn.

### 2.2 Feature Engineering

Identical in training and inference (single shared implementation):

1. **Amino Acid Composition (AAC)** — 20: frequency of each standard amino acid.
2. **Dipeptide Composition (DPC)** — 400: frequency of every ordered dipeptide.
3. **Physicochemical** — 8: estimated MW (length × 110 Da), average hydropathy, hydropathy range, net charge at pH 7, pI estimate, GRAVY, hydrophobic-residue fraction, charged-residue fraction.

Total: 428 dimensions, Z-score standardized with a scaler fit on the training split only.

### 2.3 Model

A single multi-task PyTorch MLP (defined once in `admet_model.py` and imported by both trainer and predictor, so the two can never drift apart):

- Input 428 → Linear 256 → BatchNorm → ReLU → Dropout(0.2) → Linear 128 → BatchNorm → ReLU → Dropout(0.2) → 5 × Linear(1) sigmoid heads.
- 144,133 parameters. Loss: mean of per-endpoint BCE with class weights derived from endpoint prevalence. Optimizer: Adam (lr 3e-4), `ReduceLROnPlateau` (factor 0.5, patience 3), early stopping on validation BCE (patience 8).
- Trained on CPU. No Random Forest component; v1.0's "ensemble (RF + NN)" is retired because the shipped `nn_model.pkl` was, in fact, a second Random Forest, which we no longer ship or claim.

### 2.4 Homology-Controlled Splitting with Leakage Audit (AMPBench-MT style)

Following the leakage analysis in AMPBench-MT (arXiv:2607.25518):

1. Each sequence is assigned to its **composition family** (the generating family in §2.1; in a real dataset one would use e.g. BLAST/HHblits clustering or a fixed-similarity threshold).
2. Families (not sequences) are allocated to train / val / test at 70 / 10 / 20, so no family appears in more than one split.
3. A **leakage audit** is computed and shipped (`split/leakage_audit.json`): maximum pairwise Jaccard index on composition vectors between train and test families, and per-endpoint positive-rate deltas between train and test. In our run: max Jaccard = 0.250 (below the 0.5 similarity threshold of concern) and all endpoint label-rate deltas ≤ 0.013.
4. As a **comparison**, the identical model is trained and evaluated on a plain stratified-random 70/10/20 split. The delta between the two test metrics quantifies the leakage present in the random protocol *on this dataset*.

### 2.5 Evaluation

Per endpoint: AUC-ROC, Matthews correlation coefficient (MCC), accuracy, and positive rate, at threshold 0.5, computed on the held-out split only. Headline metric: **macro AUC across the 5 endpoints on the homology-controlled test set** (3,020 sequences). All numbers are written to `metrics.json` by the training script; the inference CLI reads that file and prints only measured values.

### 2.6 Multi-Objective Composite Score

For candidate prioritization (mirroring the multi-objective ranking in AMPGAN v3 / PepCraft generative campaigns), we define the composite score as the **geometric mean** of favorable endpoint probabilities:

```
score = ( p(GI) · p(Caco-2) · p(BBB) · (1 − p(Ames)) · (1 − p(hERG)) )^(1/5)
```

The geometric mean penalizes any single poor endpoint (a candidate that is well-absorbed but hERG-positive scores low), which matches the "no fatal flaw" logic used when ranking generated peptide candidates before experimental testing.

---

## 3. Results

### 3.1 Measured Performance

All values below are taken from `peptide_admet_model/metrics.json` produced by the released training script.

**Homology-controlled test split (headline; 3,020 sequences):**

| Endpoint | AUC | MCC | Accuracy | Positive rate |
|---|---|---|---|---|
| GI absorption | 0.8810 | 0.4457 | 0.8037 | 0.132 |
| Caco-2 permeability | 0.8882 | 0.5930 | 0.8094 | 0.319 |
| BBB penetration | 0.9070 | 0.4575 | 0.8367 | 0.105 |
| Ames mutagenicity | 0.8011 | 0.3418 | 0.7016 | 0.171 |
| hERG inhibition | 0.8645 | 0.5261 | 0.7665 | 0.299 |
| **Macro AUC** | **0.8684** | — | **mean 0.7836** | — |

**Homology-controlled validation split:** macro AUC 0.8705 (per-endpoint AUC 0.8568–0.9200).

### 3.2 Dual-Split Comparison (the leakage question)

| Protocol | Macro AUC (test) | Mean accuracy |
|---|---|---|
| Homology-controlled (headline) | 0.8684 | 0.7836 |
| Stratified random (comparison) | 0.8688 | 0.7850 |
| **Delta (random − homology)** | **+0.0004** | +0.0014 |

The near-zero delta is informative in both directions. On *this* demo set, the random split leaks almost nothing because the 10,000 composition families are spread thinly enough that a random draw rarely re-presents the exact same composition region with strong overlap. The protocol matters — and must be audited — precisely when families are large or the dataset is built by mutating a small seed set, which is the regime where AMPBench-MT documents inflation. We ship the audit so the reader can check the regime, rather than trusting an unexamined split.

### 3.3 Why the Numbers Are 0.8–0.9, Not 0.99

The labels are, by construction, noisy thresholded linear functions of a handful of physicochemical features. A model restricted to composition-level features can in principle recover most of that signal, and the residual 0.1–0.2 AUC gap to 1.0 is exactly the label noise plus the information lost by discarding sequence *order* (DPC captures only bigram frequency, not context). We consider this agreement between label-model complexity and measured performance the strongest evidence that the pipeline is honest: the model is neither overfitting (homology ≈ random) nor underfitting (AUC well above 0.5 at every endpoint).

### 3.4 Multi-Objective Ranking (demo)

Ranking the five demo sequences in `test_sequences.txt` by composite score separates the clearly hydrophobic, uncharged candidate (top; high GI/Caco-2/BBB probabilities but elevated hERG probability pulling the geometric mean down) from the strongly charged candidates (bottom; poor permeability). The ranking behaves qualitatively as the literature would predict for passive permeability, which is the intended use of the demo set.

---

## 4. Discussion

### 4.1 What this contribution is

A **reproducibility and evaluation-protocol contribution**: a regenerable dataset, an audited homology-controlled split, a shared model definition, and measured-only metrics, packaged so that every number in the paper is re-derivable from the repository. It is *not* a claim of real-peptide ADMET accuracy, and the abstract, TOC, and repository documentation are written to that standard.

### 4.2 Relation to 2026 work

- **AMPBench-MT (arXiv:2607.25518)**: our split + audit implements the leakage controls it advocates for AMP/ADMET evaluation; our §3.2 reports the split-protocol delta it calls for.
- **AMPGAN v3 / PepCraft (arXiv, 2026-06)**: their generative campaigns rank candidates by multi-objective criteria tied to experimental MIC; our composite score (§2.6) is the in-silico analogue for a 5-endpoint ADMET panel.
- **ApexGO (Nat. Mach. Intell., 2026-05)**: generative redesign of antibiotic molecules with honest validation gates — the same "validate before claiming" stance we adopt.
- **Integrated agentic peptide pipelines (npj Drug Discovery, 2026-05)**: end-to-end campaigns that combine LLM-planned experiments with ML models; our pipeline is deliberately simple and fully deterministic so it can serve as a *verification baseline* inside such campaigns.
- **Genotypic Triggers (2026-08)**: shows safety blind spots arise from missing pharmacogenomic endpoint dimensions; we add toxicogenomics endpoints to the roadmap rather than silently omitting them.

### 4.3 Practical guidance we recommend

1. Ship the dataset (or its exact generator) with the code — never reference a CSV the repository does not contain.
2. Audit and publish the split's leakage (similarity statistics + label-rate deltas), not just its stratification.
3. Use one shared model class for training and inference; release `metrics.json` and have the CLI print measured values only.
4. Report the random-vs-controlled-split delta whenever the dataset contains near-duplicate or family-structured sequences.

### 4.4 Limitations

1. **Synthetic labels.** Every number here measures the pipeline, not biology. Real-peptide performance must be re-measured on experimental data before any accuracy claim.
2. **Five endpoints only.** No toxicogenomic / pharmacogenomic, immunogenicity, or protease-stability endpoints (the blind-spot class documented by Genotypic Triggers).
3. **Composition-level features.** AAC/DPC discard order beyond bigrams; sequence-order-sensitive properties (protease cleavage context, local membrane-interaction motifs) are out of reach for this model class.
4. **Length range.** The demo set spans 10–30 aa; behavior outside that range is unvalidated.
5. **No wet-lab validation.** Nothing here has been confirmed experimentally.

### 4.5 Future directions

Retrain on licensed experimental peptide ADMET data with the *same* split/audit/reporting protocol; extend features with a frozen protein-language-model embedding (e.g., ProtGPT2-style soft-prompt embeddings as used in the npj Drug Discovery 2026 pipeline) or a GNN backbone for order sensitivity; add toxicogenomic and stability endpoints; and expose the composite score as an objective for generative peptide design in the AMPGAN v3 / PepCraft style.

---

## 5. Conclusions

We replaced a submission package whose metrics could not be reproduced from its artifacts with a fully reproducible pipeline: a regenerable synthetic demo set with provenance, an AMPBench-MT-style homology-controlled split with a shipped leakage audit, and a shared multi-task PyTorch MLP whose measured performance (macro AUC 0.8684, mean accuracy 0.7836, on the homology-controlled split; 0.8688 random) we report without inflation. The near-parity between split protocols on this dataset, together with the audit, is our central demonstration: peptide ADMET numbers should be reported with their split provenance, and honest numbers on a demo set are the appropriate claim until real data arrive.

**Availability**: all code, the trained model (144,133 parameters), the scaler, and `metrics.json` are at https://github.com/c00jsw00/openclaw-peptide-admet.

---

## 6. Acknowledgments

We acknowledge the AMPBench-MT authors and the 2026 generative-AMP communities for the evaluation-integrity standards adopted here.

---

## 7. References

(1–17 as in v1.0, condensed: peptide therapeutics background, ADMET tool literature, classical QSAR/deep-learning comparisons.)

18. **AMPBench-MT**: Multi-task benchmarking for antimicrobial peptide prediction: the case for homology-controlled evaluation. *arXiv:2607.25518* (2026).
19. **AMPGAN v3 / PepCraft**: Generative redesign of antimicrobial peptides with multi-objective candidate ranking and wet-lab MIC validation. *arXiv* (2026).
20. **ApexGO**: Generative redesign of antibiotic scaffolds with validation-gated evaluation. *Nature Machine Intelligence* (2026-05).
21. **Integrated agentic peptide-discovery pipeline** (ProtGPT2 soft-prompt integration, LLM-planned experiments). *npj Drug Discovery* (2026-05).
22. **Genotypic Triggers**: pharmacogenomic "back doors" as a safety blind spot in polypharmacy risk prediction. (2026-08).

---

## Supporting Information Available

- **S1.** `data/peptide_admet_demo.meta.json` — dataset provenance and label model.
- **S2.** `data/split/leakage_audit.json` — split leakage audit (max Jaccard, endpoint label-rate deltas).
- **S3.** `peptide_admet_model/metrics.json` — all measured per-endpoint metrics, both splits.
- **S4.** Example predictions for `test_sequences.txt` (composite-score ranking).

---

## Author Information

**Corresponding Author**: Pinwan (品丸), OpenClaw Team.

**Data and Code Availability**: https://github.com/c00jsw00/openclaw-peptide-admet

**Submission positioning**: benchmark / reproducibility-protocol contribution. We do **not** recommend submitting this as a "real-peptide accuracy" paper; the appropriate venue framing is a methods/benchmark article (e.g., JCIM's benchmark track or a workshop on ML evaluation integrity) pending real-data retraining.

---

**Manuscript prepared**: 2026-08-24
**Version**: 2.0 (integrity revision; replaces 2026-03-24 v1.0)
**Status**: internally consistent; submission framing adjusted per §"Submission positioning"
