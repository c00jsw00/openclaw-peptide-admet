#!/usr/bin/env python3
"""
ingest_external.py
==================

External-data ingestion for the openclaw peptide ADMET pipeline (v3.0).

This is the entry point for the "bigger training set" extension. It takes
one or more external CSVs, validates them, maps their columns onto the 9
standard endpoint labels, attempts sequence recovery when only SMILES are
given, de-duplicates, stamps provenance, and emits a single clean CSV ready
for ``prepare_data.py --merge``.

HONESTY RULES (deliberate, do not loosen):
  * Rows with NO usable sequence (sequence column missing AND SMILES->sequence
    conversion failed or returned a non-peptide) are EXCLUDED and listed in
    the audit -- never silently dropped, never faked.
  * SMILES-only rows that DO convert are kept but stamped
    ``sequence_provenance='smiles_inferred'`` so downstream docs can report
    that the sequence was reverse-engineered from the structure, not
    measured.
  * Label columns absent from the source become NaN (partial labels) -- the
    mixed model's per-endpoint masks then simply do not train that endpoint
    on those rows. This is the pepADMET partial-label mechanism.

Two input flavours are auto-detected:
  1. sequence-based:  has a ``sequence`` column (one-letter Aa, no H-termini
     required -- both are normalised).
  2. SMILES-based (pepADMET flavour): has ``SMILES`` + ``toxicity_nontoxicity``
     / ``toxicity_type_class`` / ``neurotoxicity_type_class`` / ``HC50``.

Usage:
  python ingest_external.py --input a.csv --source external_a \
      --input b.csv --source pepadmet_sample \
      --output data/external_ingested.csv

  # optional explicit column mapping when the source uses custom names:
  python ingest_external.py --input x.csv --source x \
      --map GI_absorption=gi_col hERG_inhibition=herg_col \
      --output data/external_ingested.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

from endpoint_config import (
    ENDPOINTS,
    ENDPOINT_NAMES,
    ENDPOINT_BY_NAME,
    SEQUENCE_MIN_LEN,
    SEQUENCE_MAX_LEN,
    BINARY_NAMES,
    MULTICLASS_NAMES,
    REGRESSION_NAMES,
    VALID_AA,
)

# pepADMET native column -> canonical endpoint (used when no explicit --map)
PEPADMET_FIELD_MAP = {
    e.pep_column: e.name for e in ENDPOINTS if e.pep_column is not None
}
SEQ_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def normalise_sequence(seq: str) -> str:
    """Strip H-termini / whitespace / lowercase -> canonical one-letter form."""
    if seq is None:
        return ""
    s = str(seq).strip().upper()
    s = s.replace("H2N-", "").replace("H2N", "").replace("-OH", "").replace("-OH", "")
    s = re.sub(r"\s+", "", s)
    # remove any remaining non-Aa chars (e.g. '*', 'X', '-')
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s)
    return s


def _smiles_to_sequence(smiles: str):
    from smiles_to_sequence import smiles_to_sequence as _stq
    return _stq(smiles)


def ingest_one(
    path: str,
    source: str,
    column_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ingest a single CSV. Returns (df with standard label cols, audit_dict)."""
    column_map = column_map or {}
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)

    audit: dict = {
        "source": source,
        "file": path,
        "rows_raw": int(n_raw),
        "rows_sequence_based": 0,
        "rows_smiles_converted": 0,
        "rows_smiles_failed": 0,
        "rows_deduped": 0,
        "rows_length_out": 0,
        "rows_kept": 0,
        "rows_excluded_no_sequence": 0,
        "excluded_no_sequence": [],
    }

    # ---- 1) build the sequence column ------------------------------------
    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].map(normalise_sequence)
        df["sequence_provenance"] = "external_sequence"
    elif "SMILES" in df.columns or "smiles" in df.columns:
        sm_col = "SMILES" if "SMILES" in df.columns else "smiles"
        seqs, provs, fails = [], [], 0
        for sm in df[sm_col].astype(str):
            res = _smiles_to_sequence(sm)   # SeqResult
            if res.matched and res.sequence and SEQ_RE.match(res.sequence):
                seqs.append(res.sequence)
                provs.append("smiles_inferred")
            else:
                seqs.append("")
                provs.append("smiles_failed")
                fails += 1
        # build the two columns at once to avoid fragmentation warnings
        seq_df = pd.DataFrame({"sequence": seqs, "sequence_provenance": provs},
                              index=df.index)
        df = pd.concat([df, seq_df], axis=1)
        audit["rows_smiles_converted"] = int(sum(1 for p in provs if p == "smiles_inferred"))
        audit["rows_smiles_failed"] = fails
    else:
        raise ValueError(
            f"{path}: no 'sequence' and no 'SMILES' column; cannot ingest."
        )

    df["sequence"] = df["sequence"].fillna("").astype(str)
    n_seq = int((df["sequence"] != "").sum())
    audit["rows_sequence_based"] = n_seq - audit["rows_smiles_converted"]

    # ---- 2) drop rows with no usable sequence (honest exclusion) ---------
    no_seq = df[df["sequence"] == ""]
    if len(no_seq):
        # record up to 50 examples for the audit
        recs = []
        for _, r in no_seq.head(50).iterrows():
            rec = {"sequence": ""}
            smc = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else None)
            if smc:
                rec["SMILES"] = str(r[smc])[:80]
            rec["sequence_provenance"] = str(r.get("sequence_provenance", ""))
            recs.append(rec)
        audit["rows_excluded_no_sequence"] = int(len(no_seq))
        audit["excluded_no_sequence"] = recs
    df = df[df["sequence"] != ""].copy()

    # ---- 3) length filter -------------------------------------------------
    ok_len = df["sequence"].str.len().between(SEQUENCE_MIN_LEN, SEQUENCE_MAX_LEN)
    audit["rows_length_out"] = int((~ok_len).sum())
    df = df[ok_len].copy()

    # ---- 4) map label columns onto the standard 9 endpoints ---------------
    for ep in ENDPOINT_NAMES:
        src_col = column_map.get(ep, ep)  # explicit map, else same-name
        if src_col in df.columns:
            df[ep] = pd.to_numeric(df[src_col], errors="coerce")
        else:
            df[ep] = pd.NA

    # pepADMET field-name fallback (field->name map; only for pepADMET eps)
    if not column_map:
        for pep_field, ep in PEPADMET_FIELD_MAP.items():
            if pep_field in df.columns and df[ep].isna().all():
                df[ep] = pd.to_numeric(df[pep_field], errors="coerce")

    # ---- 5) label sanity: binary in {0,1}, classes in range --------------
    for ep in BINARY_NAMES:
        df.loc[~df[ep].isin([0, 1]), ep] = pd.NA
    for ep in MULTICLASS_NAMES:
        k = int(ENDPOINT_BY_NAME[ep].num_classes)
        df.loc[~df[ep].isin(range(k)), ep] = pd.NA
    for ep in REGRESSION_NAMES:
        df[ep] = pd.to_numeric(df[ep], errors="coerce")

    # ---- 6) provenance -----------------------------------------------------
    df["data_origin"] = source
    df["group"] = "external"  # placeholder; homology_split re-groups by family

    # ---- 7) de-duplicate by canonical sequence ----------------------------
    before = len(df)
    df = df.drop_duplicates(subset="sequence", keep="first").copy()
    audit["rows_deduped"] = int(before - len(df))

    # ---- 8) keep only standard columns ------------------------------------
    keep = ["sequence", "sequence_provenance", "data_origin", "group"] + ENDPOINT_NAMES
    df = df[keep].copy()
    df = df.reset_index(drop=True)
    audit["rows_kept"] = int(len(df))

    return df, audit


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest external peptide ADMET CSVs.")
    ap.add_argument("--input", action="append", default=[], help="external CSV (repeatable)")
    ap.add_argument("--source", action="append", default=[], help="provenance name (repeatable, matches --input order)")
    ap.add_argument("--map", action="append", default=[], help="EP=SRC_COL (repeatable) explicit label mapping")
    ap.add_argument("--output", default="data/external_ingested.csv")
    args = ap.parse_args()

    if not args.input:
        print("No --input given. Provide at least one external CSV.")
        return
    if len(args.source) < len(args.input):
        # auto-name sources
        args.source += [f"external_{i+1}" for i in range(len(args.input) - len(args.source))]
    source_map = dict(zip(args.input, args.source))

    column_map = {}
    for m in args.map:
        if "=" in m:
            k, v = m.split("=", 1)
            column_map[k.strip()] = v.strip()

    frames, audits = [], []
    for inp in args.input:
        src = source_map.get(inp, "external")
        print(f"\n[ingest] {src}: {inp}")
        df, audit = ingest_one(inp, src, column_map)
        frames.append(df)
        audits.append(audit)
        print(f"  kept {audit['rows_kept']} / {audit['rows_raw']} rows "
              f"(smiles_converted={audit['rows_smiles_converted']}, "
              f"smiles_failed={audit['rows_smiles_failed']}, "
              f"excluded_no_sequence={audit['rows_excluded_no_sequence']}, "
              f"deduped={audit['rows_deduped']})")

    merged = pd.concat(frames, ignore_index=True)
    # global de-dup across sources
    before = len(merged)
    merged = merged.drop_duplicates(subset="sequence", keep="first").copy()
    print(f"\n[ingest] merged {len(frames)} sources -> {len(merged)} unique rows "
          f"(cross-source dedup removed {before - len(merged)})")

    # per-endpoint label availability
    print("\nPer-endpoint label coverage (rows with a non-NaN label):")
    for ep in ENDPOINT_NAMES:
        n = int(merged[ep].notna().sum())
        print(f"  {ep:<22} {n:>7} rows")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\n[ingest] wrote {args.output}")

    audit_out = {
        "merged_rows": int(len(merged)),
        "sources": audits,
        "endpoint_label_coverage": {
            ep: int(merged[ep].notna().sum()) for ep in ENDPOINT_NAMES
        },
    }
    audit_path = os.path.join(os.path.dirname(args.output) or ".", "ingest_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_out, f, indent=2, ensure_ascii=False)
    print(f"[ingest] wrote {audit_path}")


if __name__ == "__main__":
    main()
