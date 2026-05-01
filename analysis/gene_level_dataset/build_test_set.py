#!/usr/bin/env python3
"""
Build a test set combining:
  1. 3' UTR sequences (UCSC RefSeq, hg19)
  2. Measured miRNA fold-changes (mirna_fcs.csv)
  3. TargetScan context++ predictions (Conserved + Nonconserved Site Context Scores)

Two modes for joining TargetScan scores (--ts-join):
  gene_symbol: Sum TS scores per (Gene Symbol, miRNA). Simple but may mix transcripts.
  ensembl:     Look up Gene Symbol in TS Gene_info.txt to find the representative
               Ensembl transcript, then use that transcript's scores.
               Priorities: representative > highest 3P-seq tags > longest UTR > Gene Symbol fallback.

Outputs:
  - {prefix}_wide.tsv:  One row per gene, all 7 miRNA FCs + TargetScan scores as columns
  - {prefix}_long.tsv:  One row per (UTR, miRNA, log2fc, TS_score) — ready for model input

Usage:
    # Gene Symbol mode (default):
    python build_test_set.py \\
        --utrs 3utr_sequences.txt \\
        --fold-change mirna_fcs.csv \\
        --scores Conserved.txt Nonconserved.txt \\
        --output test_set

    # Ensembl mode (matches TS methodology):
    python build_test_set.py \\
        --utrs 3utr_sequences.txt \\
        --fold-change mirna_fcs.csv \\
        --scores Conserved.txt Nonconserved.txt \\
        --ts-join ensembl --gene-info Gene_info.txt \\
        --output test_set
"""

import argparse
import csv
import sys
from collections import defaultdict
from io import StringIO

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEST_MIRNAS = [
    "hsa-miR-16-5p",
    "hsa-miR-106b-5p",
    "hsa-miR-200a-3p",
    "hsa-miR-200b-3p",
    "hsa-miR-215-5p",
    "hsa-let-7c-5p",
    "hsa-miR-103a-3p",
]

HUMAN_TAX_ID = "9606"

# ---------------------------------------------------------------------------
# Excel date corruption fix
# ---------------------------------------------------------------------------

# Map Excel-corrupted gene symbols back to original gene names.
# These use pre-2020 HGNC symbols (MARCH/SEPT) since the Agarwal 2015
# data and TargetScan 7/8 predate the HGNC rename to MARCHF/SEPTIN.
# Post-2020 HGNC names shown in comments for reference.
EXCEL_DATE_TO_GENE = {
    "1-Sep":  "SEPT1",     # now SEPTIN1
    "2-Sep":  "SEPT2",     # now SEPTIN2
    "3-Sep":  "SEPT3",     # now SEPTIN3
    "5-Sep":  "SEPT5",     # now SEPTIN5
    "6-Sep":  "SEPT6",     # now SEPTIN6
    "7-Sep":  "SEPT7",     # now SEPTIN7
    "9-Sep":  "SEPT9",     # now SEPTIN9
    "10-Sep": "SEPT10",    # now SEPTIN10
    "11-Sep": "SEPT11",    # now SEPTIN11
    "2-Mar":  "MARCH2",    # now MARCHF2 (NM_001005416)
    "5-Mar":  "MARCH5",    # now MARCHF5
    "6-Mar":  "MARCH6",    # now MARCHF6
    "7-Mar":  "MARCH7",    # now MARCHF7
    "8-Mar":  "MARCH8",    # now MARCHF8
    "9-Mar":  "MARCH9",    # now MARCHF9
}


TARGETSCAN_COLUMN_TO_SEQUENCE = {
    "hsa-miR-16-5p":    "TAGCAGCACGTAAATATTGGCG",
    "hsa-miR-106b-5p":  "TAAAGTGCTGACAGTGCAGAT",
    "hsa-miR-200a-3p":  "TAACACTGTCTGGTAACGATGT",
    "hsa-miR-200b-3p":  "TAATACTGCCTGGTAATGATGA",
    "hsa-miR-215-5p":   "ATGACCTATGAATTGACAGAC",
    "hsa-let-7c-5p":    "TGAGGTAGTAGGTTGTATGGTT",
    "hsa-miR-103a-3p":  "AGCAGCATTGTACAGGGCTATGA",
}


MIN_UTR_LENGTH = 25  # skip very short UTRs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utrs", required=True,
                   help="UCSC RefSeq 3' UTR sequences (FASTA-like txt from Table Browser)")
    p.add_argument("--fold-change", required=True,
                   help="mirna_fcs.csv with measured log2 fold-changes")
    p.add_argument("--scores", default=None, nargs="+",
                   help="TargetScan context++ score file(s), e.g. Conserved + Nonconserved "
                        "(optional, multiple files are summed together)")
    p.add_argument("--ts-join", choices=["gene_symbol", "ensembl"], default="gene_symbol",
                   help="How to join TS scores: 'gene_symbol' (by Gene Symbol) or "
                        "'ensembl' (by Ensembl transcript ID via UCSC id_map, "
                        "using TS representative transcript)")
    p.add_argument("--id-map", default=None,
                   help="UCSC id_map file mapping RefSeq → Ensembl IDs "
                        "(required for --ts-join ensembl)")
    p.add_argument("--gene-info", default=None,
                   help="TargetScan Gene_info.txt with representative transcript flags "
                        "(required for --ts-join ensembl)")
    p.add_argument("--output", default="test_set",
                   help="Output file prefix (default: test_set)")
    args = p.parse_args()

    if args.ts_join == "ensembl":
        if args.gene_info is None:
            p.error("--ts-join ensembl requires --gene-info")

    return args


# ---------------------------------------------------------------------------
# 1. Load UCSC RefSeq UTR sequences
# ---------------------------------------------------------------------------

def load_ucsc_utrs(path):
    """
    Parse UCSC Table Browser 3' UTR sequence output.
    
    Expected format (tab-separated or FASTA-like with headers like):
      >hg19_ncbiRefSeq_NM_000017.3 range=chr12:...
      ACGTACGT...

    Or tab-delimited with columns including an ID and sequence.
    
    We try both formats. Returns dict: {refseq_id_no_version: (refseq_id_versioned, sequence)}
    keeping the longest UTR per RefSeq ID (no version).
    """
    utrs = {}  # refseq_id -> (versioned_id, sequence, length)

    print(f"Loading UTR sequences from: {path}")

    with open(path, "r") as f:
        first_line = f.readline()
        f.seek(0)

        if first_line.startswith(">"):
            # FASTA format
            current_id = None
            current_chrom = None
            current_start = None
            current_end = None
            current_strand = None
            current_seq_parts = []

            def _parse_fasta_header(header_line):
                """
                Parse UCSC Table Browser FASTA header like:
                >hg19_ncbiRefSeq_NM_000017.3 range=chr12:121163570-121164929 5'pad=0 3'pad=0 strand=- repeatMasking=none
                Returns: (versioned_id, chrom, start, end, strand)
                """
                fields = header_line[1:].split()
                name = fields[0]
                # Extract NM_XXXXXX.X
                name_parts = name.split("_")
                vid = None
                for i, p in enumerate(name_parts):
                    if p in ("NM", "NR"):
                        vid = f"{p}_{name_parts[i+1]}" if i + 1 < len(name_parts) else None
                        break
                if vid is None:
                    vid = name

                chrom, start, end, strand = None, None, None, None
                for field in fields[1:]:
                    if field.startswith("range="):
                        # range=chr12:121163570-121164929
                        range_val = field.split("=", 1)[1]
                        try:
                            chrom_part, coords = range_val.rsplit(":", 1)
                            s, e = coords.split("-")
                            chrom = chrom_part
                            start = int(s)
                            end = int(e)
                        except (ValueError, IndexError):
                            pass
                    elif field.startswith("strand="):
                        strand = field.split("=", 1)[1]
                return vid, chrom, start, end, strand

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq_parts:
                        _store_utr(utrs, current_id, "".join(current_seq_parts),
                                   current_chrom, current_start, current_end, current_strand)
                    current_id, current_chrom, current_start, current_end, current_strand = \
                        _parse_fasta_header(line)
                    current_seq_parts = []
                elif line:
                    current_seq_parts.append(line)
            if current_id and current_seq_parts:
                _store_utr(utrs, current_id, "".join(current_seq_parts),
                           current_chrom, current_start, current_end, current_strand)
        else:
            # Tab-delimited format — look for columns with RefSeq ID and Sequence
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header is None:
                print("ERROR: Empty UTR file")
                sys.exit(1)

            # Try to find ID and sequence columns
            col_map = {name.strip().lower(): i for i, name in enumerate(header)}
            id_col = None
            seq_col = None

            for name, idx in col_map.items():
                if "id" in name and ("refseq" in name or "transcript" in name or name == "id"):
                    id_col = idx
                if "sequence" in name or "seq" in name:
                    seq_col = idx

            if id_col is None or seq_col is None:
                # Fallback: assume first column is ID, last is sequence
                id_col = 0
                seq_col = len(header) - 1
                print(f"  Warning: Could not identify columns by name. Using col {id_col} as ID, col {seq_col} as sequence.")

            for row in reader:
                if len(row) <= max(id_col, seq_col):
                    continue
                versioned_id = row[id_col].strip()
                seq = row[seq_col].strip().upper()
                if seq and versioned_id:
                    _store_utr(utrs, versioned_id, seq)

    print(f"  Loaded {len(utrs):,} unique UTRs (longest per RefSeq ID)")
    return utrs


def _store_utr(utrs, versioned_id, sequence, chrom=None, start=None, end=None, strand=None):
    """Store UTR, keeping the longest per base RefSeq ID (no version)."""
    base_id = versioned_id.split(".")[0]
    seq_len = len(sequence)
    if base_id not in utrs or seq_len > utrs[base_id][2]:
        utrs[base_id] = (versioned_id, sequence, seq_len, chrom, start, end, strand)


# ---------------------------------------------------------------------------
# 2. Load fold-changes
# ---------------------------------------------------------------------------

def load_fold_changes(path):
    """
    Load mirna_fcs.csv.
    Returns list of dicts with keys: RefSeq ID, Gene symbol, and per-miRNA FCs.
    Also returns the miRNA column names.
    """
    print(f"\nLoading fold-changes from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")

    lines = content.strip().split("\n")
    header = lines[0].split(",")
    mirna_cols = [c.strip() for c in header[2:]]

    records = []
    corrected = 0
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) < len(header):
            continue
        refseq_id = fields[0].strip()
        gene_symbol = fields[1].strip()

        # Fix Excel date-corrupted gene symbols
        if gene_symbol in EXCEL_DATE_TO_GENE:
            old = gene_symbol
            gene_symbol = EXCEL_DATE_TO_GENE[gene_symbol]
            corrected += 1
            if corrected <= 5:
                print(f"    Fixed: {old} -> {gene_symbol} ({refseq_id})")

        fcs = {}
        for i, mirna in enumerate(mirna_cols):
            val = fields[i + 2].strip()
            if val in ("N/A", "", "NA"):
                fcs[mirna] = None
            else:
                fcs[mirna] = float(val)
        records.append({
            "refseq_id": refseq_id,
            "gene_symbol": gene_symbol,
            "fcs": fcs,
        })

    print(f"  {len(records):,} genes, {len(mirna_cols)} miRNAs: {mirna_cols}")
    if corrected:
        print(f"  Fixed {corrected} Excel date-corrupted gene symbols")
    return records, mirna_cols


# ---------------------------------------------------------------------------
# 3. Load TargetScan context++ scores (optional)
# ---------------------------------------------------------------------------

def load_targetscan_scores(paths):
    """
    Stream through TargetScan context++ score file(s).
    Accepts Conserved and/or Nonconserved Site Context Scores files.
    Filter for human (Tax ID 9606).

    From ALL human rows:
      - Collect max(UTR end) per Ensembl transcript (proxy for UTR length)
      - Collect the set of all Gene Symbols and Ensembl IDs TS has modeled

    From rows matching the 7 test miRNAs:
      - Sum scores per (Gene Symbol, miRNA) and per (Ensembl ID, miRNA)
    
    Returns: {
        "scores_by_gs": (context_dict, weighted_dict),
        "scores_by_ens": (context_dict, weighted_dict),
        "utr_length_by_ens": {ensembl_id: max_utr_end},
        "all_human_gene_symbols": set,
        "all_human_ensembl_ids": set,
    }
    """
    if paths is None:
        return None

    test_mirna_set = set(TEST_MIRNAS)
    # Gene Symbol keyed
    gs_context = defaultdict(float)
    gs_weighted = defaultdict(float)
    # Ensembl ID (no version) keyed
    ens_context = defaultdict(float)
    ens_weighted = defaultdict(float)
    # UTR length per Ensembl transcript (from all human rows)
    utr_length_by_ens = defaultdict(int)
    # All human genes/transcripts TS has modeled
    all_human_gene_symbols = set()
    all_human_ensembl_ids = set()

    grand_total = 0
    grand_human = 0
    grand_matched = 0
    grand_skipped_null = 0

    for path in paths:
        total = 0
        human = 0
        matched = 0
        skipped_null = 0

        print(f"\nLoading TargetScan scores from: {path}")
        print(f"  Filtering for Tax ID = {HUMAN_TAX_ID}, {len(TEST_MIRNAS)} miRNAs...")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            col_map = {name.strip(): i for i, name in enumerate(header)}

            gene_sym_col = col_map.get("Gene Symbol")
            tax_id_col = col_map.get("Gene Tax ID")
            mirna_col = col_map.get("miRNA")
            wcs_col = col_map.get("weighted context++ score")
            cs_col = col_map.get("context++ score")
            tx_col = col_map.get("Transcript ID")
            utr_end_col = col_map.get("UTR end")

            required = {"Gene Symbol": gene_sym_col, "Gene Tax ID": tax_id_col,
                         "miRNA": mirna_col, "weighted context++ score": wcs_col,
                         "context++ score": cs_col, "Transcript ID": tx_col}
            missing = [k for k, v in required.items() if v is None]
            if missing:
                print(f"  ERROR: Missing columns: {missing}")
                print(f"  Available: {list(col_map.keys())}")
                sys.exit(1)

            if utr_end_col is None:
                print(f"  WARNING: 'UTR end' column not found — UTR length heuristic disabled")

            max_col = max(v for v in required.values() if v is not None)

            for row in reader:
                total += 1
                if total % 5_000_000 == 0:
                    print(f"    {total:,} lines processed...")

                if len(row) <= max_col:
                    continue
                tax_raw = row[tax_id_col].strip()
                if tax_raw != HUMAN_TAX_ID and tax_raw != HUMAN_TAX_ID + ".0":
                    continue
                human += 1

                gene_symbol = row[gene_sym_col].strip()
                tx_id_raw = row[tx_col].strip()
                ensembl_base = tx_id_raw.split(".")[0] if tx_id_raw else ""

                # Collect from ALL human rows: gene sets and UTR lengths
                if gene_symbol:
                    all_human_gene_symbols.add(gene_symbol)
                if ensembl_base:
                    all_human_ensembl_ids.add(ensembl_base)
                    if utr_end_col is not None and utr_end_col < len(row):
                        try:
                            utr_end = int(row[utr_end_col].strip())
                            if utr_end > utr_length_by_ens[ensembl_base]:
                                utr_length_by_ens[ensembl_base] = utr_end
                        except (ValueError, TypeError):
                            pass

                # Sum scores only for our 7 test miRNAs
                mirna = row[mirna_col].strip()
                if mirna not in test_mirna_set:
                    continue
                matched += 1

                wcs = row[wcs_col].strip()
                cs = row[cs_col].strip()
                if cs in ("NULL", "", "NA"):
                    skipped_null += 1
                    continue

                gs_key = (gene_symbol, mirna)
                if cs not in ("NULL", "", "NA"):
                    gs_context[gs_key] += float(cs)
                if wcs not in ("NULL", "", "NA"):
                    gs_weighted[gs_key] += float(wcs)

                if ensembl_base:
                    ens_key = (ensembl_base, mirna)
                    if cs not in ("NULL", "", "NA"):
                        ens_context[ens_key] += float(cs)
                    if wcs not in ("NULL", "", "NA"):
                        ens_weighted[ens_key] += float(wcs)

        print(f"  Total lines:          {total:,}")
        print(f"  Human lines:          {human:,}")
        print(f"  Matching miRNA lines: {matched:,}")
        print(f"  Skipped NULL:         {skipped_null:,}")

        grand_total += total
        grand_human += human
        grand_matched += matched
        grand_skipped_null += skipped_null

    if len(paths) > 1:
        print(f"\n  COMBINED across {len(paths)} files:")
        print(f"    Total lines:          {grand_total:,}")
        print(f"    Human lines:          {grand_human:,}")
        print(f"    Matching miRNA lines: {grand_matched:,}")
        print(f"    Skipped NULL:         {grand_skipped_null:,}")
    print(f"  Unique (gene_symbol, miRNA) with context++ score:    {len(gs_context):,}")
    print(f"  Unique (gene_symbol, miRNA) with weighted ctx++ score: {len(gs_weighted):,}")
    print(f"  Unique (ensembl_id, miRNA) with context++ score:     {len(ens_context):,}")
    print(f"  Unique (ensembl_id, miRNA) with weighted ctx++ score:  {len(ens_weighted):,}")
    print(f"  Human gene symbols in TS predictions (all miRNAs):   {len(all_human_gene_symbols):,}")
    print(f"  Human Ensembl IDs in TS predictions (all miRNAs):    {len(all_human_ensembl_ids):,}")
    print(f"  Ensembl IDs with UTR length info:                    {len(utr_length_by_ens):,}")

    return {
        "scores_by_gs": (dict(gs_context), dict(gs_weighted)),
        "scores_by_ens": (dict(ens_context), dict(ens_weighted)),
        "utr_length_by_ens": dict(utr_length_by_ens),
        "all_human_gene_symbols": all_human_gene_symbols,
        "all_human_ensembl_ids": all_human_ensembl_ids,
    }


# ---------------------------------------------------------------------------
# 3b. Load UCSC id_map (for --ts-join ensembl)
# ---------------------------------------------------------------------------

def load_ucsc_id_map(path):
    """
    Load UCSC id_map file mapping RefSeq IDs to Ensembl transcript IDs.

    The file has no header. Columns (positional, matching the notebook):
        0: knownGene.name, 1: knownGene.chrom,
        2: kgAlias.kgID, 3: kgAlias.alias,
        4: kgXref.kgID, 5: kgXref.mRNA,
        6: kgXref.geneSymbol, 7: kgXref.refseq,
        8: knownToEnsembl.name, 9: knownToEnsembl.value,
        10: knownToRefSeq.name, 11: knownToRefSeq.value

    RefSeq ID = knownToRefSeq.value (col 11), falling back to kgXref.refseq (col 7).
    Ensembl ID = knownToEnsembl.value (col 9).

    Returns: dict {refseq_id (no version) -> list of ensembl_transcript_ids (no version)}
    """
    print(f"\nLoading UCSC id_map from: {path}")

    COL_REFSEQ_1 = 11   # knownToRefSeq.value (preferred)
    COL_REFSEQ_2 = 7    # kgXref.refseq (fallback)
    COL_ENSEMBL = 9      # knownToEnsembl.value

    id_map = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= COL_REFSEQ_1:
                continue

            # RefSeq ID: prefer col 11, fall back to col 7
            refseq_raw = fields[COL_REFSEQ_1].strip()
            if not refseq_raw:
                refseq_raw = fields[COL_REFSEQ_2].strip() if len(fields) > COL_REFSEQ_2 else ""
            if not refseq_raw:
                continue

            ensembl_raw = fields[COL_ENSEMBL].strip() if len(fields) > COL_ENSEMBL else ""
            if not ensembl_raw:
                continue

            refseq_base = refseq_raw.split(".")[0]
            ensembl_base = ensembl_raw.split(".")[0]
            id_map[refseq_base].add(ensembl_base)

    # Convert sets to sorted lists
    id_map = {k: sorted(v) for k, v in id_map.items()}

    n_refseq = len(id_map)
    n_ensembl = len(set(e for es in id_map.values() for e in es))
    n_multi = sum(1 for es in id_map.values() if len(es) > 1)
    print(f"  {n_refseq:,} RefSeq IDs mapped to {n_ensembl:,} unique Ensembl IDs")
    print(f"  {n_multi:,} RefSeq IDs map to multiple Ensembl transcripts")

    return id_map


# ---------------------------------------------------------------------------
# 3c. Load TargetScan Gene_info.txt (for --ts-join ensembl)
# ---------------------------------------------------------------------------

def load_ts_gene_info(path):
    """
    Load TargetScan Gene_info.txt.

    Returns:
        by_ensembl: dict {ensembl_id (no version) -> {
            "is_representative": bool, "3p_seq_tags": int, "gene_symbol": str
        }}
        by_gene_symbol: dict {gene_symbol -> list of {
            "ensembl_id": str, "is_representative": bool, "3p_seq_tags": int
        }}  (sorted: representative first, then by 3P-seq tags descending)
    """
    print(f"\nLoading TargetScan Gene_info from: {path}")

    by_ensembl = {}
    by_gene_symbol = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        col_map = {name.strip(): i for i, name in enumerate(header)}

        tx_col = col_map.get("Transcript ID")
        rep_col = col_map.get("Representative transcript?")
        gs_col = col_map.get("Gene symbol")
        tags_col = col_map.get("3P-seq tags")
        species_col = col_map.get("Species ID")

        required = {"Transcript ID": tx_col, "Representative transcript?": rep_col,
                     "Gene symbol": gs_col}
        missing = [k for k, v in required.items() if v is None]
        if missing:
            print(f"  ERROR: Missing columns: {missing}")
            print(f"  Available: {list(col_map.keys())}")
            sys.exit(1)

        n_total = 0
        n_human = 0
        n_representative = 0
        for row in reader:
            if len(row) <= max(v for v in required.values()):
                continue
            n_total += 1

            # Filter to human if Species ID column exists
            if species_col is not None and species_col < len(row):
                species = row[species_col].strip()
                if species not in ("9606", "9606.0"):
                    continue
            n_human += 1

            tx_id = row[tx_col].strip()
            rep_val = row[rep_col].strip()
            gene_sym = row[gs_col].strip()
            tags = 0
            if tags_col is not None and tags_col < len(row):
                try:
                    tags = int(float(row[tags_col].strip()))
                except (ValueError, TypeError):
                    tags = 0

            ensembl_base = tx_id.split(".")[0]
            is_rep = (rep_val == "1" or rep_val.lower() == "true" or rep_val == "1.0")

            if is_rep:
                n_representative += 1

            entry = {
                "ensembl_id": ensembl_base,
                "is_representative": is_rep,
                "3p_seq_tags": tags,
                "gene_symbol": gene_sym,
            }

            by_ensembl[ensembl_base] = entry
            by_gene_symbol[gene_sym].append(entry)

    # Sort per-gene lists: representative first, then by 3P-seq tags descending
    for gs in by_gene_symbol:
        by_gene_symbol[gs].sort(
            key=lambda e: (-int(e["is_representative"]), -e["3p_seq_tags"])
        )

    print(f"  {n_total:,} transcripts total, {n_human:,} human")
    print(f"  {n_representative:,} representative")
    print(f"  {len(by_gene_symbol):,} unique gene symbols")

    return by_ensembl, dict(by_gene_symbol)


# ---------------------------------------------------------------------------
# 4. Build and write the test set
# ---------------------------------------------------------------------------

def build_test_set(utrs, fc_records, mirna_cols,
                   ts_gs_context, ts_gs_weighted,
                   ts_ens_context, ts_ens_weighted,
                   utr_length_by_ens,
                   ts_join_mode,
                   gene_info_by_ensembl, gene_info_by_symbol,
                   output_prefix):
    """
    Join UTRs + fold-changes on RefSeq ID.
    Join TargetScan scores on Gene Symbol or Ensembl transcript ID.
    Write wide and long format outputs with both context++ and weighted context++ scores.
    """
    has_ts = ts_gs_context is not None

    # --- Resolve duplicates: 47 gene symbols map to 2 RefSeq IDs ---
    # Group by gene symbol, keep the RefSeq ID with more non-NA fold-changes
    gene_to_records = defaultdict(list)
    for rec in fc_records:
        gene_to_records[rec["gene_symbol"]].append(rec)

    resolved_records = []
    dup_count = 0
    for gene, recs in gene_to_records.items():
        if len(recs) == 1:
            resolved_records.append(recs[0])
        else:
            dup_count += 1
            # Pick the one with more valid fold-changes
            best = max(recs, key=lambda r: sum(1 for v in r["fcs"].values() if v is not None))
            resolved_records.append(best)

    print(f"\nBuilding test set:")
    print(f"  Input genes:          {len(fc_records):,}")
    print(f"  Duplicate gene symbols resolved: {dup_count}")
    print(f"  Genes after dedup:    {len(resolved_records):,}")

    # --- Join with UTR sequences on RefSeq ID ---
    joined = []
    no_utr = 0
    short_utr = 0

    for rec in resolved_records:
        refseq_base = rec["refseq_id"].split(".")[0]
        utr_entry = utrs.get(refseq_base)
        if utr_entry is None:
            no_utr += 1
            continue
        versioned_id, sequence, seq_len, chrom, start, end, strand = utr_entry
        if seq_len < MIN_UTR_LENGTH:
            short_utr += 1
            continue

        joined.append({
            "refseq_id": rec["refseq_id"],
            "gene_symbol": rec["gene_symbol"],
            "utr_sequence": sequence,
            "utr_length": seq_len,
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "fcs": rec["fcs"],
        })

    print(f"  No UTR found:         {no_utr}")
    print(f"  UTR < {MIN_UTR_LENGTH} nt:          {short_utr}")
    print(f"  Genes with UTR + FC:  {len(joined):,}")

    # --- Resolve Ensembl transcript IDs (for ensembl join mode) ---
    if has_ts and ts_join_mode == "ensembl":
        n_p1 = 0  # Representative via Gene Symbol in Gene_info
        n_p2 = 0  # Highest 3P-seq tags via Gene Symbol in Gene_info
        n_p3 = 0  # Longest UTR via Gene Symbol → Gene_info Ensembl IDs → score file
        n_p4 = 0  # Gene Symbol fallback

        for rec in joined:
            gene_sym = rec["gene_symbol"]

            resolved_ens_id = None
            resolution = None

            if gene_sym in gene_info_by_symbol:
                entries = gene_info_by_symbol[gene_sym]

                # P1: Representative transcript in Gene_info
                for entry in entries:
                    if entry["is_representative"]:
                        resolved_ens_id = entry["ensembl_id"]
                        resolution = "P1_repr"
                        n_p1 += 1
                        break

                # P2: Highest 3P-seq tags in Gene_info
                if resolved_ens_id is None:
                    # entries are already sorted by 3P-seq tags descending
                    resolved_ens_id = entries[0]["ensembl_id"]
                    resolution = "P2_3pseq"
                    n_p2 += 1

                # P3: Longest UTR from score file
                if resolved_ens_id is None:
                    best_len = -1
                    best_eid = None
                    for entry in entries:
                        ulen = utr_length_by_ens.get(entry["ensembl_id"], 0)
                        if ulen > best_len:
                            best_len = ulen
                            best_eid = entry["ensembl_id"]
                    if best_eid is not None and best_len > 0:
                        resolved_ens_id = best_eid
                        resolution = "P3_longest_utr"
                        n_p3 += 1

            # P4: Gene Symbol fallback
            if resolved_ens_id is None:
                resolution = "P4_genesym_fallback"
                n_p4 += 1

            rec["ensembl_id"] = resolved_ens_id
            rec["ensembl_resolution"] = resolution

        print(f"\n  Ensembl ID resolution (--ts-join ensembl):")
        print(f"    P1 - Representative in Gene_info:       {n_p1:,}")
        print(f"    P2 - Highest 3P-seq tags in Gene_info:  {n_p2:,}")
        print(f"    P3 - Longest UTR from score file:       {n_p3:,}")
        print(f"    P4 - Gene Symbol fallback:              {n_p4:,}")

    # --- Helper: look up TS scores for a record ---
    def get_ts_scores(rec, mirna):
        """Return (context++ score, weighted context++ score) for a (record, miRNA) pair."""
        if not has_ts:
            return 0.0, 0.0

        if ts_join_mode == "ensembl":
            ens_id = rec.get("ensembl_id")
            if ens_id is not None:
                ctx = ts_ens_context.get((ens_id, mirna), 0.0)
                wgt = ts_ens_weighted.get((ens_id, mirna), 0.0)
                return ctx, wgt
            else:
                # Fallback to Gene Symbol
                ctx = ts_gs_context.get((rec["gene_symbol"], mirna), 0.0)
                wgt = ts_gs_weighted.get((rec["gene_symbol"], mirna), 0.0)
                return ctx, wgt
        else:
            # gene_symbol mode
            ctx = ts_gs_context.get((rec["gene_symbol"], mirna), 0.0)
            wgt = ts_gs_weighted.get((rec["gene_symbol"], mirna), 0.0)
            return ctx, wgt

    # --- Write WIDE format ---
    wide_path = f"{output_prefix}_wide.tsv"
    with open(wide_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        header = ["RefSeq_ID", "Gene_Symbol", "Chrom", "Start", "End", "Strand",
                  "UTR_length", "UTR_sequence"]
        for m in mirna_cols:
            header.append(f"fc_{m}")
        if has_ts:
            for m in mirna_cols:
                header.append(f"ts_context_{m}")
            for m in mirna_cols:
                header.append(f"ts_weighted_{m}")
        for m in mirna_cols:
            header.append(f"seq_{m}")
        w.writerow(header)

        for rec in sorted(joined, key=lambda r: r["gene_symbol"]):
            row = [
                rec["refseq_id"],
                rec["gene_symbol"],
                rec["chrom"] or "NA",
                rec["start"] if rec["start"] is not None else "NA",
                rec["end"] if rec["end"] is not None else "NA",
                rec["strand"] or "NA",
                rec["utr_length"],
                rec["utr_sequence"],
            ]
            for m in mirna_cols:
                val = rec["fcs"].get(m)
                row.append(f"{val:.6f}" if val is not None else "NA")
            if has_ts:
                for m in mirna_cols:
                    ctx, _ = get_ts_scores(rec, m)
                    row.append(f"{ctx:.6f}")
                for m in mirna_cols:
                    _, wgt = get_ts_scores(rec, m)
                    row.append(f"{wgt:.6f}")
            for m in mirna_cols:
                row.append(TARGETSCAN_COLUMN_TO_SEQUENCE.get(m, ""))
            w.writerow(row)

    print(f"\n  Wide output: {wide_path}")

    # --- Write LONG format (one row per UTR × miRNA, NAs dropped) ---
    long_path = f"{output_prefix}_long.tsv"
    n_long = 0
    ts_nonzero_ctx = 0
    ts_nonzero_wgt = 0

    with open(long_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        header = ["Gene_Symbol", "RefSeq_ID", "Chrom", "Start", "End", "Strand",
                  "miRNA", "miRNA_sequence", "UTR_sequence", "log2fc"]
        if has_ts:
            header.append("TS_context_pp")
            header.append("TS_weighted_context_pp")
        w.writerow(header)

        for rec in sorted(joined, key=lambda r: r["gene_symbol"]):
            for m in mirna_cols:
                fc_val = rec["fcs"].get(m)
                if fc_val is None:
                    continue

                row = [
                    rec["gene_symbol"],
                    rec["refseq_id"],
                    rec["chrom"] or "NA",
                    rec["start"] if rec["start"] is not None else "NA",
                    rec["end"] if rec["end"] is not None else "NA",
                    rec["strand"] or "NA",
                    m,
                    TARGETSCAN_COLUMN_TO_SEQUENCE.get(m, ""),
                    rec["utr_sequence"],
                    f"{fc_val:.6f}",
                ]
                if has_ts:
                    ts_ctx, ts_wgt = get_ts_scores(rec, m)
                    row.append(f"{ts_ctx:.6f}")
                    row.append(f"{ts_wgt:.6f}")
                    if ts_ctx != 0.0:
                        ts_nonzero_ctx += 1
                    if ts_wgt != 0.0:
                        ts_nonzero_wgt += 1
                w.writerow(row)
                n_long += 1

    print(f"  Long output: {long_path}")
    print(f"  Long rows:   {n_long:,}")
    if has_ts:
        print(f"  Rows with TS context++ sites:          {ts_nonzero_ctx:,} ({100*ts_nonzero_ctx/max(n_long,1):.1f}%)")
        print(f"  Rows with TS weighted context++ sites:  {ts_nonzero_wgt:,} ({100*ts_nonzero_wgt/max(n_long,1):.1f}%)")

    # --- Summary statistics ---
    if has_ts:
        print("\n  Per-miRNA coverage (context++ score):")
        for m in mirna_cols:
            n_genes = sum(1 for rec in joined if rec["fcs"].get(m) is not None)
            n_ctx = sum(1 for rec in joined
                        if rec["fcs"].get(m) is not None
                        and get_ts_scores(rec, m)[0] != 0.0)
            n_wgt = sum(1 for rec in joined
                        if rec["fcs"].get(m) is not None
                        and get_ts_scores(rec, m)[1] != 0.0)
            print(f"    {m:<22} {n_genes:>6} genes, {n_ctx:>5} ctx++ ({100*n_ctx/max(n_genes,1):.1f}%), {n_wgt:>5} weighted ({100*n_wgt/max(n_genes,1):.1f}%)")


def main():
    args = parse_args()
    utrs = load_ucsc_utrs(args.utrs)
    fc_records, mirna_cols = load_fold_changes(args.fold_change)

    # Load TS scores (keyed by both Gene Symbol and Ensembl ID)
    ts_result = load_targetscan_scores(args.scores)
    if ts_result is not None:
        ts_gs_context, ts_gs_weighted = ts_result["scores_by_gs"]
        ts_ens_context, ts_ens_weighted = ts_result["scores_by_ens"]
        utr_length_by_ens = ts_result["utr_length_by_ens"]
        all_human_gene_symbols = ts_result["all_human_gene_symbols"]
        all_human_ensembl_ids = ts_result["all_human_ensembl_ids"]
    else:
        ts_gs_context = ts_gs_weighted = ts_ens_context = ts_ens_weighted = None
        utr_length_by_ens = {}
        all_human_gene_symbols = set()
        all_human_ensembl_ids = set()

    # Load gene_info for ensembl mode
    gene_info_by_ensembl = None
    gene_info_by_symbol = None
    if args.ts_join == "ensembl":
        gene_info_by_ensembl, gene_info_by_symbol = load_ts_gene_info(args.gene_info)

    print(f"\n  TS join mode: {args.ts_join}")

    build_test_set(
        utrs, fc_records, mirna_cols,
        ts_gs_context, ts_gs_weighted,
        ts_ens_context, ts_ens_weighted,
        utr_length_by_ens,
        args.ts_join,
        gene_info_by_ensembl, gene_info_by_symbol,
        args.output,
    )

    # Save the set of all human gene symbols TS has modeled (across all miRNAs)
    if all_human_gene_symbols:
        gs_path = f"{args.output}_ts_all_genes.txt"
        with open(gs_path, "w") as f:
            for gs in sorted(all_human_gene_symbols):
                f.write(gs + "\n")
        print(f"\n  TS all-miRNA gene set: {gs_path} ({len(all_human_gene_symbols):,} genes)")

    print("\nDone.")


if __name__ == "__main__":
    main()