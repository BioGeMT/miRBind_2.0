#!/usr/bin/env python3
import argparse
import csv
import pandas as pd
import re

def deduplicate_by_prefix(headers):
    deduped = []
    seen = set()
    for header in headers:
        prefix = header.split("_")[0] if "_" in header else header
        if prefix not in seen:
            seen.add(prefix)
            deduped.append(header)
    return deduped

def parse_fasta(fasta_file):
    mapping = {}
    with open(fasta_file, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    sequence = "".join(seq_lines)
                    sequence = sequence.upper().replace('U', 'T')
                    mapping[sequence] = header
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequence = "".join(seq_lines)
            sequence = sequence.upper().replace('U', 'T')
            mapping[sequence] = header
    return mapping

def process_mapping(raw_mapping):
    if pd.isna(raw_mapping) or raw_mapping is None:
        return None, None
    first_mapping = re.split(r'\s*[;,]\s*', str(raw_mapping).strip())[0]
    cleaned = re.sub(r"_[35]p[*]*$", "", first_mapping)
    ref_id = re.sub(r"-v[0-9]+$", "", cleaned)
    return cleaned, ref_id

def process_combined(tsv_file, fasta_file, mirgenedb_file, output_file_for_standalone_run, drop_nan_families=False):
    fasta_mapping = parse_fasta(fasta_file)
    
    rows = []
    stats = {"total": 0, "kept_match": 0, "appended_no_match_or_empty_seq": 0, "multiple_ambiguous": 0}
    
    with open(tsv_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row_dict_original in reader:
            stats["total"] += 1
            row = row_dict_original.copy()

            row.setdefault('mirgenedb_name', None)

            tsv_seq_rna = row.get("noncodingRNA", "").strip()
            if not tsv_seq_rna:
                stats["appended_no_match_or_empty_seq"] +=1
                rows.append(row)
                continue

            tsv_seq_dna = tsv_seq_rna.upper().replace('U', 'T')
            
            matched_mirgenedb_name_for_row = None
            
            if tsv_seq_dna in fasta_mapping:
                matched_mirgenedb_name_for_row = fasta_mapping[tsv_seq_dna]
            else:
                possible_matches = []
                for fasta_s, header in fasta_mapping.items():
                    if fasta_s in tsv_seq_dna:
                        unmatched_len = len(tsv_seq_dna) - len(fasta_s)
                        possible_matches.append({'header': header, 'diff': unmatched_len})
                for fasta_s, header in fasta_mapping.items():
                    if tsv_seq_dna in fasta_s:
                        unmatched_len = len(fasta_s) - len(tsv_seq_dna)
                        possible_matches.append({'header': header, 'diff': unmatched_len})
                
                if possible_matches:
                    min_unmatched_diff = min(m['diff'] for m in possible_matches)
                    best_raw_matches = [m['header'] for m in possible_matches if m['diff'] == min_unmatched_diff]
                    best_deduplicated_matches = deduplicate_by_prefix(best_raw_matches)
                    
                    if len(best_deduplicated_matches) == 1:
                        matched_mirgenedb_name_for_row = best_deduplicated_matches[0]
                    else:
                        stats["multiple_ambiguous"] += 1
            
            row["mirgenedb_name"] = matched_mirgenedb_name_for_row
            if matched_mirgenedb_name_for_row:
                stats["kept_match"] += 1
            else:
                stats["appended_no_match_or_empty_seq"] +=1
            rows.append(row)
    
    df_annotations = pd.DataFrame(rows)
    
    df_mirgene_families = pd.read_csv(mirgenedb_file, sep="\t", dtype=str)
    
    mirgenedb_id_col_in_fam_file = next((c for c in df_mirgene_families.columns if c.lower() in ['mirgenedb_id', 'id']), None)
    family_col_in_fam_file = next((c for c in df_mirgene_families.columns if c.lower() == 'family'), None)
    
    # Assume columns are present
    
    if 'mirgenedb_name' not in df_annotations.columns:
        df_annotations['mirgenedb_name'] = None 
        
    annotation_results = df_annotations['mirgenedb_name'].apply(process_mapping)
    df_annotations['mirgenedb_id'] = [r[0] for r in annotation_results]
    df_annotations['mirgene_ref_for_fam_match'] = [r[1] for r in annotation_results]
    
    df_result = df_annotations.merge(
        df_mirgene_families[[mirgenedb_id_col_in_fam_file, family_col_in_fam_file]], 
        left_on="mirgene_ref_for_fam_match", 
        right_on=mirgenedb_id_col_in_fam_file, 
        how="left"
    )
    
    df_result = df_result.rename(columns={family_col_in_fam_file: "mirgenedb_fam"})
    df_result.drop(columns=["mirgene_ref_for_fam_match"], inplace=True, errors='ignore')
    if mirgenedb_id_col_in_fam_file in df_result.columns and mirgenedb_id_col_in_fam_file != 'mirgenedb_id':
         df_result.drop(columns=[mirgenedb_id_col_in_fam_file], inplace=True, errors='ignore')
    
    if 'mirgenedb_fam' in df_result.columns:
        df_result['mirgenedb_fam'] = df_result['mirgenedb_fam'].fillna('unknown')
    if 'mirgenedb_id' in df_result.columns:
        df_result['mirgenedb_id'] = df_result['mirgenedb_id'].fillna('unknown')
    if 'mirgenedb_name' in df_result.columns:
        df_result['mirgenedb_name'] = df_result['mirgenedb_name'].fillna('unknown')

    
    if output_file_for_standalone_run:
        df_result.to_csv(output_file_for_standalone_run, sep="\t", index=False, na_rep='NA')

    return df_result

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process a TSV file (with 'noncodingRNA' column), map sequences to miRNA IDs, "
            "and enrich with miRNA family data in a single workflow.\n"
            "1. Match sequences against FASTA to identify miRNAs\n"
            "2. Merge with MirGeneDB to add family information\n"
            "3. Always keep rows; label missing IDs/families/names as 'unknown'\n"
            "Rows with multiple mappings are dropped from mirgenedb_name assignment but kept in output."
        )
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file containing miRNA sequences.")
    parser.add_argument("--tsv", required=True, help="TSV file with 'noncodingRNA' column to process.")
    parser.add_argument("--mirgenedb", required=True, help="MirGeneDB ID to Family TSV file.")
    parser.add_argument("--output", required=True, help="Output TSV file name.")
    args = parser.parse_args()

    df_annotated = process_combined(args.tsv, args.fasta, args.mirgenedb, args.output, drop_nan_families=False)

    if df_annotated is not None and not df_annotated.empty and args.output != "temp_ann_module_output.tsv":
        df_annotated.to_csv(args.output, sep="\t", index=False, na_rep='NA')


if __name__ == "__main__":
    main()
