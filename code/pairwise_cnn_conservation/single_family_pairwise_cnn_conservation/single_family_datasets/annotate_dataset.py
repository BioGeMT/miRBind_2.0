#!/usr/bin/env python3
import argparse
import csv
import os
import sys
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

def parse_fasta(fasta_file): # Renamed from parse_fasta_for_annotation for internal consistency
    mapping = {}
    try:
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
                        sequence = sequence.upper().replace('U', 'T') # Ensure DNA and uppercase for keys
                        mapping[sequence] = header
                    header = line[1:].strip()
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if header is not None: # Last sequence
                sequence = "".join(seq_lines)
                sequence = sequence.upper().replace('U', 'T')
                mapping[sequence] = header
    except FileNotFoundError:
        # If called from main script, sys.exit is fine. If imported, might be too abrupt.
        # For now, keeping sys.exit as per original script's severity for this error.
        print(f"Error: FASTA file '{fasta_file}' not found.")
        sys.exit(1) # Or raise an exception to be caught by the caller
    except Exception as e:
        print(f"Error parsing FASTA file '{fasta_file}': {e}")
        sys.exit(1) # Or raise
        
    if not mapping:
        # This was sys.exit before. If imported, we might want to return empty and let caller decide.
        # However, original script considered this fatal.
        # For now, keeping the fatal exit if no sequences, as per original logic.
        # If this function is called by an importer that can handle an empty mapping,
        # this could be changed to: print warning and return empty mapping.
        print(f"Error: No sequences found in the FASTA file: {fasta_file}.")
        sys.exit(1)
    return mapping

def process_mapping(raw_mapping): # Renamed from process_mapping_for_annotation
    if pd.isna(raw_mapping) or raw_mapping is None:
        return None, None
    first_mapping = re.split(r'\s*[;,]\s*', str(raw_mapping).strip())[0]
    cleaned = re.sub(r"_[35]p[*]*$", "", first_mapping)
    ref_id = re.sub(r"-v[0-9]+$", "", cleaned)
    return cleaned, ref_id

def process_combined(tsv_file, fasta_file, mirgenedb_file, output_file_for_standalone_run, drop_nan_families=True):
    # combined workflow that processes tsv file to add mirgenedb_name column and enriches with family data
    print(f"Annotation Module - Phase 1: Processing sequences from {tsv_file}")
    
    fasta_mapping = parse_fasta(fasta_file)
    # Original script exits if fasta_mapping is empty (handled in parse_fasta now)
    
    rows = []
    stats = {"total": 0, "kept_match": 0, "appended_no_match_or_empty_seq": 0, "multiple_ambiguous": 0}
    
    try:
        with open(tsv_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile, delimiter="\t")
            if not reader.fieldnames or "noncodingRNA" not in reader.fieldnames:
                print(f"Error: '{tsv_file}' does not contain the required 'noncodingRNA' column or is empty/malformed.")
                # If imported, returning None allows caller to handle. If standalone, sys.exit is fine.
                # For consistency with original severity:
                sys.exit(1) 
            
            for row_dict_original in reader:
                stats["total"] += 1
                row = row_dict_original.copy() # Work on a copy

                # Ensure mirgenedb_name column is at least None initially for all rows
                row.setdefault('mirgenedb_name', None)

                tsv_seq_rna = row.get("noncodingRNA", "").strip()
                if not tsv_seq_rna:
                    stats["appended_no_match_or_empty_seq"] +=1
                    rows.append(row) # Keep row, but it won't get mirgenedb_name
                    continue

                # Convert TSV sequence to DNA uppercase for matching FASTA keys
                tsv_seq_dna = tsv_seq_rna.upper().replace('U', 'T')
                
                matched_mirgenedb_name_for_row = None # Reset for each row
                
                if tsv_seq_dna in fasta_mapping:
                    matched_mirgenedb_name_for_row = fasta_mapping[tsv_seq_dna]
                else:
                    possible_matches = []
                    # Substring check 1: FASTA sequence is a substring of TSV sequence.
                    for fasta_s, header in fasta_mapping.items():
                        if fasta_s in tsv_seq_dna:
                            unmatched_len = len(tsv_seq_dna) - len(fasta_s)
                            possible_matches.append({'header': header, 'diff': unmatched_len})
                    # Substring check 2: TSV sequence is a substring of FASTA sequence.
                    for fasta_s, header in fasta_mapping.items():
                        if tsv_seq_dna in fasta_s: # tsv_seq_dna already uppercased & T
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
                else: # This counts rows where no match was found OR it was ambiguous
                    stats["appended_no_match_or_empty_seq"] +=1 # Re-evaluate this counter name/logic if needed
                rows.append(row)
    except FileNotFoundError:
        print(f"Error: Input TSV '{tsv_file}' not found."); sys.exit(1)
    except Exception as e:
        print(f"Error processing TSV file '{tsv_file}': {e}"); sys.exit(1)
    
    df_annotations = pd.DataFrame(rows)
    if df_annotations.empty and stats["total"] > 0:
        print("Warning (Annotation): DataFrame is empty after sequence mapping, though input rows were processed.")

    # Phase 1 summary
    print(f"Annotation Module - Phase 1 Seq Mapping Summary:")
    print(f"  Total rows processed: {stats['total']}")
    print(f"  Rows with mirgenedb_name assigned: {stats['kept_match']}")
    print(f"  Rows without mirgenedb_name (no match/empty seq/ambiguous): {stats['appended_no_match_or_empty_seq'] + stats['multiple_ambiguous']}") # Sum of these two
    print(f"     (Specifically ambiguous & dropped from assignment: {stats['multiple_ambiguous']})")
    if 'mirgenedb_name' in df_annotations.columns:
        print(f"  Unique mature miRNA IDs (mirgenedb_name) found: {df_annotations['mirgenedb_name'].nunique(dropna=True)}")
    
    # Phase 2: Merge with MirGeneDB data
    print(f"Annotation Module - Phase 2: Merging with MirGeneDB data from {mirgenedb_file}")
    try:
        df_mirgene_families = pd.read_csv(mirgenedb_file, sep="\t", dtype=str) # Read all as str for safety
        print(f"  MirGeneDB family file columns: {df_mirgene_families.columns.tolist()}")
    except FileNotFoundError: print(f"Error reading MirGeneDB family file: {mirgenedb_file} not found."); sys.exit(1)
    except Exception as e: print(f"Error reading MirGeneDB family file '{mirgenedb_file}': {e}"); sys.exit(1)
    print(f"  Loaded {len(df_mirgene_families)} records from MirGeneDB family file.")
    
    mirgenedb_id_col_in_fam_file = next((c for c in df_mirgene_families.columns if c.lower() in ['mirgenedb_id', 'id']), None)
    family_col_in_fam_file = next((c for c in df_mirgene_families.columns if c.lower() == 'family'), None)
    
    if not mirgenedb_id_col_in_fam_file: print(f"Error: MirGeneDB ID column not found in family file. Cols: {df_mirgene_families.columns.tolist()}"); sys.exit(1)
    if not family_col_in_fam_file: print(f"Error: Family column not found in family file. Cols: {df_mirgene_families.columns.tolist()}"); sys.exit(1)
    print(f"  Using ID column: '{mirgenedb_id_col_in_fam_file}', Family column: '{family_col_in_fam_file}' from family file.")
    
    # Ensure mirgenedb_name column exists before trying to apply functions to it
    if 'mirgenedb_name' not in df_annotations.columns:
        df_annotations['mirgenedb_name'] = None 
        
    annotation_results = df_annotations['mirgenedb_name'].apply(process_mapping)
    df_annotations['mirgenedb_id'] = [r[0] for r in annotation_results] # Precursor ID for PHACTn
    df_annotations['mirgene_ref_for_fam_match'] = [r[1] for r in annotation_results] # Version-stripped for family matching
    
    df_result = df_annotations.merge(
        df_mirgene_families[[mirgenedb_id_col_in_fam_file, family_col_in_fam_file]], 
        left_on="mirgene_ref_for_fam_match", 
        right_on=mirgenedb_id_col_in_fam_file, 
        how="left"
    )
    
    df_result = df_result.rename(columns={family_col_in_fam_file: "mirgenedb_fam"})
    df_result.drop(columns=["mirgene_ref_for_fam_match"], inplace=True, errors='ignore')
    # Avoid deleting the primary ID column if it was also the merge key and named differently
    if mirgenedb_id_col_in_fam_file in df_result.columns and mirgenedb_id_col_in_fam_file != 'mirgenedb_id':
         df_result.drop(columns=[mirgenedb_id_col_in_fam_file], inplace=True, errors='ignore')
    
    # NEW: Drop rows with NaN families if requested
    if drop_nan_families:
        rows_before_drop = len(df_result)
        df_result = df_result.dropna(subset=['mirgenedb_fam'])
        rows_after_drop = len(df_result)
        rows_dropped = rows_before_drop - rows_after_drop
        
        print(f"Annotation Module - NaN Family Filtering:")
        print(f"  Rows before filtering: {rows_before_drop}")
        print(f"  Rows with NaN families (dropped): {rows_dropped}")
        print(f"  Rows after filtering: {rows_after_drop}")
        
        if rows_dropped > 0:
            print(f"  ** {rows_dropped} rows were removed due to missing family assignments **")
    
    # Final summary from this module's perspective
    print(f"Annotation Module - Final Results Summary:")
    print(f"  Total records in returned DataFrame: {len(df_result)}")
    if 'mirgenedb_id' in df_result.columns:
        print(f"  Unique precursor miRNA IDs (mirgenedb_id): {df_result['mirgenedb_id'].nunique(dropna=False)}") # count Nones too
    if 'mirgenedb_fam' in df_result.columns:
        print(f"  Rows successfully mapped to families: {df_result['mirgenedb_fam'].notna().sum()} ({df_result['mirgenedb_fam'].notna().sum()/len(df_result)*100 if len(df_result)>0 else 0:.1f}%)")
        print(f"  Unique families: {df_result['mirgenedb_fam'].nunique()}")
    
    # The original script saved to output_file here.
    # If this function is called from its own __main__, output_file_for_standalone_run will be set.
    # If imported, the calling script (1.py) passes a temp path.
    # For clarity, the main script (1.py) doesn't strictly need this module to save this temp file if it gets the df.
    # However, the original script structure had this save here. We'll keep it conditional on the argument.
    if output_file_for_standalone_run:
        try:
            df_result.to_csv(output_file_for_standalone_run, sep="\t", index=False, na_rep='NA')
            print(f"Annotation Module - Output file saved to: {output_file_for_standalone_run}")
        except Exception as e:
            print(f"Annotation Module - Error saving to {output_file_for_standalone_run}: {e}")

    return df_result  # <<< THIS IS THE CRUCIAL CHANGE FOR IMPORTING

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process a TSV file (with 'noncodingRNA' column), map sequences to miRNA IDs, "
            "and enrich with miRNA family data in a single workflow.\n"
            "1. Match sequences against FASTA to identify miRNAs\n"
            "2. Merge with MirGeneDB to add family information\n"
            "3. Optionally drop rows with missing family assignments\n"
            "Rows with multiple mappings are dropped from mirgenedb_name assignment but kept in output."
        )
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file containing miRNA sequences.")
    parser.add_argument("--tsv", required=True, help="TSV file with 'noncodingRNA' column to process.")
    parser.add_argument("--mirgenedb", required=True, help="MirGeneDB ID to Family TSV file.")
    parser.add_argument("--output", required=True, help="Output TSV file name.")
    parser.add_argument("--keep-nan-families", action="store_true", 
                       help="Keep rows with NaN/missing family assignments (default: drop them)")
    args = parser.parse_args()

    # Call process_combined and get the DataFrame
    # The drop_nan_families parameter is inverted from the --keep-nan-families flag
    drop_nan = not args.keep_nan_families
    df_annotated = process_combined(args.tsv, args.fasta, args.mirgenedb, args.output, drop_nan_families=drop_nan)
    
    # The process_combined function now handles its own saving if output_file_for_standalone_run is given.
    # So, the explicit save here might be redundant if process_combined saves to args.output.
    # For clarity, let's ensure it's saved from main() based on the df returned.
    if df_annotated is not None and not df_annotated.empty:
        if args.output != "temp_ann_module_output.tsv": # Avoid double saving the temp file if it's passed as temp.
            try:
                df_annotated.to_csv(args.output, sep="\t", index=False, na_rep='NA')
                print(f"Annotation Script (main) - Final output saved to: {args.output}")
            except Exception as e:
                 print(f"Annotation Script (main) - Error saving output: {e}")
    elif df_annotated is None:
        print("Annotation Script (main) - process_combined returned None. No output file saved by main().")
    else: # df_annotated is empty
        print("Annotation Script (main) - process_combined returned an empty DataFrame. No output file saved by main().")


if __name__ == "__main__":
    main()