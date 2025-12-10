#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

def create_output_filename(directory, family_name):
    # creates a safe filename for the family within the directory
    # Basic sanitization: replace potentially problematic characters
    safe_name = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in family_name)
    if not safe_name: # Handle cases where sanitization removes everything
        safe_name = "unnamed_family"
    return os.path.join(directory, f"{safe_name}.tsv")

def split_by_family(
    original_data_file,
    counts_file,
    output_dir,
    threshold=1000,
    family_col_name='mirgenedb_fam',
    count_col_name='count',
    progress_interval=50000
):
    # splits a large tsv file into multiple tsv files based on values in a specified family column

    print(f"--- Step 1: Reading Counts File ({counts_file}) ---")
    families_to_keep = set()
    counts_fam_col_index = -1
    counts_count_col_index = -1

    try:
        with open(counts_file, 'r', newline='', encoding='utf-8') as cf:
            reader = csv.reader(cf, delimiter='\t')
            try:
                header = next(reader)
            except StopIteration:
                print(f"Error: Counts file '{counts_file}' appears to be empty.", file=sys.stderr)
                sys.exit(1)

            try:
                counts_fam_col_index = header.index(family_col_name)
            except ValueError:
                print(f"Error: Family column '{family_col_name}' not found in counts file header.", file=sys.stderr)
                print(f"Header found: {header}", file=sys.stderr)
                sys.exit(1)
            try:
                counts_count_col_index = header.index(count_col_name)
            except ValueError:
                print(f"Error: Count column '{count_col_name}' not found in counts file header.", file=sys.stderr)
                print(f"Header found: {header}", file=sys.stderr)
                sys.exit(1)

            print(f"Found '{family_col_name}' at index {counts_fam_col_index}, '{count_col_name}' at index {counts_count_col_index} in counts file.")

            processed_count_rows = 0
            for row in reader:
                processed_count_rows += 1
                if len(row) > max(counts_fam_col_index, counts_count_col_index):
                    try:
                        family = row[counts_fam_col_index]
                        count = int(row[counts_count_col_index])
                        if count >= threshold:
                            families_to_keep.add(family)
                    except ValueError:
                         print(f"Warning: Could not convert count to int in counts file row {processed_count_rows + 1}. Skipping row: {row}", file=sys.stderr)
                    except IndexError: # Should be caught by len check, but belt-and-suspenders
                         print(f"Warning: Skipping short row {processed_count_rows + 1} in counts file: {row}", file=sys.stderr)
                else:
                     print(f"Warning: Skipping short row {processed_count_rows + 1} in counts file: {row}", file=sys.stderr)

        if not families_to_keep:
            print(f"Warning: No families found in '{counts_file}' meeting the threshold of {threshold}.", file=sys.stderr)
            print("No output files will be generated.")
            sys.exit(0)

        print(f"Identified {len(families_to_keep)} families meeting threshold >= {threshold}.")
        # print(f"Families to process: {', '.join(sorted(list(families_to_keep)))}") # Optional: print families

    except FileNotFoundError:
        print(f"Error: Counts file not found: {counts_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading counts file '{counts_file}': {e}", file=sys.stderr)
        sys.exit(1)


    print(f"\n--- Step 2: Creating Output Directory ({output_dir}) ---")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' ensured.")
    except OSError as e:
        print(f"Error: Could not create output directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)


    print(f"\n--- Step 3: Processing Original Data File ({original_data_file}) ---")
    original_header = None
    data_fam_col_index = -1
    family_data = defaultdict(list)

    try:
        with open(original_data_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')

            # read header from original data
            try:
                original_header = next(reader)
            except StopIteration:
                print(f"Error: Original data file '{original_data_file}' appears to be empty.", file=sys.stderr)
                sys.exit(1)

            # find family column index in original data
            try:
                data_fam_col_index = original_header.index(family_col_name)
                print(f"Found '{family_col_name}' at index {data_fam_col_index} in original data file.")
            except ValueError:
                print(f"Error: Family column '{family_col_name}' not found in original data file header.", file=sys.stderr)
                print(f"Header found: {original_header}", file=sys.stderr)
                sys.exit(1)

            # collect data for each family
            processed_data_rows = 0
            print("Processing rows and collecting family data...")
            for i, row in enumerate(reader):
                processed_data_rows = i + 1
                if processed_data_rows % progress_interval == 0:
                    print(f"  Processed {processed_data_rows} rows...")

                if len(row) > data_fam_col_index:
                    family = row[data_fam_col_index]
                    if family in families_to_keep:
                        family_data[family].append(row)
                else:
                     print(f"Warning: Skipping short row {processed_data_rows + 1} in original data file: {row}", file=sys.stderr)

            print(f"Finished processing {processed_data_rows} data rows.")

    except FileNotFoundError:
        print(f"Error: Original data file not found: {original_data_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing '{original_data_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # write family data to files using context managers
    print(f"\n--- Step 4: Writing Family Files ---")
    written_rows_count = {}
    
    for family, rows in family_data.items():
        output_filename = create_output_filename(output_dir, family)
        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile, delimiter='\t')
                writer.writerow(original_header)
                writer.writerows(rows)
                written_rows_count[family] = len(rows)
                print(f"  Created output file for family '{family}': {output_filename}")
        except Exception as e:
            print(f"Error: Could not write output file '{output_filename}' for family '{family}'. Error: {e}", file=sys.stderr)

    print("\n--- Processing Complete ---")
    if written_rows_count:
         print("Summary of rows written per file:")
         for family, count in sorted(written_rows_count.items()):
              print(f"  - {create_output_filename(output_dir, family)}: {count} rows")
    else:
         print("No rows were written to any output files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Splits a large TSV file into smaller TSVs based on family counts meeting a threshold.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'original_data_file',
        metavar='ORIGINAL_DATA_TSV',
        type=str,
        help='Path to the large original TSV data file.'
    )
    parser.add_argument(
        'counts_file',
        metavar='COUNTS_TSV',
        type=str,
        help='Path to the TSV file containing family counts (output from previous script).'
    )
    parser.add_argument(
        'output_dir',
        metavar='OUTPUT_DIRECTORY',
        type=str,
        help='Path to the directory where family-specific TSV files will be created.'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=1000,
        help='Minimum count for a family in the counts file to be extracted.'
    )
    parser.add_argument(
        '--family_col',
        type=str,
        default='mirgenedb_fam',
        help='Name of the column containing family names (must exist in both input files).'
    )
    parser.add_argument(
        '--count_col',
        type=str,
        default='count',
        help='Name of the column containing counts in the counts file.'
    )
    parser.add_argument(
        '--progress_interval',
        type=int,
        default=50000,
        help='Number of rows to process before printing progress update.'
    )

    args = parser.parse_args()

    split_by_family(
        args.original_data_file,
        args.counts_file,
        args.output_dir,
        args.threshold,
        args.family_col,
        args.count_col,
        args.progress_interval
    )