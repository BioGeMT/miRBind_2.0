#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from collections import defaultdict

def create_output_filename(directory, family_name):
    """Creates a safe filename for the family within the directory."""
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
    count_col_name='count'
):
    """
    Splits a large TSV file into multiple TSV files based on values
    in a specified family column, but only for families exceeding a
    count threshold defined in a separate counts file.

    Args:
        original_data_file (str): Path to the large input TSV data file.
        counts_file (str): Path to the TSV file containing family counts.
        output_dir (str): Path to the directory where output TSVs will be saved.
        threshold (int): Minimum count for a family to be extracted.
        family_col_name (str): Name of the column containing family names
                               (must be present in both input files).
        count_col_name (str): Name of the column containing counts in the counts_file.
    """

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
    output_writers = {} # Dictionary to store {family_name: csv_writer}
    output_files = {}   # Dictionary to store {family_name: file_handle}
    original_header = None
    data_fam_col_index = -1

    try:
        with open(original_data_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')

            # Read header from original data
            try:
                original_header = next(reader)
            except StopIteration:
                print(f"Error: Original data file '{original_data_file}' appears to be empty.", file=sys.stderr)
                sys.exit(1) # Exit if header can't be read

            # Find family column index in original data
            try:
                data_fam_col_index = original_header.index(family_col_name)
                print(f"Found '{family_col_name}' at index {data_fam_col_index} in original data file.")
            except ValueError:
                print(f"Error: Family column '{family_col_name}' not found in original data file header.", file=sys.stderr)
                print(f"Header found: {original_header}", file=sys.stderr)
                sys.exit(1)

            # Process data rows
            processed_data_rows = 0
            written_rows_count = defaultdict(int)
            print("Processing rows and writing to family-specific files...")
            for i, row in enumerate(reader):
                processed_data_rows = i + 1
                if processed_data_rows % 50000 == 0: # Progress indicator
                    print(f"  Processed {processed_data_rows} rows...")

                if len(row) > data_fam_col_index:
                    family = row[data_fam_col_index]

                    # Check if this family is one we need to keep
                    if family in families_to_keep:
                        # If we haven't opened a file for this family yet, do it now
                        if family not in output_writers:
                            output_filename = create_output_filename(output_dir, family)
                            try:
                                # Open file, create writer, write header, store them
                                outfile = open(output_filename, 'w', newline='', encoding='utf-8')
                                writer = csv.writer(outfile, delimiter='\t')
                                writer.writerow(original_header) # Write header ONCE
                                output_files[family] = outfile
                                output_writers[family] = writer
                                print(f"  Created output file for family '{family}': {output_filename}")
                            except Exception as e:
                                print(f"Error: Could not create or write header to output file '{output_filename}' for family '{family}'. Skipping this family. Error: {e}", file=sys.stderr)
                                families_to_keep.remove(family) # Stop trying to write for this family
                                if family in output_files: # Clean up if file was partially opened
                                     output_files[family].close()
                                     del output_files[family]
                                continue # Skip to next row

                        # Write the current row to the correct family file
                        try:
                           output_writers[family].writerow(row)
                           written_rows_count[family] += 1
                        except Exception as e:
                            print(f"Error writing row {processed_data_rows + 1} for family '{family}' to its file. Skipping row. Error: {e}", file=sys.stderr)
                            # Decide if you want to remove the family from processing here too
                else:
                     print(f"Warning: Skipping short row {processed_data_rows + 1} in original data file: {row}", file=sys.stderr)

            print(f"Finished processing {processed_data_rows} data rows.")

    except FileNotFoundError:
        print(f"Error: Original data file not found: {original_data_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while processing '{original_data_file}': {e}", file=sys.stderr)
        # Ensure files opened so far are closed before exiting
    finally:
        # --- Step 4: Clean up - Close all opened output files ---
        print("\n--- Step 4: Closing Output Files ---")
        closed_count = 0
        for family, f_handle in output_files.items():
            try:
                f_handle.close()
                closed_count += 1
                # print(f"  Closed file for family '{family}'. Wrote {written_rows_count[family]} rows.")
            except Exception as e:
                print(f"Warning: Error closing file for family '{family}': {e}", file=sys.stderr)
        print(f"Closed {closed_count} output files.")
        if closed_count != len(output_files):
             print(f"Warning: Expected to close {len(output_files)} files, but only closed {closed_count}.", file=sys.stderr)


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

    args = parser.parse_args()

    split_by_family(
        args.original_data_file,
        args.counts_file,
        args.output_dir,
        args.threshold,
        args.family_col,
        args.count_col
    )