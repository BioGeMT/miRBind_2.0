#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
import sys

def count_mirgen_families(input_filename, output_filename, column_name='mirgenedb_fam'):
    # reads a tsv file, counts unique values in a specified column, and writes the counts to a new tsv file
    fam_counts = Counter()
    col_index = -1

    print(f"Reading input file: {input_filename}...")
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')

            # --- Read Header and find column index ---
            try:
                header = next(reader)
            except StopIteration:
                print(f"Error: Input file '{input_filename}' appears to be empty.", file=sys.stderr)
                sys.exit(1)

            try:
                col_index = header.index(column_name)
                print(f"Found column '{column_name}' at index {col_index}.")
            except ValueError:
                print(f"Error: Column '{column_name}' not found in header.", file=sys.stderr)
                print(f"Header found: {header}", file=sys.stderr)
                sys.exit(1)

            # --- Process Data Rows ---
            row_count = 0
            for row in reader:
                row_count += 1
                # Ensure row has enough columns before accessing index
                if len(row) > col_index:
                    family_name = row[col_index]
                    # Optional: Skip empty strings if desired
                    # if family_name:
                    fam_counts[family_name] += 1
                else:
                    # Handle rows that are shorter than expected (optional)
                    print(f"Warning: Skipping short row {row_count + 1}: {row}", file=sys.stderr)

            print(f"Processed {row_count} data rows.")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{input_filename}': {e}", file=sys.stderr)
        sys.exit(1)

    if not fam_counts:
        print("Warning: No data found or counted for the specified column.", file=sys.stderr)
        # Decide if you want to create an empty output file or exit
        # Creating an empty file with header:
        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
                 writer = csv.writer(outfile, delimiter='\t')
                 writer.writerow([column_name, 'count'])
            print(f"Output file '{output_filename}' created with header only (no data counted).")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred while writing empty output file '{output_filename}': {e}", file=sys.stderr)
            sys.exit(1)


    # --- Sort Results ---
    # Sort by count (item[1]) descending, then alphabetically by family (item[0]) as a tie-breaker
    sorted_counts = sorted(fam_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    print(f"Found {len(sorted_counts)} unique values in '{column_name}'.")

    # --- Write Output TSV ---
    print(f"Writing output file: {output_filename}...")
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            # Write output header
            writer.writerow([column_name, 'count'])
            # Write data rows
            writer.writerows(sorted_counts) # More efficient for multiple rows

        print("Processing complete.")
        print(f"Output successfully written to '{output_filename}'.")

    except Exception as e:
        print(f"An error occurred while writing '{output_filename}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Counts unique values in the "mirgenedb_fam" column of a TSV file and outputs sorted counts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument(
        'input_file',
        metavar='INPUT_TSV',
        type=str,
        help='Path to the input TSV file.'
    )
    parser.add_argument(
        'output_file',
        metavar='OUTPUT_TSV',
        type=str,
        help='Path for the output TSV file (mirgenedb_fam, count).'
    )
    parser.add_argument(
        '--column',
        type=str,
        default='mirgenedb_fam',
        help='Name of the column containing the values to count.'
    )

    args = parser.parse_args()

    count_mirgen_families(args.input_file, args.output_file, args.column)