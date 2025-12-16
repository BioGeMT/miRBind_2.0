#!/usr/bin/env python3
"""
Zenodo Dataset Downloader

Downloads all files from a Zenodo record with progress tracking and MD5 verification.

Usage:
    python zenodo_downloader.py <record_id_or_url> [--output-dir OUTPUT_DIR] [--verify]
    
Examples:
    python zenodo_downloader.py 14734014
    python zenodo_downloader.py https://zenodo.org/records/14734014
    python zenodo_downloader.py 14734014 --output-dir ./data --verify
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


def parse_record_id(record_input: str) -> str:
    """Extract record ID from URL or return as-is if already an ID."""
    # Match patterns like zenodo.org/records/14734014 or zenodo.org/record/14734014
    match = re.search(r'zenodo\.org/records?/(\d+)', record_input)
    if match:
        return match.group(1)
    # Check if it's just a number
    if record_input.isdigit():
        return record_input
    raise ValueError(f"Could not parse record ID from: {record_input}")


def get_record_metadata(record_id: str) -> dict:
    """Fetch record metadata from Zenodo API."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching metadata from: {api_url}")
    
    try:
        with urlopen(api_url) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        if e.code == 404:
            raise ValueError(f"Record {record_id} not found on Zenodo")
        raise


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def download_file(url: str, filepath: Path, expected_size: int = None, 
                  expected_md5: str = None, verify: bool = True) -> bool:
    """
    Download a file with progress display and optional MD5 verification.
    
    Returns True if download successful and verification passed.
    """
    # Create parent directories if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists with correct size
    if filepath.exists():
        existing_size = filepath.stat().st_size
        if expected_size and existing_size == expected_size:
            print(f"  File already exists with correct size, skipping download")
            if verify and expected_md5:
                print(f"  Verifying MD5 checksum...")
                if verify_md5(filepath, expected_md5):
                    print(f"  ✓ Checksum verified")
                    return True
                else:
                    print(f"  ✗ Checksum mismatch, re-downloading...")
            else:
                return True
    
    # Download with progress
    request = Request(url, headers={'User-Agent': 'zenodo-downloader/1.0'})
    
    try:
        with urlopen(request) as response:
            total_size = int(response.headers.get('Content-Length', 0)) or expected_size or 0
            downloaded = 0
            chunk_size = 8192
            
            # Calculate MD5 while downloading if verification requested
            md5_hash = hashlib.md5() if (verify and expected_md5) else None
            
            with open(filepath, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    if md5_hash:
                        md5_hash.update(chunk)
                    downloaded += len(chunk)
                    
                    # Progress display
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r  [{bar}] {percent:5.1f}% ({format_size(downloaded)}/{format_size(total_size)})", end='', flush=True)
                    else:
                        print(f"\r  Downloaded: {format_size(downloaded)}", end='', flush=True)
            
            print()  # New line after progress bar
            
            # Verify MD5 if requested
            if md5_hash and expected_md5:
                calculated_md5 = md5_hash.hexdigest()
                if calculated_md5 == expected_md5:
                    print(f"  ✓ Checksum verified")
                    return True
                else:
                    print(f"  ✗ Checksum mismatch!")
                    print(f"    Expected: {expected_md5}")
                    print(f"    Got:      {calculated_md5}")
                    return False
            
            return True
            
    except (HTTPError, URLError) as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def decompress_gz(filepath: Path, keep_original: bool = False) -> Path:
    """
    Decompress a .gz file.
    
    Args:
        filepath: Path to the .gz file
        keep_original: If True, keep the original .gz file
        
    Returns:
        Path to the decompressed file
    """
    if not filepath.suffix == '.gz':
        return filepath
    
    output_path = filepath.with_suffix('')  # Remove .gz extension
    
    print(f"  Decompressing to {output_path.name}...")
    
    try:
        with gzip.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        if not keep_original:
            filepath.unlink()  # Delete original .gz file
            
        print(f"  ✓ Decompressed ({format_size(output_path.stat().st_size)})")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Decompression failed: {e}")
        return filepath


def download_zenodo_record(record_id: str, output_dir: Path, verify: bool = True,
                           file_filter: list = None, list_only: bool = False,
                           unzip: bool = False) -> dict:
    """
    Download all files from a Zenodo record.
    
    Args:
        record_id: Zenodo record ID
        output_dir: Directory to save files
        verify: Whether to verify MD5 checksums
        file_filter: List of filename patterns to match (partial matching)
        list_only: If True, just list files without downloading
        unzip: If True, decompress .gz files after downloading
    
    Returns a dict with download statistics.
    """
    # Get record metadata
    metadata = get_record_metadata(record_id)
    
    title = metadata.get('metadata', {}).get('title', 'Unknown')
    files = metadata.get('files', [])
    
    # Filter files if specified
    if file_filter:
        filtered_files = []
        for f in files:
            filename = f.get('key', '')
            # Check if any filter pattern matches (case-insensitive partial match)
            if any(pattern.lower() in filename.lower() for pattern in file_filter):
                filtered_files.append(f)
        files = filtered_files
        
        if not files:
            print(f"\nNo files matched the filter: {file_filter}")
            print("Use --list to see available files.")
            return {'success': 0, 'failed': 0, 'skipped': 0}
    
    print(f"\n{'='*60}")
    print(f"Record: {title}")
    print(f"DOI: {metadata.get('doi', 'N/A')}")
    print(f"Files: {len(files)}")
    
    total_size = sum(f.get('size', 0) for f in files)
    print(f"Total size: {format_size(total_size)}")
    
    if list_only:
        print(f"{'='*60}\n")
        print("Available files:")
        for i, file_info in enumerate(files, 1):
            filename = file_info.get('key', 'unknown')
            size = file_info.get('size', 0)
            print(f"  {i}. {filename} ({format_size(size)})")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Download each file
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    for i, file_info in enumerate(files, 1):
        filename = file_info.get('key', 'unknown')
        size = file_info.get('size', 0)
        checksum = file_info.get('checksum', '')
        
        # Extract MD5 from checksum (format: "md5:abc123...")
        md5 = checksum.split(':')[1] if checksum.startswith('md5:') else None
        
        # Build download URL
        download_url = file_info.get('links', {}).get('self')
        if not download_url:
            # Fallback URL construction
            download_url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
        
        filepath = output_dir / filename
        
        print(f"[{i}/{len(files)}] {filename} ({format_size(size)})")
        
        success = download_file(
            url=download_url,
            filepath=filepath,
            expected_size=size,
            expected_md5=md5,
            verify=verify
        )
        
        if success:
            stats['success'] += 1
            # Decompress if requested and file is .gz
            if unzip and filepath.suffix == '.gz':
                decompress_gz(filepath)
        else:
            stats['failed'] += 1
        
        print()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets from Zenodo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 14734014
  %(prog)s https://zenodo.org/records/14734014
  %(prog)s 14734014 --output-dir ./mirbench_data
  %(prog)s 14734014 --list                        # List available files
  %(prog)s 14734014 --file train                  # Download files containing "train"
  %(prog)s 14734014 -f test -f leftout            # Download multiple specific files
  %(prog)s 14734014 --unzip                       # Download and decompress .gz files
  %(prog)s 14734014 -f train --unzip              # Download specific file and decompress
        """
    )
    parser.add_argument(
        'record',
        help='Zenodo record ID or URL (e.g., 14734014 or https://zenodo.org/records/14734014)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./zenodo_data',
        help='Output directory for downloaded files (default: ./zenodo_data)'
    )
    parser.add_argument(
        '--file', '-f',
        action='append',
        dest='files',
        help='Download only specific file(s). Can be used multiple times. Supports partial matching.'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available files without downloading'
    )
    parser.add_argument(
        '--unzip', '-u',
        action='store_true',
        help='Decompress .gz files after downloading'
    )
    parser.add_argument(
        '--verify', '-v',
        action='store_true',
        default=True,
        help='Verify MD5 checksums after download (default: True)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip MD5 checksum verification'
    )
    
    args = parser.parse_args()
    
    try:
        record_id = parse_record_id(args.record)
        output_dir = Path(args.output_dir)
        verify = not args.no_verify
        
        print(f"Zenodo Dataset Downloader")
        print(f"Record ID: {record_id}")
        
        stats = download_zenodo_record(
            record_id, 
            output_dir, 
            verify,
            file_filter=args.files,
            list_only=args.list,
            unzip=args.unzip
        )
        
        if args.list:
            sys.exit(0)
        
        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"  Successful: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"{'='*60}")
        
        if stats['failed'] > 0:
            sys.exit(1)
            
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)


if __name__ == '__main__':
    main()